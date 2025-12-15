import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class FAISSManager:
    """FAISS index management with metadata storage."""
    
    FLOAT_DTYPE = 'float32'
    IVF_MAX_NLIST = 100
    HNSW_M_CONNECTIONS = 32
    HNSW_EF_CONSTRUCTION = 40
    GPU_DEVICE_ID = 0
    CATEGORY_SEARCH_MULTIPLIER = 10
    INVALID_INDEX = -1
    
    def __init__(self):
        """Initialize empty FAISS index."""
        self.index = None
        self.metadata = None
        self.dimension = None
        self.index_type = None
        self.embeddings = None
        
    def create_index(
        self, 
        embeddings: np.ndarray, 
        product_df: pd.DataFrame, 
        index_type: str = 'flatl2', 
        use_gpu: bool = False
    ) -> None:
        """Create FAISS index from embeddings.
        
        Args:
            embeddings: Product vectors (n_products, embedding_dim)
            product_df: Product metadata for storage with index
            index_type: 'flatl2', 'flatip', 'ivfflat', or 'hnsw'
            use_gpu: Transfer index to GPU memory if available
        """
        print(f"Creating FAISS index (type={index_type})...")
        
        embeddings = embeddings.astype(self.FLOAT_DTYPE)
        self.dimension = embeddings.shape[1]
        self.index_type = index_type
        
        if index_type == 'flatl2':
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == 'flatip':
            self.index = faiss.IndexFlatIP(self.dimension)
            faiss.normalize_L2(embeddings)
        elif index_type == 'ivfflat':
            nlist = min(int(np.sqrt(len(embeddings))), self.IVF_MAX_NLIST)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            self.index.train(embeddings)
        elif index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(self.dimension, self.HNSW_M_CONNECTIONS)
            self.index.hnsw.efConstruction = self.HNSW_EF_CONSTRUCTION
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        if use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, self.GPU_DEVICE_ID, self.index)
            except Exception as e:
                print(f"Warning: Could not move to GPU: {e}")
        
        self.index.add(embeddings)
        self.embeddings = embeddings.copy()
        self.metadata = product_df.to_dict('records')
        
        print(f"FAISS index created: {self.index.ntotal} vectors, dim={self.dimension}")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5, 
        category_filter: Optional[str] = None
    ) -> List[Dict]:
        """Search similar vectors with optional category filter.
        
        Args:
            query_embedding: Query vector (embedding_dim,)
            top_k: Number of nearest neighbors to return
            category_filter: Only return products from this category
            
        Returns:
            Products with similarity scores, distances, and metadata
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")
        
        query_embedding = query_embedding.reshape(1, -1).astype(self.FLOAT_DTYPE)
        
        if self.index_type == 'flatip':
            faiss.normalize_L2(query_embedding)
        
        search_k = top_k * self.CATEGORY_SEARCH_MULTIPLIER if category_filter else top_k
        distances, indices = self.index.search(query_embedding, search_k)
        
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx == self.INVALID_INDEX:
                continue
                
            metadata = self.metadata[idx].copy()
            
            if category_filter and metadata.get('main_category') != category_filter:
                continue
            
            if self.index_type == 'flatip':
                similarity = dist
            else:
                similarity = 1.0 / (1.0 + dist)
            
            metadata['similarity'] = float(similarity)
            metadata['distance'] = float(dist)
            metadata['rank'] = len(results) + 1
            results.append(metadata)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def search_by_index(
        self, 
        query_idx: int, 
        top_k: int = 5, 
        same_category_only: bool = False
    ) -> List[Dict]:
        """Search similar products by dataset index.
        
        Args:
            query_idx: Index of query product in original dataset
            top_k: Number of similar products (excluding query itself)
            same_category_only: Filter results to matching category
            
        Returns:
            Similar products with scores, excluding query product
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")
        
        if query_idx < 0 or query_idx >= len(self.metadata):
            raise ValueError(f"Invalid query index: {query_idx}")
        
        query_metadata = self.metadata[query_idx]
        category_filter = query_metadata['main_category'] if same_category_only else None
        query_embedding = self.embeddings[query_idx]
        results = self.search(query_embedding, top_k + 1, category_filter)
        results = [r for r in results if r.get('parent_asin') != query_metadata['parent_asin']]
        
        return results[:top_k]
    
    def save(self, save_path: Path) -> None:
        """Save index and metadata to disk.
        
        Args:
            save_path: Base path (extensions .index and .pkl added automatically)
        """
        if self.index is None:
            raise ValueError("No index to save. Call create_index first.")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        index_path = str(save_path.with_suffix('.index'))
        faiss.write_index(self.index, index_path)
        
        metadata_path = str(save_path.with_suffix('.pkl'))
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'embeddings': self.embeddings
            }, f)
        print(f"Saved index to {index_path}")
    
    def load(self, load_path: Path) -> None:
        """Load index and metadata from disk.
        
        Args:
            load_path: Base path (looks for .index and .pkl files)
        """
        load_path = Path(load_path)
        
        index_path = str(load_path.with_suffix('.index'))
        self.index = faiss.read_index(index_path)
        
        metadata_path = str(load_path.with_suffix('.pkl'))
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
            self.embeddings = data.get('embeddings')
        print(f"Loaded index: {self.index.ntotal} vectors, dim={self.dimension}, type={self.index_type}")
    
    def get_statistics(self) -> Dict:
        """Return index statistics.
        
        Returns:
            Dict with total_vectors, dimension, index_type, metadata_entries
        """
        if self.index is None:
            return {}
        
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metadata_entries': len(self.metadata) if self.metadata else 0,
            'is_trained': self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }
    
    def get_processed_product_ids(self) -> set:
        """Return set of indexed product IDs.
        
        Returns:
            Set of parent_asin values currently in index
        """
        if self.metadata is None:
            return set()
        return {item['parent_asin'] for item in self.metadata if 'parent_asin' in item}
    
    def add_products(
        self,
        new_embeddings: np.ndarray,
        new_product_df: pd.DataFrame
    ) -> None:
        """Add new products to existing index.
        
        Args:
            new_embeddings: Vectors for new products (n_new, embedding_dim)
            new_product_df: Metadata for new products matching embeddings
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index first.")
        
        if len(new_embeddings) != len(new_product_df):
            raise ValueError(
                f"Embeddings count ({len(new_embeddings)}) must match "
                f"products count ({len(new_product_df)})"
            )
        
        if new_embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"New embeddings dimension ({new_embeddings.shape[1]}) "
                f"must match index dimension ({self.dimension})"
            )
        
        print(f"Adding {len(new_embeddings)} new products to index...")
        
        new_embeddings = new_embeddings.astype(self.FLOAT_DTYPE)
        
        if self.index_type == 'flatip':
            faiss.normalize_L2(new_embeddings)
        
        self.index.add(new_embeddings)
        
        new_metadata = new_product_df.to_dict('records')
        self.metadata.extend(new_metadata)
        
        if self.embeddings is not None:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        else:
            self.embeddings = new_embeddings.copy()
        
        print(f"Added {len(new_embeddings)} products, total: {self.index.ntotal}")
