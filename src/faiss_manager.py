import numpy as np
import pandas as pd
import faiss
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class FAISSManager:
    """Manages FAISS index for efficient similarity search with metadata storage.
    
    This class handles creation, loading, saving, and searching of FAISS indices
    with associated product metadata for fast similarity search operations.
    """
    
    FLOAT_DTYPE = 'float32'
    IVF_MAX_NLIST = 100
    HNSW_M_CONNECTIONS = 32
    HNSW_EF_CONSTRUCTION = 40
    GPU_DEVICE_ID = 0
    CATEGORY_SEARCH_MULTIPLIER = 10
    INVALID_INDEX = -1
    
    def __init__(self):
        """Initialize FAISSManager with empty index and metadata."""
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
        """
        Create FAISS index from embeddings with metadata
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            product_df: DataFrame containing product metadata
            index_type: Type of FAISS index ('flatl2', 'flatip', 'ivfflat', 'hnsw')
            use_gpu: Whether to use GPU for indexing (if available)
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
        """
        Search for similar vectors in the index
        
        Args:
            query_embedding: Query vector (1D array)
            top_k: Number of results to return
            category_filter: Optional category to filter results
            
        Returns:
            List of dictionaries containing results with metadata
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
        """
        Search for similar products using an index from the dataset
        
        Args:
            query_idx: Index of the query product in the original dataset
            top_k: Number of results to return (excluding the query itself)
            same_category_only: Whether to filter by same category
            
        Returns:
            List of dictionaries containing results with metadata
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
        """
        Save FAISS index and metadata to disk
        
        Args:
            save_path: Path to save the index (without extension)
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
        """
        Load FAISS index and metadata from disk
        
        Args:
            load_path: Path to load the index from (without extension)
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
        """Get statistics about the FAISS index.
        
        Returns:
            Dictionary containing index statistics including total vectors,
            dimension, type, metadata entries, and training status
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
        """
        Get set of product IDs that are already in the index
        
        Returns:
            Set of parent_asin values currently in the index
        """
        if self.metadata is None:
            return set()
        return {item['parent_asin'] for item in self.metadata if 'parent_asin' in item}
    
    def add_products(
        self,
        new_embeddings: np.ndarray,
        new_product_df: pd.DataFrame
    ) -> None:
        """
        Add new products to existing FAISS index
        
        Args:
            new_embeddings: numpy array of new embeddings to add
            new_product_df: DataFrame containing new product metadata
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
