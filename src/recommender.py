from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import DataLoader
from src.embedding_generator import EmbeddingGenerator
from src.recommendation_display import RecommendationDisplay
from src.faiss_manager import FAISSManager
from src.similarity_computer import SimilarityComputer
from src.reranker import BGEReranker


class ProductRecommender:
    """FAISS-based product recommendation system."""
    
    DEFAULT_MODEL_NAME = 'all-mpnet-base-v2'
    FIRST_ELEMENT = 0
    DEFAULT_RERANK_CANDIDATES = 200
    
    def __init__(
        self, 
        reranker_model: str = 'BAAI/bge-reranker-v2-m3', 
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES, 
        device: str = 'cpu'
    ):
        """Initialize recommender with reranker model."""
        self.product_df = None
        self.embeddings = None
        self.rerank_candidates = rerank_candidates
        self.device = device
        
        self.data_loader = DataLoader()
        self.embedding_generator = EmbeddingGenerator()
        self.display = RecommendationDisplay()
        self.faiss_manager = FAISSManager()
        self.similarity_computer = SimilarityComputer()
        self.reranker = BGEReranker(reranker_model, device=device)
    
    def load_data(self, data_path: Path, max_products: Optional[int] = None) -> pd.DataFrame:
        """Load and prepare product data.
        
        Args:
            data_path: Path to CSV with product reviews
            max_products: Limit unique products (None for all)
            
        Returns:
            DataFrame with parent_asin, title, category, text_features
        """
        self.product_df = self.data_loader.load_data(data_path, max_products)
        return self.product_df
    
    def create_tfidf_embeddings(self, max_features: int = 5000) -> np.ndarray:
        """Generate TF-IDF embeddings.
        
        Args:
            max_features: Maximum vocabulary size
            
        Returns:
            TF-IDF vectors (n_products, max_features)
        """
        self.embeddings = self.embedding_generator.create_tfidf_embeddings(
            self.product_df, max_features
        )
        return self.embeddings
    
    def create_sentence_embeddings(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 1000) -> np.ndarray:
        """Generate Sentence-BERT embeddings.
        
        Args:
            model_name: HuggingFace model (e.g., 'all-mpnet-base-v2')
            batch_size: Number of texts to encode per batch
            
        Returns:
            Dense embeddings (n_products, embedding_dim)
        """
        self.embeddings = self.embedding_generator.create_sentence_embeddings(
            self.product_df, model_name, device=self.device, batch_size=batch_size
        )
        return self.embeddings
    
    def create_faiss_index(self, index_type: str = 'flatip', use_gpu: bool = False) -> None:
        """Create FAISS index from embeddings.
        
        Args:
            index_type: 'flatl2', 'flatip', 'ivfflat', or 'hnsw'
            use_gpu: Move index to GPU if available
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Create embeddings first.")
        
        self.faiss_manager.create_index(self.embeddings, self.product_df, index_type, use_gpu)
    
    def find_similar_products(self, query_idx: int, top_k: int = 5, 
                            same_category_only: bool = False) -> List[Dict]:
        """Find similar products with rating-based filtering and BGE reranking.
        
        Args:
            query_idx: Index of query product in dataset
            top_k: Number of recommendations to return
            same_category_only: Filter to same category
            
        Returns:
            Products with rank, asin, title, similarity, rerank_score
        """
        if self.faiss_manager.index is None:
            raise ValueError("FAISS index not created. Call create_faiss_index first.")
        
        initial_k = self.rerank_candidates
        results = self.faiss_manager.search_by_index(query_idx, initial_k, same_category_only)
        
        filtered_results = self._filter_by_rating_weight(results)
        
        query_product = self.product_df.iloc[query_idx].to_dict()
        results = self.reranker.rerank_products(query_product, filtered_results, top_k)
        
        return [{
            'rank': result['rank'],
            'asin': result['parent_asin'],
            'title': result['title'],
            'category': result['main_category'],
            'rating': result['average_rating'],
            'rating_number': result['rating_number'],
            'similarity': result.get('similarity', 0.0),
            'rerank_score': result.get('rerank_score', None),
            'original_rank': result.get('original_rank', None)
        } for result in results]
    
    def _filter_by_rating_weight(self, candidates: List[Dict]) -> List[Dict]:
        """Filter candidates by rating weight and rating threshold.
        
        Args:
            candidates: List of candidate products with ratings
            
        Returns:
            Filtered list of candidates (top 10% by weight, rating > 3)
        """
        for candidate in candidates:
            avg_rating = candidate.get('average_rating', 0)
            rating_num = candidate.get('rating_number', 0)
            
            if pd.isna(avg_rating):
                avg_rating = 0
            if pd.isna(rating_num):
                rating_num = 0
            
            candidate['weight'] = float(avg_rating) * float(rating_num)
        
        candidates_sorted = sorted(candidates, key=lambda x: x['weight'], reverse=True)
        target_count = max(1, int(len(candidates) * 0.1))

        top_candidates = candidates_sorted[:target_count]
        replacement_pool = candidates_sorted[target_count:]
        
        final_candidates = []
        replacement_index = 0
        
        for candidate in top_candidates:
            if candidate.get('average_rating', 0) > 3:
                final_candidates.append(candidate)
            else:
                replaced = False
                while replacement_index < len(replacement_pool):
                    replacement = replacement_pool[replacement_index]
                    replacement_index += 1
                    if replacement.get('average_rating', 0) > 3:
                        final_candidates.append(replacement)
                        replaced = True
                        break
                
                if not replaced:
                    final_candidates.append(candidate)
        
        return final_candidates
    
    def display_recommendations(self, query_idx: int, similar_products: List[Dict], method: str):
        """Display formatted recommendations.
        
        Args:
            query_idx: Index of query product
            similar_products: List from find_similar_products()
            method: Display label (e.g., 'FAISS', 'BGE Reranked')
        """
        query_product = self.product_df.iloc[query_idx].to_dict()
        self.display.display_recommendations(query_product, similar_products, method)
    
    def search_text(self, query_text: str, top_k: int = 5, 
                   category_filter: Optional[str] = None) -> List[Dict]:
        """Search products by text query with rating filtering and reranking.
        
        Args:
            query_text: Natural language query (e.g., "wireless headphones")
            top_k: Number of results to return
            category_filter: Only search within this category
            
        Returns:
            Products with rerank_score, original_rank, and metadata
        """
        if self.faiss_manager.index is None:
            raise ValueError("FAISS index not created. Call create_faiss_index first.")
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.DEFAULT_MODEL_NAME, device=self.device)
        query_embedding = model.encode([query_text], convert_to_numpy=True)[self.FIRST_ELEMENT]
        
        initial_k = self.rerank_candidates
        results = self.faiss_manager.search(query_embedding, initial_k, category_filter)
        
        filtered_results = self._filter_by_rating_weight(results)
        
        candidate_texts = [self.reranker._prepare_product_text(prod) for prod in filtered_results]
        rerank_results = self.reranker.rerank(query_text, candidate_texts, top_k)
        
        reranked_products = []
        for result in rerank_results:
            idx = result['original_index']
            product = filtered_results[idx].copy()
            product['rerank_score'] = result['rerank_score']
            product['original_rank'] = product.get('rank', idx + 1)
            product['rank'] = result['rank']
            reranked_products.append(product)
        
        return reranked_products
    
    def save_index(self, save_path: Path) -> None:
        """Save index and metadata.
        
        Args:
            save_path: Base path without extension
        """
        if self.faiss_manager.index is None:
            raise ValueError("FAISS index not created. Call create_faiss_index first.")
        
        self.faiss_manager.save(save_path)
    
    def load_index(self, load_path: Path) -> None:
        """Load index and metadata.
        
        Args:
            load_path: Base path without extension
        """
        self.faiss_manager.load(load_path)
        self.product_df = pd.DataFrame(self.faiss_manager.metadata)
        self.embeddings = self.faiss_manager.embeddings
    
    def get_statistics(self) -> Dict:
        """Return index statistics.
        
        Returns:
            Dict with total_vectors, dimension, index_type
        """
        return self.faiss_manager.get_statistics()
    
    def check_and_add_new_products(
        self,
        data_path: Path,
        model_name: str = DEFAULT_MODEL_NAME,
        max_products: Optional[int] = None,
        batch_size: int = 1000
    ) -> Dict:
        """Add new products to existing index.
        
        Args:
            data_path: CSV file with product data
            model_name: Model for generating new embeddings
            max_products: Limit total products loaded
            batch_size: Number of texts to encode per batch
            
        Returns:
            Dict with new_products_found, new_products_added, total_products_after
        """
        if self.faiss_manager.index is None:
            raise ValueError("FAISS index not loaded. Call load_index first.")
        
        print("Checking for new products...")
        
        current_product_ids = self.faiss_manager.get_processed_product_ids()
        all_products_df = self.data_loader.load_data(data_path, max_products)
        print(f"Current: {len(current_product_ids):,}, Total: {len(all_products_df):,}")
        
        new_products_mask = ~all_products_df['parent_asin'].isin(current_product_ids)
        new_products_df = all_products_df[new_products_mask].reset_index(drop=True)
        
        if len(new_products_df) == 0:
            print("No new products found")
            return {
                'new_products_found': 0,
                'new_products_added': 0,
                'total_products_after': len(current_product_ids)
            }
        
        print(f"Found {len(new_products_df):,} new products, generating embeddings...")
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=self.device)
        new_embeddings = self.embedding_generator._generate_embeddings_in_batches(
            new_products_df, model, batch_size
        )
        
        self.faiss_manager.add_products(new_embeddings, new_products_df)
        
        self.product_df = pd.DataFrame(self.faiss_manager.metadata)
        self.embeddings = self.faiss_manager.embeddings
        
        print("New products added successfully")
        
        return {
            'new_products_found': len(new_products_df),
            'new_products_added': len(new_products_df),
            'total_products_after': self.faiss_manager.index.ntotal
        }
