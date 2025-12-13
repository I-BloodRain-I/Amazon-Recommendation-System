from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import DataLoader
from embedding_generator import EmbeddingGenerator
from recommendation_display import RecommendationDisplay
from faiss_manager import FAISSManager


class ProductRecommender:
    """Product recommender using FAISS for fast similarity search"""
    
    DEFAULT_MODEL_NAME = 'all-mpnet-base-v2'
    FIRST_ELEMENT = 0
    
    def __init__(self):
        self.product_df = None
        self.embeddings = None
        
        self.data_loader = DataLoader()
        self.embedding_generator = EmbeddingGenerator()
        self.display = RecommendationDisplay()
        self.faiss_manager = FAISSManager()
    
    def load_data(self, data_path: Path, max_products: Optional[int] = None) -> pd.DataFrame:
        """Load and prepare product data"""
        self.product_df = self.data_loader.load_data(data_path, max_products)
        return self.product_df
    
    def create_tfidf_embeddings(self, max_features: int = 5000) -> np.ndarray:
        """Create TF-IDF embeddings"""
        self.embeddings = self.embedding_generator.create_tfidf_embeddings(
            self.product_df, max_features
        )
        return self.embeddings
    
    def create_sentence_embeddings(self, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """Create Sentence-BERT embeddings"""
        self.embeddings = self.embedding_generator.create_sentence_embeddings(
            self.product_df, model_name
        )
        return self.embeddings
    
    def create_faiss_index(self, index_type: str = 'flatip', use_gpu: bool = False) -> None:
        """
        Create FAISS index from embeddings with metadata
        
        Args:
            index_type: Type of FAISS index ('flatl2', 'flatip', 'ivfflat', 'hnsw')
            use_gpu: Whether to use GPU for indexing
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Create embeddings first.")
        
        self.faiss_manager.create_index(self.embeddings, self.product_df, index_type, use_gpu)
    
    def find_similar_products(self, query_idx: int, top_k: int = 5, 
                            same_category_only: bool = True) -> List[Dict]:
        """
        Find similar products using FAISS index
        
        Args:
            query_idx: Index of the query product
            top_k: Number of results to return
            same_category_only: Whether to filter by same category
            
        Returns:
            List of similar products with metadata
        """
        if self.faiss_manager.index is None:
            raise ValueError("FAISS index not created. Call create_faiss_index first.")
        
        results = self.faiss_manager.search_by_index(query_idx, top_k, same_category_only)
        
        return [{
            'rank': result['rank'],
            'asin': result['parent_asin'],
            'title': result['title'],
            'category': result['main_category'],
            'rating': result['average_rating'],
            'rating_number': result['rating_number'],
            'similarity': result['similarity']
        } for result in results]
    
    def display_recommendations(self, query_idx: int, similar_products: List[Dict], method: str):
        """Display product recommendations"""
        query_product = self.product_df.iloc[query_idx].to_dict()
        self.display.display_recommendations(query_product, similar_products, method)
    
    def search_text(self, query_text: str, top_k: int = 5, 
                   category_filter: Optional[str] = None) -> List[Dict]:
        """
        Search using raw text query
        
        Args:
            query_text: Text query to search for
            top_k: Number of results to return
            category_filter: Optional category to filter results
            
        Returns:
            List of similar products with metadata
        """
        if self.faiss_manager.index is None:
            raise ValueError("FAISS index not created. Call create_faiss_index first.")
        
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(self.DEFAULT_MODEL_NAME)
        query_embedding = model.encode([query_text], convert_to_numpy=True)[self.FIRST_ELEMENT]
        
        return self.faiss_manager.search(query_embedding, top_k, category_filter)
    
    def save_index(self, save_path: Path) -> None:
        """
        Save FAISS index and metadata to disk
        
        Args:
            save_path: Path to save the index
        """
        if self.faiss_manager.index is None:
            raise ValueError("FAISS index not created. Call create_faiss_index first.")
        
        self.faiss_manager.save(save_path)
    
    def load_index(self, load_path: Path) -> None:
        """
        Load FAISS index and metadata from disk
        
        Args:
            load_path: Path to load the index from
        """
        self.faiss_manager.load(load_path)
        self.product_df = pd.DataFrame(self.faiss_manager.metadata)
    
    def get_statistics(self) -> Dict:
        """Get statistics about the FAISS index"""
        return self.faiss_manager.get_statistics()
