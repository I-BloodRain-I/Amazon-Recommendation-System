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
    """Product recommender using FAISS for fast similarity search.
    
    This class orchestrates the entire recommendation pipeline including data loading,
    embedding generation, FAISS indexing, similarity search, and result reranking.
    """
    
    DEFAULT_MODEL_NAME = 'all-mpnet-base-v2'
    FIRST_ELEMENT = 0
    DEFAULT_RERANK_CANDIDATES = 200
    
    def __init__(
        self, 
        reranker_model: str = 'BAAI/bge-reranker-v2-m3', 
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES, 
        device: str = 'cpu'
    ):
        """Initialize ProductRecommender with specified models and settings.
        
        Args:
            reranker_model: Name of the BGE reranker model to use
            rerank_candidates: Number of candidates to retrieve before reranking
            device: Device to use for models ('cpu', 'cuda', etc.)
        """
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
        """Load and prepare product data from CSV file.
        
        Args:
            data_path: Path to the CSV file containing product data
            max_products: Optional maximum number of products to load
            
        Returns:
            DataFrame with processed product data
        """
        self.product_df = self.data_loader.load_data(data_path, max_products)
        return self.product_df
    
    def create_tfidf_embeddings(self, max_features: int = 5000) -> np.ndarray:
        """Create TF-IDF embeddings from product text features.
        
        Args:
            max_features: Maximum number of features for TF-IDF vectorizer
            
        Returns:
            Numpy array of TF-IDF embeddings
        """
        self.embeddings = self.embedding_generator.create_tfidf_embeddings(
            self.product_df, max_features
        )
        return self.embeddings
    
    def create_sentence_embeddings(self, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """Create Sentence-BERT embeddings from product text features.
        
        Args:
            model_name: Name of the Sentence-BERT model to use
            
        Returns:
            Numpy array of sentence embeddings
        """
        self.embeddings = self.embedding_generator.create_sentence_embeddings(
            self.product_df, model_name, device=self.device
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
                            same_category_only: bool = False) -> List[Dict]:
        """
        Find similar products using FAISS index with BGE reranking
        
        Args:
            query_idx: Index of the query product
            top_k: Number of results to return
            same_category_only: Whether to filter by same category
            
        Returns:
            List of similar products with metadata
        """
        if self.faiss_manager.index is None:
            raise ValueError("FAISS index not created. Call create_faiss_index first.")
        
        initial_k = self.rerank_candidates
        results = self.faiss_manager.search_by_index(query_idx, initial_k, same_category_only)
        
        query_product = self.product_df.iloc[query_idx].to_dict()
        results = self.reranker.rerank_products(query_product, results, top_k)
        
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
    
    def display_recommendations(self, query_idx: int, similar_products: List[Dict], method: str):
        """Display product recommendations in a formatted output.
        
        Args:
            query_idx: Index of the query product
            similar_products: List of similar products with metadata
            method: Name of the method used for recommendations
        """
        query_product = self.product_df.iloc[query_idx].to_dict()
        self.display.display_recommendations(query_product, similar_products, method)
    
    def search_text(self, query_text: str, top_k: int = 5, 
                   category_filter: Optional[str] = None) -> List[Dict]:
        """
        Search using raw text query with BGE reranking
        
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
        model = SentenceTransformer(self.DEFAULT_MODEL_NAME, device=self.device)
        query_embedding = model.encode([query_text], convert_to_numpy=True)[self.FIRST_ELEMENT]
        
        initial_k = self.rerank_candidates
        results = self.faiss_manager.search(query_embedding, initial_k, category_filter)
        
        candidate_texts = [self.reranker._prepare_product_text(prod) for prod in results]
        rerank_results = self.reranker.rerank(query_text, candidate_texts, top_k)
        
        reranked_products = []
        for result in rerank_results:
            idx = result['original_index']
            product = results[idx].copy()
            product['rerank_score'] = result['rerank_score']
            product['original_rank'] = product.get('rank', idx + 1)
            product['rank'] = result['rank']
            reranked_products.append(product)
        
        return reranked_products
    
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
        self.embeddings = self.faiss_manager.embeddings
    
    def get_statistics(self) -> Dict:
        """Get statistics about the FAISS index.
        
        Returns:
            Dictionary containing index statistics
        """
        return self.faiss_manager.get_statistics()
    
    def evaluate_recommendations(
        self, 
        categories_column: str = 'categories',
        sample_ratio: float = 0.2,
        max_category_level: int = 2,
        top_k: int = 10,
        random_state: int = 42
    ) -> Dict[str, float]:
        """
        Evaluate recommendation quality using recall and precision metrics
        
        Args:
            categories_column: Name of the column containing categories
            sample_ratio: Ratio of data to use for evaluation (default 0.2 = 20%)
            max_category_level: Maximum category hierarchy level to consider
            top_k: Number of similar products to retrieve
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with recall, precision, and F1 score metrics
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available.")
        
        if self.product_df is None:
            raise ValueError("No product data available.")
        
        return self.similarity_computer.compute_metrics(
            embeddings=self.embeddings,
            product_df=self.product_df,
            categories_column=categories_column,
            sample_ratio=sample_ratio,
            max_category_level=max_category_level,
            top_k=top_k,
            random_state=random_state,
            reranker=self.reranker,
            rerank_candidates=self.rerank_candidates
        )
    
    def check_and_add_new_products(
        self,
        data_path: Path,
        model_name: str = DEFAULT_MODEL_NAME,
        max_products: Optional[int] = None
    ) -> Dict:
        """
        Check for new products not in the index and add them
        
        Args:
            data_path: Path to the data file
            model_name: Sentence-BERT model name to use for embeddings
            max_products: Maximum number of products to load from data
            
        Returns:
            Dictionary with statistics about added products
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
            new_products_df, model
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
