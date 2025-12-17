from typing import Dict, Optional, TYPE_CHECKING

from src.similarity_computer import SimilarityComputer

if TYPE_CHECKING:
    from src.recommender import ProductRecommender


class RecommenderEvaluator:
    """Evaluates recommendation system quality using category-based metrics."""
    
    __slots__ = ('_recommender', '_similarity_computer')
    
    def __init__(self, recommender: 'ProductRecommender'):
        """Initialize evaluator with a recommender instance.
        
        Args:
            recommender: ProductRecommender to evaluate
        """
        self._recommender = recommender
        self._similarity_computer = SimilarityComputer()
    
    def evaluate(
        self,
        categories_column: str = 'categories',
        sample_ratio: float = 0.2,
        max_category_level: int = 2,
        top_k: int = 10,
        random_state: int = 42,
        rerank_candidates: Optional[int] = None,
        rerank_batch_size: Optional[int] = 32
    ) -> Dict[str, float]:
        """Evaluate recommendations using recall/precision metrics.
        
        Args:
            categories_column: Column with hierarchical categories
            sample_ratio: Fraction of products to evaluate (0.0-1.0)
            max_category_level: Category depth for matching
            top_k: Number of recommendations per query
            random_state: Seed for reproducibility
            rerank_candidates: Override default rerank candidates (reduce for speed)
            rerank_batch_size: Batch size for reranking (increase for speed)
            
        Returns:
            Dict with 'recall', 'precision', 'f1_score', 'valid_samples'
            
        Raises:
            ValueError: If recommender has no embeddings or product data
        """
        embeddings = self._recommender.embeddings
        product_df = self._recommender.product_df
        
        if embeddings is None:
            raise ValueError("Recommender has no embeddings. Load or create embeddings first.")
        
        if product_df is None:
            raise ValueError("Recommender has no product data. Load data first.")
        
        if rerank_candidates is None:
            rerank_candidates = self._recommender.rerank_candidates
        
        return self._similarity_computer.compute_metrics(
            embeddings=embeddings,
            product_df=product_df,
            categories_column=categories_column,
            sample_ratio=sample_ratio,
            max_category_level=max_category_level,
            top_k=top_k,
            random_state=random_state,
            reranker=self._recommender.reranker,
            rerank_candidates=rerank_candidates,
            rerank_batch_size=rerank_batch_size,
            rating_filter_ratio=self._recommender.rating_filter_ratio
        )
