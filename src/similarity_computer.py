import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Set, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from reranker import BGEReranker


class SimilarityComputer:
    """Computes similarity matrices for product embeddings and evaluates recommendations.
    
    This class provides methods to compute cosine similarity between products
    and evaluate recommendation quality using recall and precision metrics.
    """
    
    def compute_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix for all products.
        
        Args:
            embeddings: Numpy array of product embeddings
            
        Returns:
            Numpy array containing pairwise cosine similarity scores
        """
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def _parse_categories(self, categories_str: str) -> List[str]:
        """Parse categories from string format to list
        
        Args:
            categories_str: String representation of categories like "['Cat1' 'Cat2' 'Cat3']"
            
        Returns:
            List of category strings
        """
        if pd.isna(categories_str) or categories_str == '':
            return []
        
        try:
            if isinstance(categories_str, str):
                categories_str = categories_str.strip()
                if categories_str.startswith('[') and categories_str.endswith(']'):
                    content = categories_str[1:-1].strip()
                    if not content:
                        return []
                    categories = [cat.strip().strip("'\"") for cat in content.split("' '") if cat.strip()]
                    if not categories:
                        categories = [cat.strip().strip("'\"") for cat in content.split('"') if cat.strip()]
                    return categories
                else:
                    try:
                        parsed = ast.literal_eval(categories_str)
                        if isinstance(parsed, (list, tuple)):
                            return [str(cat).strip() for cat in parsed if cat]
                    except:
                        pass
            return []
        except Exception as e:
            return []
    
    def _get_categories_up_to_level(self, categories: List[str], max_level: int = 2) -> Set[str]:
        """Get categories up to specified level
        
        Args:
            categories: List of category strings in hierarchical order
            max_level: Maximum level to include (1-indexed)
            
        Returns:
            Set of categories up to max_level
        """
        return set(categories[:max_level]) if categories else set()
    
    def compute_metrics(
        self, 
        embeddings: np.ndarray, 
        product_df: pd.DataFrame, 
        categories_column: str = 'categories',
        sample_ratio: float = 0.2,
        max_category_level: int = 2,
        top_k: int = 10,
        random_state: int = 42,
        reranker: Optional['BGEReranker'] = None,
        rerank_candidates: int = 200
    ) -> Dict[str, float]:
        """Compute recall and precision metrics for recommendations with optional reranking
        
        Args:
            embeddings: Product embeddings array
            product_df: DataFrame containing product information
            categories_column: Name of the column containing categories
            sample_ratio: Ratio of data to use for evaluation (default 0.2 = 20%)
            max_category_level: Maximum category hierarchy level to consider
            top_k: Number of similar products to retrieve
            random_state: Random seed for reproducibility
            reranker: Optional BGEReranker instance for reranking results
            rerank_candidates: Number of candidates to retrieve before reranking
            
        Returns:
            Dictionary with recall and precision metrics
        """
        if categories_column not in product_df.columns:
            print(f"Error: Column '{categories_column}' not found")
            return {'recall': 0.0, 'precision': 0.0, 'f1_score': 0.0, 'valid_samples': 0}
        
        max_possible_k = len(product_df) - 1
        if top_k > max_possible_k:
            top_k = max_possible_k
        
        n_samples = int(len(product_df) * sample_ratio)
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(product_df), size=n_samples, replace=False)
        
        print(f"Evaluating {n_samples:,} samples, top_k={top_k}, max_level={max_category_level}")
        
        all_recalls = []
        all_precisions = []
        valid_samples = 0
        
        similarity_matrix = cosine_similarity(embeddings)
        
        for i, idx in enumerate(sample_indices):
            if (i + 1) % 200 == 0:
                print(f"Processed {i+1}/{n_samples}")
            
            query_categories_str = product_df.iloc[idx][categories_column]
            query_categories = self._parse_categories(query_categories_str)
            query_cats_set = self._get_categories_up_to_level(query_categories, max_category_level)
            
            if not query_cats_set:
                continue
            
            similarities = similarity_matrix[idx].copy()
            similarities[idx] = -np.inf
            
            if reranker is not None:
                initial_k = min(rerank_candidates, len(product_df) - 1)
                initial_indices = np.argsort(similarities)[-initial_k:][::-1]
                
                query_product = product_df.iloc[idx].to_dict()
                candidate_products = []
                for candidate_idx in initial_indices:
                    candidate = product_df.iloc[candidate_idx].to_dict()
                    candidate['_index'] = candidate_idx
                    candidate_products.append(candidate)
                
                reranked = reranker.rerank_products(query_product, candidate_products, top_k)
                top_k_indices = np.array([prod['_index'] for prod in reranked])
            else:
                top_k_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_in_recommendations = 0
            total_relevant = 0
            
            for rec_idx in top_k_indices:
                rec_categories_str = product_df.iloc[rec_idx][categories_column]
                rec_categories = self._parse_categories(rec_categories_str)
                
                # ALL levels up to max_category_level must match (or up to available levels)
                is_relevant = True
                levels_to_compare = min(len(query_categories), len(rec_categories), max_category_level)
                
                if levels_to_compare == 0:
                    is_relevant = False
                else:
                    for level in range(levels_to_compare):
                        if query_categories[level] != rec_categories[level]:
                            is_relevant = False
                            break
                
                if is_relevant:
                    relevant_in_recommendations += 1
            
            for j in range(len(product_df)):
                if j == idx:
                    continue
                other_categories_str = product_df.iloc[j][categories_column]
                other_categories = self._parse_categories(other_categories_str)
                
                # ALL levels up to max_category_level must match (or up to available levels)
                is_match = True
                levels_to_compare = min(len(query_categories), len(other_categories), max_category_level)
                
                if levels_to_compare == 0:
                    is_match = False
                else:
                    for level in range(levels_to_compare):
                        if query_categories[level] != other_categories[level]:
                            is_match = False
                            break
                
                if is_match:
                    total_relevant += 1
            
            if total_relevant > 0:
                recall = relevant_in_recommendations / total_relevant
                precision = relevant_in_recommendations / top_k
                
                all_recalls.append(recall)
                all_precisions.append(precision)
                valid_samples += 1
        
        if valid_samples == 0:
            print("Warning: No valid samples found")
            return {'recall': 0.0, 'precision': 0.0, 'f1_score': 0.0, 'valid_samples': 0}
        
        avg_recall = np.mean(all_recalls)
        avg_precision = np.mean(all_precisions)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        print(f"Results - Samples: {valid_samples:,}, Recall: {avg_recall:.4f}, Precision: {avg_precision:.4f}, F1: {f1_score:.4f}")
        
        return {
            'recall': avg_recall,
            'precision': avg_precision,
            'f1_score': f1_score,
            'valid_samples': valid_samples
        }
