import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from typing import List, Set, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from reranker import BGEReranker


class SimilarityComputer:
    """Similarity computation and recommendation evaluation."""
    
    __slots__ = ()
    
    DEFAULT_RATING_FILTER_RATIO = 0.1
    
    def compute_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise cosine similarity.
        
        Args:
            embeddings: Product embedding vectors (n_products, embedding_dim)
            
        Returns:
            Cosine similarity matrix (n_products, n_products)
        """
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    
    def _parse_categories(self, categories_str: str) -> List[str]:
        """Parse category string to list.
        
        Args:
            categories_str: String representation like "['Cat1' 'Cat2']"
            
        Returns:
            Parsed category strings in hierarchical order
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
        """Extract categories up to specified hierarchy level.
        
        Args:
            categories: Hierarchical category list from broad to specific
            max_level: Maximum depth to include (1-indexed)
            
        Returns:
            Set of categories from first N levels
        """
        return set(categories[:max_level]) if categories else set()
    
    def _filter_by_rating_weight(self, candidates: List[Dict], rating_filter_ratio: float = DEFAULT_RATING_FILTER_RATIO, top_k: int = 5) -> List[Dict]:
        """Filter candidates by rating weight and rating threshold.
        
        Args:
            candidates: List of candidate products with ratings
            rating_filter_ratio: Fraction of candidates to keep (e.g., 0.1 = top 10%)
            top_k: Minimum number of candidates to keep
            
        Returns:
            Filtered list of candidates (ensures at least top_k are available)
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
        ratio_based_count = int(len(candidates) * rating_filter_ratio)
        target_count = max(top_k, max(1, ratio_based_count))

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
    
    def _batch_rerank_for_evaluation(
        self,
        sample_indices: np.ndarray,
        embeddings_normalized: np.ndarray,
        product_df: pd.DataFrame,
        reranker: 'BGEReranker',
        rerank_candidates: int,
        top_k: int,
        batch_size: int = 32,
        rating_filter_ratio: float = DEFAULT_RATING_FILTER_RATIO
    ) -> List[np.ndarray]:
        """Batch process reranking for all evaluation samples.
        
        Args:
            sample_indices: Indices of samples to evaluate
            embeddings_normalized: Normalized embeddings (for cosine similarity)
            product_df: Product dataframe
            reranker: BGE reranker instance
            rerank_candidates: Number of candidates before reranking
            top_k: Final number of recommendations
            batch_size: Number of queries to batch together
            rating_filter_ratio: Fraction of candidates to keep after rating filtering
            
        Returns:
            List of top-k indices arrays (one per sample)
        """
        all_top_k_indices = []
        n_samples = len(sample_indices)
        initial_k = min(rerank_candidates, len(product_df) - 1)
        
        print(f"Batch reranking {n_samples:,} queries (batch_size={batch_size})...")
        
        pbar = tqdm(total=n_samples, desc="Reranking", unit="queries")
        
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_indices = sample_indices[batch_start:batch_end]
            
            query_texts = []
            candidate_texts_list = []
            candidate_idx_mapping = []
            
            for idx in batch_indices:
                similarities = np.dot(embeddings_normalized, embeddings_normalized[idx])
                similarities[idx] = -np.inf
                initial_indices = np.argsort(similarities)[-initial_k:][::-1]
                
                candidates = []
                for candidate_idx in initial_indices:
                    candidate = product_df.iloc[candidate_idx].to_dict()
                    candidate['similarity'] = float(similarities[candidate_idx])
                    candidate['_index'] = candidate_idx
                    candidates.append(candidate)
                
                filtered_candidates = self._filter_by_rating_weight(candidates, rating_filter_ratio, top_k)
                
                query_product = product_df.iloc[idx].to_dict()
                query_text = reranker._prepare_product_text(query_product)
                query_texts.append(query_text)
                
                candidate_texts = []
                candidate_indices = []
                for candidate in filtered_candidates:
                    candidate_text = reranker._prepare_product_text(candidate)
                    candidate_texts.append(candidate_text)
                    candidate_indices.append(candidate['_index'])
                
                candidate_texts_list.append(candidate_texts)
                candidate_idx_mapping.append(candidate_indices)
            
            batch_results = reranker.rerank_batch(query_texts, candidate_texts_list, top_k)
            
            for results, candidate_indices in zip(batch_results, candidate_idx_mapping):
                top_k_indices = np.array([candidate_indices[r['original_index']] for r in results])
                all_top_k_indices.append(top_k_indices)
            
            pbar.update(len(batch_indices))
        
        pbar.close()
        return all_top_k_indices
    
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
        rerank_candidates: int = 200,
        rerank_batch_size: Optional[int] = 32,
        rating_filter_ratio: float = DEFAULT_RATING_FILTER_RATIO
    ) -> Dict[str, float]:
        """Compute recall/precision metrics with optional reranking.
        
        Args:
            embeddings: Product embedding vectors (n_products, embedding_dim)
            product_df: Product metadata with categories
            categories_column: Column name containing product categories
            sample_ratio: Fraction of products to evaluate (0.0-1.0)
            max_category_level: Hierarchy depth for category matching
            top_k: Number of recommendations per query
            random_state: Seed for reproducible sampling
            reranker: BGE reranker instance for result refinement
            rerank_candidates: Initial candidates before reranking (reduce for speed)
            rerank_batch_size: Number of queries to batch for reranking (larger=faster)
            rating_filter_ratio: Fraction of candidates to keep after rating filtering (e.g., 0.1 = top 10%)
            
        Returns:
            Dict with 'recall', 'precision', 'f1_score', 'valid_samples'
        """
        if categories_column not in product_df.columns:
            print(f"Error: Column '{categories_column}' not found")
            return {'recall': 0.0, 'precision': 0.0, 'f1_score': 0.0, 'valid_samples': 0}
        
        if rerank_batch_size is None:
            rerank_batch_size = 32
        
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
        
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        if reranker is not None:
            all_top_k_indices = self._batch_rerank_for_evaluation(
                sample_indices, embeddings_normalized, product_df, reranker, 
                rerank_candidates, top_k, rerank_batch_size, rating_filter_ratio
            )
        else:
            all_top_k_indices = None
        
        print("Computing metrics...")
        pbar = tqdm(total=n_samples, desc="Evaluating", unit="products")
        
        for i, idx in enumerate(sample_indices):
            
            query_categories_str = product_df.iloc[idx][categories_column]
            query_categories = self._parse_categories(query_categories_str)
            query_cats_set = self._get_categories_up_to_level(query_categories, max_category_level)
            
            if not query_cats_set:
                continue
            
            if reranker is not None and all_top_k_indices is not None:
                top_k_indices = all_top_k_indices[i]
            else:
                similarities = np.dot(embeddings_normalized, embeddings_normalized[idx])
                similarities[idx] = -np.inf
                top_k_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_in_recommendations = 0
            total_relevant = 0
            
            for rec_idx in top_k_indices:
                rec_categories_str = product_df.iloc[rec_idx][categories_column]
                rec_categories = self._parse_categories(rec_categories_str)
                
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
            
            pbar.update(1)
        
        pbar.close()
        
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
