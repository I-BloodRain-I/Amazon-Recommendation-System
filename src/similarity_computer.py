import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityComputer:
    """Computes similarity matrices for product embeddings"""
    
    def compute_similarity(self, embeddings: np.ndarray, product_df: pd.DataFrame) -> np.ndarray:
        """Compute cosine similarity matrix only within same category"""
        print("\nComputing cosine similarity matrix (within same category only)...")
        
        n = len(embeddings)
        similarity_matrix = np.full((n, n), -1.0, dtype=np.float32)
        
        categories = product_df['main_category'].fillna('Unknown').values
        unique_categories = np.unique(categories)
        
        print(f"Processing {len(unique_categories)} unique categories...")
        
        for category in unique_categories:
            cat_indices = np.where(categories == category)[0]
            
            if len(cat_indices) > 1:
                cat_embeddings = embeddings[cat_indices]
                cat_similarity = cosine_similarity(cat_embeddings)
                
                for i, idx_i in enumerate(cat_indices):
                    for j, idx_j in enumerate(cat_indices):
                        similarity_matrix[idx_i, idx_j] = cat_similarity[i, j]
        
        self._print_similarity_statistics(similarity_matrix, n)
        
        return similarity_matrix
    
    def _print_similarity_statistics(self, similarity_matrix: np.ndarray, n: int):
        """Print statistics about computed similarities"""
        valid_mask = (similarity_matrix >= 0) & (~np.eye(n, dtype=bool))
        similarities = similarity_matrix[valid_mask]
        
        if len(similarities) > 0:
            print(f"Similarity statistics (same category only):")
            print(f"  Mean: {np.mean(similarities):.4f}")
            print(f"  Std:  {np.std(similarities):.4f}")
            print(f"  Min:  {np.min(similarities):.4f}")
            print(f"  Max:  {np.max(similarities):.4f}")
            print(f"  Valid comparisons: {len(similarities):,}")
        else:
            print("Warning: No valid similarities computed (all products in different categories)")
