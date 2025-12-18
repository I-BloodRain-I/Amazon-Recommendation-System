"""Reranker module using BGE-reranker-v2-m3 for improved recommendation accuracy."""

import numpy as np
from typing import List, Dict
from sentence_transformers import CrossEncoder


class BGEReranker:
    """BGE-reranker-v2-m3 based product reranking.
    
    CrossEncoder model for refining similarity search results.
    """
    
    DEFAULT_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = 'cpu'):
        """Initialize BGE reranker model.
        
        Args:
            model_name: CrossEncoder model name
            device: Computation device ('cpu', 'cuda', 'cuda:0')
        """
        print(f"Loading reranker model: {model_name} on {device}")
        
        self.model = CrossEncoder(model_name, device=device)
        self.device = device
        print("Reranker loaded")
    
    def rerank(
        self, 
        query_text: str, 
        candidate_texts: List[str], 
        top_k: int = 5
    ) -> List[Dict[str, float]]:
        """Rerank candidates by relevance score.
        
        Args:
            query_text: Query string for relevance comparison
            candidate_texts: List of candidate text descriptions
            top_k: Number of top results to return
            
        Returns:
            List of dicts with 'original_index', 'rerank_score', 'rank'
        """
        if not candidate_texts:
            return []
        
        pairs = [[query_text, text] for text in candidate_texts]
        scores = self.model.predict(pairs)
        
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(ranked_indices, 1):
            results.append({
                'original_index': int(idx),
                'rerank_score': float(scores[idx]),
                'rank': rank
            })
        
        return results
    
    def rerank_batch(
        self,
        query_texts: List[str],
        candidate_texts_list: List[List[str]],
        top_k: int = 5
    ) -> List[List[Dict[str, float]]]:
        """Batch rerank multiple queries at once for better performance.
        
        Args:
            query_texts: List of query strings
            candidate_texts_list: List of candidate text lists (one per query)
            top_k: Number of top results to return per query
            
        Returns:
            List of reranked results (one per query)
        """
        if not query_texts:
            return []
        
        all_pairs = []
        pair_indices = []
        
        for query_idx, (query_text, candidate_texts) in enumerate(zip(query_texts, candidate_texts_list)):
            start_idx = len(all_pairs)
            for candidate_text in candidate_texts:
                all_pairs.append([query_text, candidate_text])
            end_idx = len(all_pairs)
            pair_indices.append((start_idx, end_idx, len(candidate_texts)))
        
        if not all_pairs:
            return [[] for _ in query_texts]
        
        all_scores = self.model.predict(all_pairs)
        
        results_list = []
        for query_idx, (start_idx, end_idx, num_candidates) in enumerate(pair_indices):
            scores = all_scores[start_idx:end_idx]
            
            ranked_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for rank, idx in enumerate(ranked_indices, 1):
                results.append({
                    'original_index': int(idx),
                    'rerank_score': float(scores[idx]),
                    'rank': rank
                })
            
            results_list.append(results)
        
        return results_list
    
    def rerank_products(
        self,
        query_product: Dict,
        candidate_products: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """Rerank products by similarity score.
        
        Args:
            query_product: Query product with title, category, features
            candidate_products: Candidate products with metadata
            top_k: Number of top results to return
            
        Returns:
            Top-k products with added 'rerank_score', 'original_rank', 'rank'
        """
        if not candidate_products:
            return []
        
        if len(candidate_products) <= top_k:
            query_text = self._prepare_product_text(query_product)
            candidate_texts = [self._prepare_product_text(prod) for prod in candidate_products]
            rerank_results = self.rerank(query_text, candidate_texts, len(candidate_products))
        else:
            query_text = self._prepare_product_text(query_product)
            candidate_texts = [self._prepare_product_text(prod) for prod in candidate_products]
            rerank_results = self.rerank(query_text, candidate_texts, top_k)
        
        reranked_products = []
        for result in rerank_results:
            idx = result['original_index']
            product = candidate_products[idx].copy()
            product['rerank_score'] = result['rerank_score']
            product['original_rank'] = product.get('rank', idx + 1)
            product['rank'] = result['rank']
            reranked_products.append(product)
        
        return reranked_products
    
    def _prepare_product_text(self, product: Dict) -> str:
        """Combine product fields into text representation.
        
        Args:
            product: Product dict with title, category, features, description
            
        Returns:
            Formatted text combining all relevant product fields
        """
        text_parts = []
        
        if 'title' in product and product['title']:
            text_parts.append(f"Title: {product['title']}")
        
        if 'main_category' in product and product['main_category']:
            text_parts.append(f"Category: {product['main_category']}")
        
        if 'categories' in product and product['categories']:
            categories = product['categories']
            if isinstance(categories, list):
                categories = ', '.join(categories)
            text_parts.append(f"Categories: {categories}")
        
        if 'features' in product and product['features']:
            features = product['features']
            if isinstance(features, list):
                features = ' '.join(features[:3])
            text_parts.append(f"Features: {features}")
        
        if 'description' in product and product['description']:
            description = product['description']
            if isinstance(description, list):
                description = ' '.join(description)
            if len(description) > 500:
                description = description[:500]
            text_parts.append(f"Description: {description}")
        
        return ' '.join(text_parts)
