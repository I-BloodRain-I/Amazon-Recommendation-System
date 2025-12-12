import ast
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class ProductRecommender:
    """Simple product recommender using text embeddings and cosine similarity"""
    
    def __init__(self):
        self.product_df = None
        self.embeddings = None
        self.similarity_matrix = None
    
    @staticmethod
    def _parse_details(details_value):
        """Parse details column from string representation of dict to key:value pairs"""
        if pd.isna(details_value) or details_value == '':
            return ''
        try:
            if isinstance(details_value, str):
                details_dict = ast.literal_eval(details_value)
                if isinstance(details_dict, dict):
                    return ' '.join(f"{k}:{v}" for k, v in details_dict.items() if v)
            return str(details_value)
        except:
            return str(details_value)
    
    @staticmethod
    def _parse_categories(categories_value):
        """Parse categories column from string representation of list/array to actual values"""
        if pd.isna(categories_value) or categories_value == '':
            return ''
        try:
            if isinstance(categories_value, str):
                categories_list = ast.literal_eval(categories_value)
                if isinstance(categories_list, (list, tuple)):
                    return ' '.join(str(cat) for cat in categories_list if cat)
            return str(categories_value)
        except:
            return str(categories_value)
        
    def load_data(self, data_path: Path, max_products: Optional[int] = None) -> pd.DataFrame:
        """Load and prepare product data"""
        cache_path = Path(data_path).parent / f'processed_products_{max_products if max_products else "all"}.pkl'
        
        if cache_path.exists():
            print(f"\nLoading cached processed data from {cache_path}...")
            self.product_df = pd.read_pickle(cache_path)
            print(f"Loaded {len(self.product_df):,} unique products from cache")
            return self.product_df
        
        print(f"\nLoading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} products")
        
        df['details_parsed'] = df['details'].apply(self._parse_details)
        df['categories_parsed'] = df['categories'].apply(self._parse_categories)
        
        df['text_features'] = (
            df['title'].fillna('') + ' ' + 
            df['details_parsed'].fillna('') + ' ' +
            df['categories_parsed'].fillna('')
        ).astype(str)
        
        self.product_df = df.groupby('parent_asin').agg({
            'title': 'first',
            'main_category': 'first',
            'average_rating': 'first',
            'rating_number': 'first',
            'text_features': 'first'
        }).reset_index()
        
        self.product_df['text_features'] = self.product_df['text_features'].astype(str)
        
        if max_products and len(self.product_df) > max_products:
            print(f"Limiting to {max_products:,} products (from {len(self.product_df):,})")
            self.product_df = self.product_df.sample(n=max_products, random_state=42).reset_index(drop=True)
        
        print(f"Prepared {len(self.product_df):,} unique products")
        return self.product_df
    
    def create_tfidf_embeddings(self, max_features: int = 5000) -> np.ndarray:
        """Create TF-IDF embeddings"""
        print(f"\nCreating TF-IDF embeddings (max_features={max_features})...")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(self.product_df['text_features'])
        self.embeddings = tfidf_matrix.toarray()
        
        print(f"TF-IDF shape: {self.embeddings.shape}")
        print(f"Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2%}")
        
        return self.embeddings
    
    @staticmethod
    def _sanitize_text(text):
        """Remove problematic Unicode characters"""
        if not isinstance(text, str):
            return ""
        try:
            return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except:
            return ""
    
    def create_sentence_embeddings(self, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """Create Sentence-BERT embeddings"""
        print(f"\nCreating Sentence-BERT embeddings (model={model_name})...")
        
        model = SentenceTransformer(model_name)
        
        print(f"\nCalculating token statistics...")
        tokenizer = model.tokenizer
        token_counts = []
        token_counts_original = []
        
        for text in self.product_df['text_features']:
            try:
                if pd.notna(text) and isinstance(text, str) and text.strip():
                    encoded_original = tokenizer(text, add_special_tokens=True, truncation=False, return_tensors=None)
                    token_counts_original.append(len(encoded_original['input_ids']))
                    
                    encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=256, return_tensors=None)
                    token_counts.append(len(encoded['input_ids']))
                else:
                    token_counts_original.append(0)
                    token_counts.append(0)
            except:
                token_counts_original.append(0)
                token_counts.append(0)
        
        token_counts = np.array(token_counts)
        token_counts_original = np.array(token_counts_original)
        
        print(f"\nToken Count Statistics (before truncation):")
        print(f"  Min:    {int(token_counts_original.min()):,} tokens")
        print(f"  Max:    {int(token_counts_original.max()):,} tokens")
        print(f"  Mean:   {float(token_counts_original.mean()):.2f} tokens")
        print(f"  Median: {float(np.median(token_counts_original)):.2f} tokens")
        print(f"  Std:    {float(token_counts_original.std()):.2f} tokens")
        print(f"  Texts exceeding 256 tokens: {(token_counts_original > 256).sum():,} ({(token_counts_original > 256).sum()/len(token_counts_original)*100:.1f}%)")
        
        print(f"\nToken Count Statistics (after truncation to 256):")
        print(f"  Min:    {int(token_counts.min()):,} tokens")
        print(f"  Max:    {int(token_counts.max()):,} tokens")
        print(f"  Mean:   {float(token_counts.mean()):.2f} tokens")
        print(f"  Median: {float(np.median(token_counts)):.2f} tokens")
        print(f"  Std:    {float(token_counts.std()):.2f} tokens")
        
        print(f"\nGenerating embeddings in batches...")
        batch_size = 1000
        embeddings_list = []
        
        texts = []
        for idx, text in enumerate(self.product_df['text_features']):
            try:
                if pd.isna(text) or text is None:
                    texts.append("")
                elif isinstance(text, str):
                    texts.append(self._sanitize_text(text))
                elif isinstance(text, (int, float)):
                    texts.append(str(text))
                else:
                    print(f"Warning: Skipping invalid text at index {idx}, type: {type(text)}")
                    texts.append("")
            except Exception as e:
                print(f"Warning: Error processing text at index {idx}: {e}")
                texts.append("")
        
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[i:end_idx]
            
            clean_batch = []
            for j, t in enumerate(batch_texts):
                if isinstance(t, str):
                    clean_batch.append(t)
                else:
                    global_idx = i + j
                    print(f"Warning: Converting non-string at index {global_idx}, type: {type(t)}, value: {repr(t)[:100]}")
                    clean_batch.append(str(t) if t is not None else "")
            
            try:
                batch_embeddings = model.encode(clean_batch, show_progress_bar=False, convert_to_numpy=True)
                embeddings_list.append(batch_embeddings)
            except Exception as e:
                print(f"Error encoding batch {i}-{end_idx}: {e}")
                print(f"Trying to process items one by one to find problematic text...")
                batch_embeddings_list = []
                for j, text in enumerate(clean_batch):
                    try:
                        emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
                        batch_embeddings_list.append(emb[0])
                    except Exception as e2:
                        global_idx = i + j
                        print(f"Failed at index {global_idx}: {repr(text)[:200]}")
                        print(f"Error: {e2}")
                        zero_emb = np.zeros(384, dtype=np.float32)
                        batch_embeddings_list.append(zero_emb)
                batch_embeddings = np.array(batch_embeddings_list)
                embeddings_list.append(batch_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {end_idx:,} / {len(texts):,} ({end_idx/len(texts)*100:.1f}%)")
        
        self.embeddings = np.vstack(embeddings_list).astype(np.float32)
        print(f"Embeddings shape: {self.embeddings.shape}")
        
        return self.embeddings
    
    def compute_similarity(self) -> np.ndarray:
        """Compute cosine similarity matrix only within same category"""
        print("\nComputing cosine similarity matrix (within same category only)...")
        
        n = len(self.embeddings)
        self.similarity_matrix = np.full((n, n), -1.0, dtype=np.float32)
        
        categories = self.product_df['main_category'].fillna('Unknown').values
        unique_categories = np.unique(categories)
        
        print(f"Processing {len(unique_categories)} unique categories...")
        
        for category in unique_categories:
            cat_indices = np.where(categories == category)[0]
            
            if len(cat_indices) > 1:
                cat_embeddings = self.embeddings[cat_indices]
                cat_similarity = cosine_similarity(cat_embeddings)
                
                for i, idx_i in enumerate(cat_indices):
                    for j, idx_j in enumerate(cat_indices):
                        self.similarity_matrix[idx_i, idx_j] = cat_similarity[i, j]
        
        valid_mask = (self.similarity_matrix >= 0) & (~np.eye(n, dtype=bool))
        similarities = self.similarity_matrix[valid_mask]
        
        if len(similarities) > 0:
            print(f"Similarity statistics (same category only):")
            print(f"  Mean: {np.mean(similarities):.4f}")
            print(f"  Std:  {np.std(similarities):.4f}")
            print(f"  Min:  {np.min(similarities):.4f}")
            print(f"  Max:  {np.max(similarities):.4f}")
            print(f"  Valid comparisons: {len(similarities):,}")
        else:
            print("Warning: No valid similarities computed (all products in different categories)")
        
        return self.similarity_matrix
    
    def find_similar_products(self, query_idx: int, top_k: int = 5) -> List[Dict]:
        """Find top-k most similar products"""
        similarities = self.similarity_matrix[query_idx].copy()
        similarities[query_idx] = -1
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            product = self.product_df.iloc[idx]
            results.append({
                'rank': rank,
                'asin': product['parent_asin'],
                'title': product['title'],
                'category': product['main_category'],
                'rating': product['average_rating'],
                'rating_number': product['rating_number'],
                'similarity': similarities[idx]
            })
        
        return results
    
    def display_recommendations(self, query_idx: int, similar_products: List[Dict], method: str):
        """Display product recommendations"""
        query_product = self.product_df.iloc[query_idx]
        
        print(f"\n{'='*80}")
        print(f"Method: {method}")
        print(f"{'='*80}")
        print(f"\nQuery Product:")
        print(f"  ASIN:     {query_product['parent_asin']}")
        print(f"  Title:    {query_product['title'][:80]}...")
        print(f"  Category: {query_product['main_category']}")
        print(f"  Rating:   {query_product['average_rating']:.1f} ({query_product['rating_number']} reviews)")
        print(f"\nTop {len(similar_products)} Similar Products:")
        
        for prod in similar_products:
            print(f"\n  {prod['rank']}. Similarity: {prod['similarity']:.4f}")
            print(f"     ASIN:     {prod['asin']}")
            print(f"     Title:    {prod['title'][:80]}...")
            print(f"     Category: {prod['category']}")
            print(f"     Rating:   {prod['rating']:.1f} ({prod['rating_number']} reviews)")
    