import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingGenerator:
    """Generates embeddings for product text features"""
    
    @staticmethod
    def _sanitize_text(text):
        """Remove problematic Unicode characters"""
        if not isinstance(text, str):
            return ""
        try:
            return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except:
            return ""
    
    def create_tfidf_embeddings(self, product_df: pd.DataFrame, max_features: int = 5000) -> np.ndarray:
        """Create TF-IDF embeddings"""
        print(f"\nCreating TF-IDF embeddings (max_features={max_features})...")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(product_df['text_features'])
        embeddings = tfidf_matrix.toarray()
        
        print(f"TF-IDF shape: {embeddings.shape}")
        print(f"Sparsity: {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2%}")
        
        return embeddings
    
    def create_sentence_embeddings(self, product_df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
        """Create Sentence-BERT embeddings"""
        print(f"\nCreating Sentence-BERT embeddings (model={model_name})...")
        
        model = SentenceTransformer(model_name)
        
        self._print_token_statistics(product_df, model)
        embeddings = self._generate_embeddings_in_batches(product_df, model)
        
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _print_token_statistics(self, product_df: pd.DataFrame, model: SentenceTransformer):
        """Calculate and print token statistics"""
        print(f"\nCalculating token statistics...")
        tokenizer = model.tokenizer
        token_counts = []
        token_counts_original = []
        
        for text in product_df['text_features']:
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
    
    def _generate_embeddings_in_batches(self, product_df: pd.DataFrame, model: SentenceTransformer, batch_size: int = 1000) -> np.ndarray:
        """Generate embeddings in batches to handle large datasets"""
        print(f"\nGenerating embeddings in batches...")
        embeddings_list = []
        
        texts = []
        for idx, text in enumerate(product_df['text_features']):
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
            
            clean_batch = self._clean_batch_texts(batch_texts, i)
            batch_embeddings = self._encode_batch(clean_batch, model, i, end_idx)
            embeddings_list.append(batch_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"  Processed {end_idx:,} / {len(texts):,} ({end_idx/len(texts)*100:.1f}%)")
        
        return np.vstack(embeddings_list).astype(np.float32)
    
    def _clean_batch_texts(self, batch_texts: list, start_idx: int) -> list:
        """Clean and validate batch texts"""
        clean_batch = []
        for j, t in enumerate(batch_texts):
            if isinstance(t, str):
                clean_batch.append(t)
            else:
                global_idx = start_idx + j
                print(f"Warning: Converting non-string at index {global_idx}, type: {type(t)}, value: {repr(t)[:100]}")
                clean_batch.append(str(t) if t is not None else "")
        return clean_batch
    
    def _encode_batch(self, clean_batch: list, model: SentenceTransformer, start_idx: int, end_idx: int) -> np.ndarray:
        """Encode a batch of texts with error handling"""
        try:
            return model.encode(clean_batch, show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            print(f"Error encoding batch {start_idx}-{end_idx}: {e}")
            print(f"Trying to process items one by one to find problematic text...")
            return self._encode_batch_one_by_one(clean_batch, model, start_idx)
    
    def _encode_batch_one_by_one(self, clean_batch: list, model: SentenceTransformer, start_idx: int) -> np.ndarray:
        """Encode texts one by one when batch encoding fails"""
        batch_embeddings_list = []
        for j, text in enumerate(clean_batch):
            try:
                emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
                batch_embeddings_list.append(emb[0])
            except Exception as e2:
                global_idx = start_idx + j
                print(f"Failed at index {global_idx}: {repr(text)[:200]}")
                print(f"Error: {e2}")
                zero_emb = np.zeros(384, dtype=np.float32)
                batch_embeddings_list.append(zero_emb)
        return np.array(batch_embeddings_list)
