import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingGenerator:
    """TF-IDF and Sentence-BERT embedding generation."""
    
    @staticmethod
    def _sanitize_text(text):
        """Remove problematic Unicode characters.
        
        Args:
            text: Input text with potential encoding issues
            
        Returns:
            UTF-8 compatible text with errors stripped
        """
        if not isinstance(text, str):
            return ""
        try:
            return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except:
            return ""
    
    def create_tfidf_embeddings(self, product_df: pd.DataFrame, max_features: int = 5000) -> np.ndarray:
        """Generate TF-IDF embeddings.
        
        Args:
            product_df: DataFrame with 'text_features' column
            max_features: Maximum vocabulary size for vectorizer
            
        Returns:
            TF-IDF vectors (n_products, max_features)
        """
        print(f"Creating TF-IDF embeddings (max_features={max_features})...")
        
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(product_df['text_features'])
        embeddings = tfidf_matrix.toarray()
        
        print(f"TF-IDF shape: {embeddings.shape}")
        
        return embeddings
    
    def create_sentence_embeddings(self, product_df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu', batch_size: int = 1000) -> np.ndarray:
        """Generate Sentence-BERT embeddings.
        
        Args:
            product_df: DataFrame with 'text_features' column
            model_name: HuggingFace model identifier (e.g., 'all-mpnet-base-v2')
            device: 'cpu' or 'cuda' for GPU acceleration
            batch_size: Number of texts to encode per batch
            
        Returns:
            Dense embeddings (n_products, embedding_dim)
        """
        print(f"Creating Sentence-BERT embeddings (model={model_name}, device={device}, batch_size={batch_size})...")
        
        model = SentenceTransformer(model_name, device=device)
        
        embeddings = self._generate_embeddings_in_batches(product_df, model, batch_size)
        
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _generate_embeddings_in_batches(self, product_df: pd.DataFrame, model: SentenceTransformer, batch_size: int = 1000) -> np.ndarray:
        """Batch embedding generation for large datasets.
        
        Args:
            product_df: DataFrame with 'text_features' column
            model: Initialized SentenceTransformer instance
            batch_size: Number of texts to encode per batch
            
        Returns:
            Stacked embeddings from all batches (n_products, embedding_dim)
        """
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
                    texts.append("")
            except:
                texts.append("")
        
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            batch_texts = texts[i:end_idx]
            
            clean_batch = self._clean_batch_texts(batch_texts, i)
            batch_embeddings = self._encode_batch(clean_batch, model, i, end_idx)
            embeddings_list.append(batch_embeddings)
            
            if (i // batch_size) % 5 == 0:
                print(f"Processed {end_idx:,}/{len(texts):,} ({end_idx/len(texts)*100:.0f}%)")
        
        return np.vstack(embeddings_list).astype(np.float32)
    
    def _clean_batch_texts(self, batch_texts: list, start_idx: int) -> list:
        """Validate and clean batch texts.
        
        Args:
            batch_texts: Raw text strings from DataFrame
            start_idx: Starting index for error logging
            
        Returns:
            List with None values converted to empty strings
        """
        clean_batch = []
        for j, t in enumerate(batch_texts):
            if isinstance(t, str):
                clean_batch.append(t)
            else:
                clean_batch.append(str(t) if t is not None else "")
        return clean_batch
    
    def _encode_batch(self, clean_batch: list, model: SentenceTransformer, start_idx: int, end_idx: int) -> np.ndarray:
        """Encode text batch with error handling.
        
        Args:
            clean_batch: Validated text strings
            model: SentenceTransformer instance
            start_idx: Batch start index for logging
            end_idx: Batch end index for logging
            
        Returns:
            Embedding array (batch_size, embedding_dim)
        """
        try:
            return model.encode(clean_batch, show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            print(f"Error encoding batch {start_idx}-{end_idx}, processing individually...")
            return self._encode_batch_one_by_one(clean_batch, model, start_idx)
    
    def _encode_batch_one_by_one(self, clean_batch: list, model: SentenceTransformer, start_idx: int) -> np.ndarray:
        """Fallback to individual encoding on batch failure.
        
        Args:
            clean_batch: Text strings that failed batch encoding
            model: SentenceTransformer instance
            start_idx: Starting index for error logging
            
        Returns:
            Embedding array with zero vectors for failed items
        """
        batch_embeddings_list = []
        for j, text in enumerate(clean_batch):
            try:
                emb = model.encode([text], show_progress_bar=False, convert_to_numpy=True)
                batch_embeddings_list.append(emb[0])
            except:
                zero_emb = np.zeros(384, dtype=np.float32)
                batch_embeddings_list.append(zero_emb)
        return np.array(batch_embeddings_list)
