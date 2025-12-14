import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingGenerator:
    """Generates embeddings for product text features using TF-IDF or Sentence-BERT.
    
    This class provides methods to create embeddings from product text data
    using either TF-IDF vectorization or Sentence-BERT models.
    """
    
    @staticmethod
    def _sanitize_text(text):
        """Remove problematic Unicode characters from text.
        
        Args:
            text: Input text string to sanitize
            
        Returns:
            Sanitized text string with Unicode errors removed
        """
        if not isinstance(text, str):
            return ""
        try:
            return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        except:
            return ""
    
    def create_tfidf_embeddings(self, product_df: pd.DataFrame, max_features: int = 5000) -> np.ndarray:
        """Create TF-IDF embeddings from product text features.
        
        Args:
            product_df: DataFrame containing product data with text_features column
            max_features: Maximum number of features for TF-IDF vectorizer
            
        Returns:
            Numpy array of TF-IDF embeddings with shape (n_products, max_features)
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
    
    def create_sentence_embeddings(self, product_df: pd.DataFrame, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu') -> np.ndarray:
        """Create Sentence-BERT embeddings from product text features.
        
        Args:
            product_df: DataFrame containing product data with text_features column
            model_name: Name of the Sentence-BERT model to use
            device: Device to use for encoding ('cpu', 'cuda', etc.)
            
        Returns:
            Numpy array of sentence embeddings with shape (n_products, embedding_dim)
        """
        print(f"Creating Sentence-BERT embeddings (model={model_name}, device={device})...")
        
        model = SentenceTransformer(model_name, device=device)
        
        embeddings = self._generate_embeddings_in_batches(product_df, model)
        
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def _generate_embeddings_in_batches(self, product_df: pd.DataFrame, model: SentenceTransformer, batch_size: int = 1000) -> np.ndarray:
        """Generate embeddings in batches to handle large datasets efficiently.
        
        Args:
            product_df: DataFrame containing product data with text_features column
            model: Initialized SentenceTransformer model
            batch_size: Number of samples to process in each batch
            
        Returns:
            Numpy array of sentence embeddings
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
        """Clean and validate batch texts before encoding.
        
        Args:
            batch_texts: List of text strings to clean
            start_idx: Starting index of the batch (for error tracking)
            
        Returns:
            List of cleaned text strings
        """
        clean_batch = []
        for j, t in enumerate(batch_texts):
            if isinstance(t, str):
                clean_batch.append(t)
            else:
                clean_batch.append(str(t) if t is not None else "")
        return clean_batch
    
    def _encode_batch(self, clean_batch: list, model: SentenceTransformer, start_idx: int, end_idx: int) -> np.ndarray:
        """Encode a batch of texts with error handling.
        
        Args:
            clean_batch: List of cleaned text strings
            model: SentenceTransformer model for encoding
            start_idx: Starting index of the batch
            end_idx: Ending index of the batch
            
        Returns:
            Numpy array of encoded embeddings
        """
        try:
            return model.encode(clean_batch, show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            print(f"Error encoding batch {start_idx}-{end_idx}, processing individually...")
            return self._encode_batch_one_by_one(clean_batch, model, start_idx)
    
    def _encode_batch_one_by_one(self, clean_batch: list, model: SentenceTransformer, start_idx: int) -> np.ndarray:
        """Encode texts one by one when batch encoding fails.
        
        Args:
            clean_batch: List of cleaned text strings
            model: SentenceTransformer model for encoding
            start_idx: Starting index of the batch (for error tracking)
            
        Returns:
            Numpy array of encoded embeddings (with zeros for failed encodings)
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
