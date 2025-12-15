import numpy as np
import random
import torch
from pathlib import Path

from src.recommender import ProductRecommender

DATA_DIR = Path(__file__).parent / 'data'
DATABASE_DIR = Path(__file__).parent / 'database'
REVIEWS_FILE = 'reviews.csv'
INDEX_NAME = 'faiss_index'
INDEX_EXT = '.index'
METADATA_EXT = '.pkl'

RANDOM_SEED = 42
MAX_PRODUCTS = 1001
SBERT_MODEL_NAME = 'all-mpnet-base-v2'
RERANKER_MODEL_NAME = 'BAAI/bge-reranker-v2-m3'
RERANK_CANDIDATES = 200
INDEX_TYPE = 'flatip'
USE_CUDA = True
TOP_K = 5
NUM_SAMPLES = 3


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility.
    
    Args:
        seed: Seed for random, numpy, and torch
    """
    random.seed(seed)
    np.random.seed(seed)


def main():
    """Train and evaluate recommendation system."""
    set_seed(RANDOM_SEED)
    
    device = 'cuda' if USE_CUDA and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_path = DATA_DIR / 'raw' / REVIEWS_FILE
    faiss_path = DATABASE_DIR / INDEX_NAME
    recommender = ProductRecommender(
        reranker_model=RERANKER_MODEL_NAME,
        rerank_candidates=RERANK_CANDIDATES,
        device=device
    )
    
    if (faiss_path.with_suffix(INDEX_EXT).exists() and 
        faiss_path.with_suffix(METADATA_EXT).exists()):
        print("Loading existing FAISS index...")
        recommender.load_index(faiss_path)
        
        stats = recommender.check_and_add_new_products(
            data_path=data_path,
            model_name=SBERT_MODEL_NAME,
            max_products=MAX_PRODUCTS
        )
        
        if stats['new_products_added'] > 0:
            recommender.save_index(faiss_path)
    else:
        print("Building new FAISS index...")
        recommender.load_data(data_path, max_products=MAX_PRODUCTS)
        recommender.create_sentence_embeddings(model_name=SBERT_MODEL_NAME)
        use_gpu_for_faiss = USE_CUDA and torch.cuda.is_available()
        recommender.create_faiss_index(index_type=INDEX_TYPE, use_gpu=use_gpu_for_faiss)
        recommender.save_index(faiss_path)
    
    random_indices = np.random.choice(len(recommender.product_df), size=NUM_SAMPLES, replace=False)
    
    print(f"Testing similarity search (BGE Reranker: top {RERANK_CANDIDATES} -> top {TOP_K})...")
    for idx in random_indices:
        similar_products = recommender.find_similar_products(idx, top_k=TOP_K)
        recommender.display_recommendations(idx, similar_products, "FAISS")
    
    stats = recommender.get_statistics()
    print(f"Index stats: {stats['total_vectors']} vectors, dim={stats['dimension']}, type={stats['index_type']}")
    
    print("Evaluating recommendation quality...")
    metrics = recommender.evaluate_recommendations(
        categories_column='categories',
        sample_ratio=1.0,
        max_category_level=6,
        top_k=10,
        random_state=RANDOM_SEED
    )

if __name__ == '__main__':
    main()
