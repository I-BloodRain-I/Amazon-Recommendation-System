import argparse
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

DEFAULT_SEED = 42
DEFAULT_MAX_PRODUCTS = 10000
DEFAULT_SBERT_MODEL = 'all-mpnet-base-v2'
DEFAULT_RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
DEFAULT_RERANK_CANDIDATES = 50
DEFAULT_INDEX_TYPE = 'flatip'
DEFAULT_TOP_K = 5
DEFAULT_NUM_SAMPLES = 3
DEFAULT_EMBEDDING_BATCH_SIZE = 50


def set_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Seed for random, numpy, and torch
    """
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Build/load FAISS index and test product recommendations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--max-products', type=int, default=DEFAULT_MAX_PRODUCTS,
                        help='Maximum number of products to load')
    parser.add_argument('--sbert-model', type=str, default=DEFAULT_SBERT_MODEL,
                        help='Sentence-BERT model name')
    parser.add_argument('--reranker-model', type=str, default=DEFAULT_RERANKER_MODEL,
                        help='BGE reranker model name')
    parser.add_argument('--rerank-candidates', type=int, default=DEFAULT_RERANK_CANDIDATES,
                        help='Number of candidates to rerank')
    parser.add_argument('--index-type', type=str, default=DEFAULT_INDEX_TYPE,
                        choices=['flatl2', 'flatip', 'ivfflat', 'hnsw'],
                        help='FAISS index type')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                        help='Number of recommendations to return')
    parser.add_argument('--num-samples', type=int, default=DEFAULT_NUM_SAMPLES,
                        help='Number of random samples to test')
    parser.add_argument('--embedding-batch-size', type=int, default=DEFAULT_EMBEDDING_BATCH_SIZE,
                        help='Batch size for embedding generation')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    return parser.parse_args()


def main():
    """Build/load index and test recommendations."""
    args = parse_args()
    set_seed(args.seed)
    
    use_cuda = not args.no_cuda
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_path = DATA_DIR / 'raw' / REVIEWS_FILE
    faiss_path = DATABASE_DIR / INDEX_NAME
    recommender = ProductRecommender(
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        device=device
    )
    
    if (faiss_path.with_suffix(INDEX_EXT).exists() and 
        faiss_path.with_suffix(METADATA_EXT).exists()):
        print("Loading existing FAISS index...")
        recommender.load_index(faiss_path)
        
        stats = recommender.check_and_add_new_products(
            data_path=data_path,
            model_name=args.sbert_model,
            max_products=args.max_products,
            batch_size=args.embedding_batch_size
        )
        
        if stats['new_products_added'] > 0:
            print(f"Added {stats['new_products_added']:,} new products to index")
            recommender.save_index(faiss_path)
    else:
        print("Building new FAISS index...")
        recommender.load_data(data_path, max_products=args.max_products)
        recommender.create_sentence_embeddings(model_name=args.sbert_model, batch_size=args.embedding_batch_size)
        use_gpu_for_faiss = use_cuda and torch.cuda.is_available()
        recommender.create_faiss_index(index_type=args.index_type, use_gpu=use_gpu_for_faiss)
        recommender.save_index(faiss_path)
        print("Index saved successfully")
    
    stats = recommender.get_statistics()
    print(f"\nIndex stats: {stats['total_vectors']:,} vectors, dim={stats['dimension']}, type={stats['index_type']}")
    
    print(f"\nTesting similarity search (BGE Reranker: top {args.rerank_candidates} -> top {args.top_k})...")
    print("=" * 70)
    
    random_indices = np.random.choice(len(recommender.product_df), size=args.num_samples, replace=False)
    
    for idx in random_indices:
        similar_products = recommender.find_similar_products(idx, top_k=args.top_k)
        recommender.display_recommendations(idx, similar_products, "FAISS + BGE Reranker")
        print("-" * 70)
    
    print("\nRecommendation testing complete!")
    print("Run 'python evaluate.py' to evaluate system quality with metrics.")

if __name__ == '__main__':
    main()
