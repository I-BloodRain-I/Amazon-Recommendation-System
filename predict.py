import argparse
import numpy as np
import random
import torch

from src.recommender import ProductRecommender
from src.config import config


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
        description='Test product recommendations using pre-built FAISS index',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--seed', type=int, default=config.get('general.random_seed'),
                        help='Random seed for reproducibility')
    parser.add_argument('--reranker-model', type=str, default=config.get('models.reranker_model'),
                        help='BGE reranker model name')
    parser.add_argument('--rerank-candidates', type=int, default=config.get('recommendation.rerank_candidates'),
                        help='Number of candidates to rerank')
    parser.add_argument('--rating-filter-ratio', type=float, default=config.get('recommendation.rating_filter_ratio'),
                        help='Fraction of candidates to keep after rating filtering (e.g., 0.1 = top 10%%)')
    parser.add_argument('--top-k', type=int, default=config.get('recommendation.top_k'),
                        help='Number of recommendations to return')
    parser.add_argument('--num-samples', type=int, default=config.get('recommendation.num_samples'),
                        help='Number of random samples to test')
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
    
    faiss_path = config.database_dir / config.index_name
    recommender = ProductRecommender(
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        rating_filter_ratio=args.rating_filter_ratio,
        device=device
    )
    
    index_file = faiss_path.with_suffix(config.index_ext)
    metadata_file = faiss_path.with_suffix(config.metadata_ext)
    
    if index_file.exists() and metadata_file.exists():
        print("Loading FAISS index...")
        recommender.load_index(faiss_path)
    else:
        print("Error: FAISS index not found at", faiss_path)
        print("Please build the index first:")
        print(f"  1. Place reviews.csv in {config.data_dir / 'raw' / config.reviews_file}")
        print(f"  2. Run: python build_index.py")
        return
    
    stats = recommender.get_statistics()
    print(f"\nIndex: {stats['total_vectors']:,} vectors, dim={stats['dimension']}, type={stats['index_type']}")
    print(f"Testing {args.num_samples} samples with top_k={args.top_k}")
    
    random_indices = np.random.choice(len(recommender.product_df), size=args.num_samples, replace=False)
    
    for idx in random_indices:
        similar_products = recommender.find_similar_products(idx, top_k=args.top_k)
        recommender.display_recommendations(idx, similar_products)

if __name__ == '__main__':
    main()
