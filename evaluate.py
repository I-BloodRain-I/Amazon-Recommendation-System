import argparse
import numpy as np
import random
import torch
from pathlib import Path

from src.recommender import ProductRecommender
from src.evaluator import RecommenderEvaluator

DATA_DIR = Path(__file__).parent / 'data'
DATABASE_DIR = Path(__file__).parent / 'database'
REVIEWS_FILE = 'reviews.csv'
INDEX_NAME = 'faiss_index'
INDEX_EXT = '.index'
METADATA_EXT = '.pkl'

DEFAULT_SEED = 42
DEFAULT_RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
DEFAULT_RERANK_CANDIDATES = 50
DEFAULT_RERANK_BATCH_SIZE = 64
DEFAULT_SAMPLE_RATIO = 1.0
DEFAULT_MAX_CATEGORY_LEVEL = 6
DEFAULT_TOP_K = 10


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
        description='Evaluate product recommendation system quality',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--reranker-model', type=str, default=DEFAULT_RERANKER_MODEL,
                        help='BGE reranker model name')
    parser.add_argument('--rerank-candidates', type=int, default=DEFAULT_RERANK_CANDIDATES,
                        help='Number of candidates to rerank')
    parser.add_argument('--rerank-batch-size', type=int, default=DEFAULT_RERANK_BATCH_SIZE,
                        help='Batch size for reranking')
    parser.add_argument('--sample-ratio', type=float, default=DEFAULT_SAMPLE_RATIO,
                        help='Fraction of products to evaluate (0.0-1.0)')
    parser.add_argument('--max-category-level', type=int, default=DEFAULT_MAX_CATEGORY_LEVEL,
                        help='Maximum category hierarchy depth')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                        help='Number of recommendations to evaluate per query')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    return parser.parse_args()


def main():
    """Evaluate recommendation system quality."""
    args = parse_args()
    set_seed(args.seed)
    
    use_cuda = not args.no_cuda
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    faiss_path = DATABASE_DIR / INDEX_NAME
    
    if not (faiss_path.with_suffix(INDEX_EXT).exists() and 
            faiss_path.with_suffix(METADATA_EXT).exists()):
        print("Error: FAISS index not found. Run predict.py first to build the index.")
        return
    
    print("Loading FAISS index for evaluation...")
    recommender = ProductRecommender(
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        device=device
    )
    recommender.load_index(faiss_path)
    
    stats = recommender.get_statistics()
    print(f"Index loaded: {stats['total_vectors']} vectors, dim={stats['dimension']}, type={stats['index_type']}")
    
    print("\nEvaluating recommendation quality...")
    print(f"Settings: sample_ratio={args.sample_ratio}, max_category_level={args.max_category_level}, top_k={args.top_k}")
    print(f"Reranking: {args.rerank_candidates} candidates -> top {args.top_k}")
    print("-" * 70)
    
    evaluator = RecommenderEvaluator(recommender)
    metrics = evaluator.evaluate(
        categories_column='categories',
        sample_ratio=args.sample_ratio,
        max_category_level=args.max_category_level,
        top_k=args.top_k,
        random_state=args.seed,
        rerank_candidates=args.rerank_candidates,
        rerank_batch_size=args.rerank_batch_size
    )
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"F1-Score:   {metrics['f1_score']:.4f}")
    print(f"Samples:    {metrics['valid_samples']:,}")
    print("=" * 70)


if __name__ == '__main__':
    main()
