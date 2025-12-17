import argparse
import numpy as np
import random
import torch

from src.recommender import ProductRecommender
from src.evaluator import RecommenderEvaluator
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
        description='Evaluate product recommendation system quality',
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
    parser.add_argument('--rerank-batch-size', type=int, default=config.get('recommendation.rerank_batch_size'),
                        help='Batch size for reranking')
    parser.add_argument('--sample-ratio', type=float, default=config.get('evaluation.sample_ratio'),
                        help='Fraction of products to evaluate (0.0-1.0)')
    parser.add_argument('--max-category-level', type=int, default=config.get('evaluation.max_category_level'),
                        help='Maximum category hierarchy depth')
    parser.add_argument('--top-k', type=int, default=config.get('recommendation.top_k'),
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
    
    faiss_path = config.database_dir / config.index_name
    index_file = faiss_path.with_suffix(config.index_ext)
    metadata_file = faiss_path.with_suffix(config.metadata_ext)
    
    if not (index_file.exists() and metadata_file.exists()):
        print("Error: FAISS index not found. Run predict.py first to build the index.")
        return
    
    print("Loading FAISS index for evaluation...")
    recommender = ProductRecommender(
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        rating_filter_ratio=args.rating_filter_ratio,
        device=device
    )
    recommender.load_index(faiss_path)
    
    stats = recommender.get_statistics()
    print(f"Index: {stats['total_vectors']:,} vectors, dim={stats['dimension']}")
    print(f"Evaluating: sample_ratio={args.sample_ratio}, max_level={args.max_category_level}, top_k={args.top_k}")
    
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
    
    print(f"\nResults:")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  Samples:   {metrics['valid_samples']:,}")


if __name__ == '__main__':
    main()
