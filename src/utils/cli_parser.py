import argparse
import random
import numpy as np

from src.utils.config import config


def set_seed(seed: int):
    """Set random seed for reproducibility.
    
    Args:
        seed: Seed for random, numpy, and torch
    """
    random.seed(seed)
    np.random.seed(seed)


def create_base_parser(description: str) -> argparse.ArgumentParser:
    """Create base argument parser with common arguments.
    
    Args:
        description: Description for the command
        
    Returns:
        ArgumentParser with common arguments
    """
    parser = argparse.ArgumentParser(
        description=description,
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
    parser.add_argument('--category-depth', type=int, default=config.get('recommendation.category_depth'),
                        help='Number of category levels to match (e.g., 3 = first 3 levels)')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    return parser


def add_build_index_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add build index specific arguments.
    
    Args:
        parser: Base argument parser
        
    Returns:
        Parser with build index arguments
    """
    parser.add_argument('--max-products', type=int, default=config.get('index.max_products'),
                        help='Maximum number of products to load')
    parser.add_argument('--sbert-model', type=str, default=config.get('models.sbert_model'),
                        help='Sentence-BERT model name')
    parser.add_argument('--index-type', type=str, default=config.get('index.type'),
                        choices=['flatl2', 'flatip', 'ivfflat', 'hnsw'],
                        help='FAISS index type')
    parser.add_argument('--embedding-batch-size', type=int, default=config.get('index.embedding_batch_size'),
                        help='Batch size for embedding generation')
    parser.add_argument('--update', action='store_true',
                        help='Update existing index with new products')
    
    return parser


def add_predict_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add predict specific arguments.
    
    Args:
        parser: Base argument parser
        
    Returns:
        Parser with predict arguments
    """
    parser.add_argument('--top-k', type=int, default=config.get('recommendation.top_k'),
                        help='Number of recommendations to return')
    parser.add_argument('--num-samples', type=int, default=config.get('recommendation.num_samples'),
                        help='Number of random samples to test')
    
    return parser


def add_evaluate_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add evaluate specific arguments.
    
    Args:
        parser: Base argument parser
        
    Returns:
        Parser with evaluate arguments
    """
    parser.add_argument('--rerank-batch-size', type=int, default=config.get('recommendation.rerank_batch_size'),
                        help='Batch size for reranking')
    parser.add_argument('--sample-ratio', type=float, default=config.get('evaluation.sample_ratio'),
                        help='Fraction of products to evaluate (0.0-1.0)')
    parser.add_argument('--max-category-level', type=int, default=config.get('evaluation.max_category_level'),
                        help='Maximum category hierarchy depth')
    parser.add_argument('--top-k', type=int, default=config.get('recommendation.top_k'),
                        help='Number of recommendations to evaluate per query')
    
    return parser


def parse_build_index_args():
    """Parse arguments for build_index script.
    
    Returns:
        Parsed arguments
    """
    parser = create_base_parser('Build FAISS index from product reviews CSV')
    parser = add_build_index_args(parser)
    return parser.parse_args()


def parse_predict_args():
    """Parse arguments for predict script.
    
    Returns:
        Parsed arguments
    """
    parser = create_base_parser('Test product recommendations using pre-built FAISS index')
    parser = add_predict_args(parser)
    return parser.parse_args()


def parse_evaluate_args():
    """Parse arguments for evaluate script.
    
    Returns:
        Parsed arguments
    """
    parser = create_base_parser('Evaluate product recommendation system quality')
    parser = add_evaluate_args(parser)
    return parser.parse_args()
