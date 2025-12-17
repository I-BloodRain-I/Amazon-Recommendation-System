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
        description='Build FAISS index from product reviews CSV',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--seed', type=int, default=config.get('general.random_seed'),
                        help='Random seed for reproducibility')
    parser.add_argument('--max-products', type=int, default=config.get('index.max_products'),
                        help='Maximum number of products to load')
    parser.add_argument('--sbert-model', type=str, default=config.get('models.sbert_model'),
                        help='Sentence-BERT model name')
    parser.add_argument('--reranker-model', type=str, default=config.get('models.reranker_model'),
                        help='BGE reranker model name')
    parser.add_argument('--rerank-candidates', type=int, default=config.get('recommendation.rerank_candidates'),
                        help='Number of candidates to rerank')
    parser.add_argument('--rating-filter-ratio', type=float, default=config.get('recommendation.rating_filter_ratio'),
                        help='Fraction of candidates to keep after rating filtering (e.g., 0.1 = top 10%%)')
    parser.add_argument('--index-type', type=str, default=config.get('index.type'),
                        choices=['flatl2', 'flatip', 'ivfflat', 'hnsw'],
                        help='FAISS index type')
    parser.add_argument('--embedding-batch-size', type=int, default=config.get('index.embedding_batch_size'),
                        help='Batch size for embedding generation')
    parser.add_argument('--update', action='store_true',
                        help='Update existing index with new products')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    return parser.parse_args()


def main():
    """Build or update FAISS index from CSV."""
    args = parse_args()
    set_seed(args.seed)
    
    use_cuda = not args.no_cuda
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_path = config.data_dir / 'raw' / config.reviews_file
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return
    
    faiss_path = config.database_dir / config.index_name
    recommender = ProductRecommender(
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        rating_filter_ratio=args.rating_filter_ratio,
        device=device
    )
    
    if args.update and faiss_path.with_suffix('.index').exists():
        print("Updating existing FAISS index...")
        recommender.load_index(faiss_path)
        
        stats = recommender.check_and_add_new_products(
            data_path=data_path,
            model_name=args.sbert_model,
            max_products=args.max_products,
            batch_size=args.embedding_batch_size
        )
        
        if stats['new_products_added'] > 0:
            print(f"Added {stats['new_products_added']:,} new products")
            recommender.save_index(faiss_path)
            print("Index updated and saved")
        else:
            print("No new products to add")
    else:
        print("Building new FAISS index...")
        recommender.load_data(data_path, max_products=args.max_products)
        recommender.create_sentence_embeddings(model_name=args.sbert_model, batch_size=args.embedding_batch_size)
        use_gpu_for_faiss = use_cuda and torch.cuda.is_available()
        recommender.create_faiss_index(index_type=args.index_type, use_gpu=use_gpu_for_faiss)
        recommender.save_index(faiss_path)
        print("Index built and saved")
    
    stats = recommender.get_statistics()
    print(f"\nFinal index: {stats['total_vectors']:,} vectors, dim={stats['dimension']}, type={stats['index_type']}")
    print(f"Saved to: {faiss_path}")
    print("\nYou can now run: python predict.py")


if __name__ == '__main__':
    main()
