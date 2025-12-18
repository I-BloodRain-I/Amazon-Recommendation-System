import numpy as np
import torch

from src.core.recommender import ProductRecommender
from src.utils.config import config
from src.utils.cli_parser import parse_predict_args, set_seed


if __name__ == '__main__':
    args = parse_predict_args()
    set_seed(args.seed)
    
    use_cuda = not args.no_cuda
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    faiss_path = config.database_dir / config.index_name
    recommender = ProductRecommender(
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        rating_filter_ratio=args.rating_filter_ratio,
        category_depth=args.category_depth,
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
        exit(0)
    
    stats = recommender.get_statistics()
    print(f"\nIndex: {stats['total_vectors']:,} vectors, dim={stats['dimension']}, type={stats['index_type']}")
    print(f"Testing {args.num_samples} samples with top_k={args.top_k}")
    
    random_indices = np.random.choice(len(recommender.product_df), size=args.num_samples, replace=False)
    
    for idx in random_indices:
        similar_products = recommender.find_similar_products(idx, top_k=args.top_k)
        recommender.display_recommendations(idx, similar_products)
