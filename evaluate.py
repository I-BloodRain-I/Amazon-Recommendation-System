import torch

from src.core import ProductRecommender, RecommenderEvaluator
from src.utils.config import config
from src.utils.cli_parser import parse_evaluate_args, set_seed


if __name__ == '__main__':
    args = parse_evaluate_args()
    set_seed(args.seed)
    
    use_cuda = not args.no_cuda
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    faiss_path = config.database_dir / config.index_name
    index_file = faiss_path.with_suffix(config.index_ext)
    metadata_file = faiss_path.with_suffix(config.metadata_ext)
    
    if not (index_file.exists() and metadata_file.exists()):
        print("Error: FAISS index not found. Run predict.py first to build the index.")
        exit(0)
    
    print("Loading FAISS index for evaluation...")
    recommender = ProductRecommender(
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        rating_filter_ratio=args.rating_filter_ratio,
        category_depth=args.category_depth,
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
