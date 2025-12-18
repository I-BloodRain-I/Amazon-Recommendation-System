import torch

from src.core.recommender import ProductRecommender
from src.utils.config import config
from src.utils.cli_parser import parse_build_index_args, set_seed


if __name__ == '__main__':
    args = parse_build_index_args()
    set_seed(args.seed)
    
    use_cuda = not args.no_cuda
    device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_path = config.data_dir / 'raw' / config.reviews_file
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        exit(0)
    
    faiss_path = config.database_dir / config.index_name
    recommender = ProductRecommender(
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        rating_filter_ratio=args.rating_filter_ratio,
        category_depth=args.category_depth,
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