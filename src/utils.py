import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Dict, Any, Tuple
import pickle
from pathlib import Path
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


def convert_to_sparse(dense_array: np.ndarray) -> csr_matrix:
    """Convert dense array to sparse matrix"""
    return csr_matrix(dense_array)


def convert_to_dense(sparse_matrix: csr_matrix) -> np.ndarray:
    """Convert sparse matrix to dense array"""
    return sparse_matrix.toarray()


def save_sparse_matrix(matrix: csr_matrix, filepath: Path):
    """Save sparse matrix to disk"""
    sp.save_npz(filepath, matrix)


def load_sparse_matrix(filepath: Path) -> csr_matrix:
    """Load sparse matrix from disk"""
    return sp.load_npz(filepath)


def save_model(model, filepath: Path):
    """Save model to disk using pickle"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: Path):
    """Load model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


def create_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/test split maintaining product distribution"""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['main_category'] if 'main_category' in df.columns else None
    )
    
    return train_df, test_df


def calculate_metrics_summary(metrics: Dict[str, Any]) -> pd.DataFrame:
    """Convert metrics dictionary to summary DataFrame"""
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                rows.append({
                    'metric': f'{key}_{sub_key}',
                    'value': sub_value
                })
        else:
            rows.append({
                'metric': key,
                'value': value
            })
    
    return pd.DataFrame(rows)


def format_recommendations(recommendations: Dict, top_k: int = 5) -> pd.DataFrame:
    """Format recommendations as DataFrame"""
    formatted = []
    
    for product_id, recs in recommendations.items():
        for rank, (rec_id, score) in enumerate(recs[:top_k], 1):
            formatted.append({
                'query_product': product_id,
                'rank': rank,
                'recommended_product': rec_id,
                'score': score
            })
    
    return pd.DataFrame(formatted)


def log_training_info(results: Dict[str, Any]):
    """Print formatted training information"""
    print(f"\n{'='*60}")
    print(f"Model Type: {results['model_type'].upper()}")
    print(f"{'='*60}")
    print(f"Products: {results['n_products']:,}")
    
    if 'n_features' in results:
        print(f"Features: {results['n_features']:,}")
    if 'content_features' in results:
        print(f"Content Features: {results['content_features']:,}")
    if 'neural_features' in results:
        print(f"Neural Features: {results['neural_features']:,}")
    
    print(f"Training Time: {results['train_time']:.2f}s")
    
    if 'train_metrics' in results:
        print(f"\nTraining Metrics:")
        for metric, value in results['train_metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    if 'val_metrics' in results:
        print(f"\nValidation Metrics:")
        for metric, value in results['val_metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"{'='*60}\n")


def log_evaluation_info(eval_results: Dict[str, Any]):
    """Print formatted evaluation information"""
    metrics = eval_results['metrics']
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Products Evaluated: {metrics.get('n_products_evaluated', metrics.get('num_queries_evaluated', 0))}")
    
    if 'avg_recommendations_per_product' in metrics:
        print(f"Avg Recommendations: {metrics['avg_recommendations_per_product']:.2f}")
    
    if 'catalog_coverage' in metrics:
        print(f"Catalog Coverage: {metrics['catalog_coverage']:.2%}")
    
    if 'eval_time' in metrics:
        print(f"Evaluation Time: {metrics['eval_time']:.2f}s")
    
    if 'avg_time_per_product' in metrics:
        print(f"Avg Time/Product: {metrics['avg_time_per_product']*1000:.2f}ms")
    
    for k in [5, 10, 20, 50]:
        if f'precision@{k}' in metrics:
            print(f"\nRanking Metrics @{k}:")
            print(f"  Precision: {metrics[f'precision@{k}']:.4f}", end='')
            if f'precision_std' in metrics:
                print(f" (±{metrics['precision_std']:.4f})", end='')
            print()
            
            print(f"  Recall: {metrics[f'recall@{k}']:.4f}", end='')
            if f'recall_std' in metrics:
                print(f" (±{metrics['recall_std']:.4f})", end='')
            print()
            
            if f'f1@{k}' in metrics:
                print(f"  F1-Score: {metrics[f'f1@{k}']:.4f}")
            
            if f'ndcg@{k}' in metrics:
                print(f"  NDCG: {metrics[f'ndcg@{k}']:.4f}")
    
    if 'avg_prediction_error' in metrics:
        print(f"\nPrediction Error: {metrics['avg_prediction_error']:.4f}")
    
    if 'avg_diversity' in metrics:
        print(f"Avg Diversity: {metrics['avg_diversity']:.4f}")
    
    print(f"{'='*60}\n")
