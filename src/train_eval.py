import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.sparse import csr_matrix
from typing import Dict, List, Tuple, Optional, Union, Any
import time
import json
from pathlib import Path

try:
    from .models import ContentBasedRecommender, NeuralRecommender, HybridRecommender
except ImportError:
    from models import ContentBasedRecommender, NeuralRecommender, HybridRecommender


def train_tfidf_model(
    feature_matrix: csr_matrix,
    product_ids: np.ndarray,
    compute_similarity: bool = True,
    similarity_batch_size: int = 1000,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Train TF-IDF based content recommender
    
    Args:
        feature_matrix: Sparse TF-IDF feature matrix
        product_ids: Array of product identifiers
        compute_similarity: Whether to precompute similarity matrix
        similarity_batch_size: Batch size for similarity computation
        **model_kwargs: Additional arguments for ContentBasedRecommender
        
    Returns:
        Dictionary containing model and training metadata
    """
    start_time = time.time()
    
    model = ContentBasedRecommender(**model_kwargs)
    model.fit(feature_matrix, product_ids)
    
    if compute_similarity:
        model.compute_similarity_matrix(batch_size=similarity_batch_size)
    
    train_time = time.time() - start_time
    
    results = {
        'model': model,
        'model_type': 'tfidf',
        'n_products': len(product_ids),
        'n_features': feature_matrix.shape[1],
        'sparsity': 1 - (feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1])),
        'train_time': train_time,
        'similarity_precomputed': compute_similarity
    }
    
    return results


def train_neural_model(
    X: np.ndarray,
    y: np.ndarray,
    product_ids: np.ndarray,
    validation_split: float = 0.2,
    embedding_dim: int = 128,
    hidden_dims: List[int] = [256, 128, 64],
    dropout_rate: float = 0.3,
    l2_reg: float = 0.001,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    random_state: int = 42,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Train PyTorch neural network based recommender
    
    Args:
        X: Feature matrix
        y: Target ratings
        product_ids: Array of product identifiers
        validation_split: Fraction of data for validation
        embedding_dim: Dimension of embedding layer
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate
        l2_reg: L2 regularization strength
        epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        early_stopping_patience: Patience for early stopping
        random_state: Random seed
        verbose: Verbosity level
        
    Returns:
        Dictionary containing model and training metadata
    """
    start_time = time.time()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_split, random_state=random_state
    )
    
    model = NeuralRecommender(
        feature_dim=X.shape[1],
        embedding_dim=embedding_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg
    )
    
    model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose
    )
    
    train_time = time.time() - start_time
    
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    results = {
        'model': model,
        'model_type': 'neural',
        'n_products': len(product_ids),
        'n_features': X.shape[1],
        'train_time': train_time,
        'train_metrics': {
            'mse': float(mean_squared_error(y_train, train_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
            'mae': float(mean_absolute_error(y_train, train_pred)),
            'r2': float(r2_score(y_train, train_pred))
        },
        'val_metrics': {
            'mse': float(mean_squared_error(y_val, val_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_val, val_pred))),
            'mae': float(mean_absolute_error(y_val, val_pred)),
            'r2': float(r2_score(y_val, val_pred))
        },
        'history': model.history
    }
    
    return results


def train_hybrid_model(
    content_features: csr_matrix,
    neural_features: np.ndarray,
    product_ids: np.ndarray,
    ratings: np.ndarray,
    content_weight: float = 0.5,
    neural_weight: float = 0.5,
    validation_split: float = 0.2,
    random_state: int = 42,
    **neural_kwargs
) -> Dict[str, Any]:
    """
    Train hybrid recommender combining content and neural models
    
    Args:
        content_features: Sparse content feature matrix
        neural_features: Dense feature matrix for neural network
        product_ids: Array of product identifiers
        ratings: Target ratings
        content_weight: Weight for content-based recommendations
        neural_weight: Weight for neural recommendations
        validation_split: Fraction for validation
        random_state: Random seed
        **neural_kwargs: Additional arguments for neural model
        
    Returns:
        Dictionary containing model and training metadata
    """
    start_time = time.time()
    
    _, _, _, y_val = train_test_split(
        neural_features, ratings, test_size=validation_split, random_state=random_state
    )
    
    model = HybridRecommender(
        content_weight=content_weight,
        neural_weight=neural_weight
    )
    
    model.fit(
        content_features=content_features,
        neural_features=neural_features,
        product_ids=product_ids,
        ratings=ratings,
        **neural_kwargs
    )
    
    train_time = time.time() - start_time
    
    results = {
        'model': model,
        'model_type': 'hybrid',
        'n_products': len(product_ids),
        'content_features': content_features.shape[1],
        'neural_features': neural_features.shape[1],
        'train_time': train_time,
        'weights': {
            'content': content_weight,
            'neural': neural_weight
        }
    }
    
    results['neural_metrics'] = {
        'train': model.neural_model.history
    }
    
    return results


def evaluate_recommendations(
    model,
    model_type: str,
    test_product_ids: List[str],
    ground_truth_ratings: Optional[Dict[str, float]] = None,
    ground_truth_interactions: Optional[Dict[str, List[str]]] = None,
    top_k: int = 10,
    content_features: Optional[csr_matrix] = None,
    neural_features: Optional[np.ndarray] = None,
    all_product_ids: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate recommendation quality with precision and recall metrics
    
    Args:
        model: Trained model (ContentBasedRecommender, NeuralRecommender, or HybridRecommender)
        model_type: Type of model ('tfidf', 'neural', 'hybrid')
        test_product_ids: Product IDs to generate recommendations for
        ground_truth_ratings: Optional dict mapping product_id to rating
        ground_truth_interactions: Optional dict mapping user/product to list of relevant items
        top_k: Number of recommendations to generate
        content_features: Content features for hybrid/content models
        neural_features: Neural features for neural/hybrid models
        all_product_ids: All product IDs for neural recommendations
        
    Returns:
        Dictionary containing evaluation metrics including precision, recall, F1
    """
    start_time = time.time()
    
    recommendations = {}
    coverage_set = set()
    
    for product_id in test_product_ids:
        try:
            if model_type == 'tfidf':
                # If the query product is known to the trained model, use the model's recommend()
                if product_id in getattr(model, 'product_id_to_idx', {}):
                    recs = model.recommend(product_id, top_k=top_k)
                else:
                    # Fall back to computing similarity between the query features and the model's catalog
                    # Requires full content_features (rows aligned with all_product_ids)
                    if content_features is None or all_product_ids is None:
                        continue
                    # locate query index in all_product_ids
                    idxs = np.where(all_product_ids == product_id)[0]
                    if len(idxs) == 0:
                        continue
                    query_idx = idxs[0]
                    from sklearn.metrics.pairwise import cosine_similarity
                    # model.feature_matrix contains the catalog the model was trained on
                    query_feat = content_features[query_idx]
                    candidate_feats = model.feature_matrix
                    sims = cosine_similarity(query_feat, candidate_feats).flatten()
                    top_indices = np.argsort(sims)[::-1][:top_k]
                    recs = [(model.product_ids[i], float(sims[i])) for i in top_indices]
            elif model_type == 'neural':
                if neural_features is None or all_product_ids is None:
                    continue
                recs = model.recommend(neural_features, all_product_ids, top_k=top_k)
            elif model_type == 'hybrid':
                recs = model.recommend(
                    product_id,
                    content_features=content_features,
                    neural_features=neural_features,
                    product_ids=all_product_ids,
                    top_k=top_k
                )
            else:
                continue

            recommendations[product_id] = recs
            coverage_set.update([pid for pid, _ in recs])

        except Exception:
            continue
    
    eval_time = time.time() - start_time
    
    metrics = {
        'n_products_evaluated': len(recommendations),
        'avg_recommendations_per_product': np.mean([len(recs) for recs in recommendations.values()]),
        'catalog_coverage': len(coverage_set) / len(all_product_ids) if all_product_ids is not None else 0,
        'eval_time': eval_time,
        'avg_time_per_product': eval_time / len(test_product_ids) if test_product_ids else 0
    }
    
    if ground_truth_ratings:
        prediction_errors = []
        for product_id, recs in recommendations.items():
            for rec_id, score in recs:
                if rec_id in ground_truth_ratings:
                    prediction_errors.append(abs(ground_truth_ratings[rec_id] - score))
        
        if prediction_errors:
            metrics['avg_prediction_error'] = float(np.mean(prediction_errors))
            metrics['std_prediction_error'] = float(np.std(prediction_errors))
    
    intra_list_diversity = []
    for recs in recommendations.values():
        if len(recs) < 2:
            continue
        
        rec_indices = []
        for rec_id, _ in recs:
            if model_type == 'tfidf' and rec_id in model.product_id_to_idx:
                rec_indices.append(model.product_id_to_idx[rec_id])
            elif model_type in ['neural', 'hybrid'] and all_product_ids is not None:
                try:
                    idx = np.where(all_product_ids == rec_id)[0]
                    if len(idx) > 0:
                        rec_indices.append(idx[0])
                except:
                    continue
        
        if len(rec_indices) < 2:
            continue
        
        if model_type == 'tfidf' and model.feature_matrix is not None:
            rec_features = model.feature_matrix[rec_indices]
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(rec_features)
            avg_similarity = (sim_matrix.sum() - sim_matrix.trace()) / (len(rec_indices) * (len(rec_indices) - 1))
            diversity = 1 - avg_similarity
            intra_list_diversity.append(diversity)
        elif model_type in ['neural', 'hybrid'] and neural_features is not None:
            rec_features = neural_features[rec_indices]
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(rec_features)
            avg_similarity = (sim_matrix.sum() - sim_matrix.trace()) / (len(rec_indices) * (len(rec_indices) - 1))
            diversity = 1 - avg_similarity
            intra_list_diversity.append(diversity)
    
    if intra_list_diversity:
        metrics['avg_diversity'] = float(np.mean(intra_list_diversity))
        metrics['diversity_std'] = float(np.std(intra_list_diversity))
    
    if ground_truth_interactions:
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for product_id, recs in recommendations.items():
            if product_id not in ground_truth_interactions:
                continue
            
            relevant_items = set(ground_truth_interactions[product_id])
            recommended_items = set([rec_id for rec_id, _ in recs])
            
            if len(recommended_items) == 0:
                continue
            
            true_positives = len(relevant_items & recommended_items)
            
            precision = true_positives / len(recommended_items)
            recall = true_positives / len(relevant_items) if len(relevant_items) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        if precision_scores:
            metrics[f'precision@{top_k}'] = float(np.mean(precision_scores))
            metrics[f'recall@{top_k}'] = float(np.mean(recall_scores))
            metrics[f'f1@{top_k}'] = float(np.mean(f1_scores))
            metrics['precision_std'] = float(np.std(precision_scores))
            metrics['recall_std'] = float(np.std(recall_scores))
            metrics['f1_std'] = float(np.std(f1_scores))
    
    return {
        'metrics': metrics,
        'recommendations': recommendations
    }


def evaluate_neural_predictions(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate neural model predictions
    
    Args:
        model: Trained NeuralRecommender
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    predictions = model.predict(X_test)
    
    metrics = {
        'mse': float(mean_squared_error(y_test, predictions)),
        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
        'mae': float(mean_absolute_error(y_test, predictions)),
        'r2': float(r2_score(y_test, predictions))
    }
    
    return metrics


def evaluate_ranking_metrics(
    model,
    model_type: str,
    ground_truth: Dict[str, List[str]],
    all_product_ids: np.ndarray,
    content_features: Optional[csr_matrix] = None,
    neural_features: Optional[np.ndarray] = None,
    top_k: int = 10,
    num_negatives: int = 100
) -> Dict[str, float]:
    """
    Evaluate ranking metrics with negative sampling (Precision@K, Recall@K, NDCG@K, Hit Rate@K)
    
    Args:
        model: Trained recommender model
        model_type: Type of model ('tfidf', 'neural', 'hybrid')
        ground_truth: Dict mapping product/user ID to list of relevant items
        all_product_ids: Array of all available product IDs
        content_features: Content features for content-based models
        neural_features: Neural features for neural models
        top_k: Number of recommendations to evaluate
        num_negatives: Number of negative samples per query
        
    Returns:
        Dictionary containing precision@K, recall@K, F1@K, NDCG@K, Hit Rate@K
    """
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    hit_scores = []
    map_scores = []
    
    all_product_set = set(all_product_ids)
    
    for query_id, relevant_items in ground_truth.items():
        if not relevant_items:
            continue
        
        relevant_set = set(relevant_items)
        
        candidate_negatives = list(all_product_set - relevant_set - {query_id})
        current_num_negatives = min(num_negatives, len(candidate_negatives))
        
        if current_num_negatives == 0:
            continue
        
        negative_samples = np.random.choice(candidate_negatives, size=current_num_negatives, replace=False)
        
        candidates = list(relevant_set) + list(negative_samples)
        
        try:
            if model_type == 'tfidf':
                if query_id not in model.product_id_to_idx:
                    continue
                
                query_idx = model.product_id_to_idx[query_id]
                candidate_indices = [model.product_id_to_idx.get(pid) for pid in candidates]
                valid_mask = [idx is not None for idx in candidate_indices]
                candidate_indices = [idx for idx in candidate_indices if idx is not None]
                candidates = [candidates[i] for i in range(len(candidates)) if valid_mask[i]]
                
                if not candidate_indices:
                    continue
                
                query_features = model.feature_matrix[query_idx]
                candidate_features = model.feature_matrix[candidate_indices]
                
                from sklearn.metrics.pairwise import cosine_similarity
                scores = cosine_similarity(query_features, candidate_features).flatten()
                
                scored_items = [(candidates[i], scores[i]) for i in range(len(scores))]
                
            elif model_type == 'neural':
                if neural_features is None:
                    continue
                
                product_id_to_idx = {pid: idx for idx, pid in enumerate(all_product_ids)}
                candidate_indices = [product_id_to_idx.get(pid) for pid in candidates]
                valid_mask = [idx is not None for idx in candidate_indices]
                candidate_indices = [idx for idx in candidate_indices if idx is not None]
                candidates = [candidates[i] for i in range(len(candidates)) if valid_mask[i]]
                
                if not candidate_indices:
                    continue
                
                candidate_features = neural_features[candidate_indices]
                predictions = model.predict(candidate_features)
                
                scored_items = [(candidates[i], predictions[i]) for i in range(len(predictions))]
                
            else:
                continue
            
            scored_items.sort(key=lambda x: x[1], reverse=True)
            top_k_items = [item_id for item_id, _ in scored_items[:top_k]]
            
            hits = sum(1 for item in top_k_items if item in relevant_set)
            
            precision = hits / top_k
            recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0
            hit_rate = 1.0 if hits > 0 else 0.0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            hit_scores.append(hit_rate)
            
            dcg = 0.0
            for i, item_id in enumerate(top_k_items):
                if item_id in relevant_set:
                    dcg += 1.0 / np.log2(i + 2)
            
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), top_k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            
            ndcg_scores.append(ndcg)
            
            avg_precision = 0.0
            num_hits = 0
            for i, item_id in enumerate(top_k_items):
                if item_id in relevant_set:
                    num_hits += 1
                    precision_at_i = num_hits / (i + 1)
                    avg_precision += precision_at_i
            
            avg_precision = avg_precision / len(relevant_set) if len(relevant_set) > 0 else 0.0
            map_scores.append(avg_precision)
            
        except Exception:
            continue
    
    metrics = {}
    if precision_scores:
        metrics[f'precision@{top_k}'] = float(np.mean(precision_scores))
        metrics[f'recall@{top_k}'] = float(np.mean(recall_scores))
        
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 
                     for p, r in zip(precision_scores, recall_scores)]
        metrics[f'f1@{top_k}'] = float(np.mean(f1_scores))
        
        metrics[f'ndcg@{top_k}'] = float(np.mean(ndcg_scores))
        metrics[f'hit_rate@{top_k}'] = float(np.mean(hit_scores))
        metrics[f'map@{top_k}'] = float(np.mean(map_scores))
        
        metrics[f'precision@{top_k}_std'] = float(np.std(precision_scores))
        metrics[f'recall@{top_k}_std'] = float(np.std(recall_scores))
        metrics[f'f1@{top_k}_std'] = float(np.std(f1_scores))
        metrics[f'ndcg@{top_k}_std'] = float(np.std(ndcg_scores))
        metrics[f'hit_rate@{top_k}_std'] = float(np.std(hit_scores))
        metrics[f'map@{top_k}_std'] = float(np.std(map_scores))
        
        metrics['num_queries_evaluated'] = len(precision_scores)
    
    return metrics


def save_model_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    model_name: str
):
    """
    Save model training results
    
    Args:
        results: Training results dictionary
        output_dir: Output directory path
        model_name: Name for the model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {k: v for k, v in results.items() if k not in ['model', 'history']}
    
    with open(output_dir / f"{model_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    if results['model_type'] == 'neural' and results.get('history'):
        history_df = pd.DataFrame(results['history'])
        history_df.to_csv(output_dir / f"{model_name}_history.csv", index=False)


def load_model_metadata(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load model metadata
    
    Args:
        model_path: Path to metadata JSON file
        
    Returns:
        Dictionary containing model metadata
    """
    with open(model_path, 'r') as f:
        metadata = json.load(f)
    return metadata
