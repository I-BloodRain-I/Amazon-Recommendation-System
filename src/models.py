import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, vstack
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class ContentBasedRecommender:
    """TF-IDF based content recommender using cosine similarity"""
    
    def __init__(self, similarity_threshold: float = 0.0):
        self.similarity_threshold = similarity_threshold
        self.feature_matrix = None
        self.product_ids = None
        self.similarity_matrix = None
        self.product_id_to_idx = {}
        
    def fit(self, feature_matrix: csr_matrix, product_ids: np.ndarray) -> 'ContentBasedRecommender':
        """Fit the content-based model"""
        self.feature_matrix = feature_matrix
        self.product_ids = product_ids
        self.product_id_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
        return self
    
    def compute_similarity_matrix(self, batch_size: int = 1000) -> np.ndarray:
        """Compute similarity matrix in batches to handle large datasets"""
        n_products = self.feature_matrix.shape[0]
        self.similarity_matrix = np.zeros((n_products, n_products), dtype=np.float32)
        
        for i in range(0, n_products, batch_size):
            end_i = min(i + batch_size, n_products)
            batch_sim = cosine_similarity(
                self.feature_matrix[i:end_i],
                self.feature_matrix
            )
            self.similarity_matrix[i:end_i] = batch_sim
            
        return self.similarity_matrix
    
    def recommend(self, product_id: str, top_k: int = 10, 
                  exclude_self: bool = True) -> List[Tuple[str, float]]:
        """Get top-k recommendations for a product"""
        if product_id not in self.product_id_to_idx:
            return []
        
        idx = self.product_id_to_idx[product_id]
        
        if self.similarity_matrix is None:
            product_features = self.feature_matrix[idx]
            similarities = cosine_similarity(product_features, self.feature_matrix).flatten()
        else:
            similarities = self.similarity_matrix[idx]
        
        if exclude_self:
            similarities[idx] = -1
        
        mask = similarities >= self.similarity_threshold
        filtered_indices = np.where(mask)[0]
        filtered_similarities = similarities[filtered_indices]
        
        top_indices = filtered_indices[np.argsort(filtered_similarities)[::-1][:top_k]]
        
        recommendations = [
            (self.product_ids[i], float(similarities[i]))
            for i in top_indices
        ]
        
        return recommendations


class NeuralNet(nn.Module):
    """PyTorch neural network architecture"""
    
    def __init__(self, 
                 feature_dim: int,
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3):
        super(NeuralNet, self).__init__()
        
        layers = []
        
        layers.append(nn.Linear(feature_dim, embedding_dim))
        layers.append(nn.BatchNorm1d(embedding_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        prev_dim = embedding_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()


class NeuralRecommender:
    """Neural network based recommender using PyTorch"""
    
    def __init__(self, 
                 feature_dim: int,
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3,
                 l2_reg: float = 0.001):
        
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
        self.scaler_mean = None
        self.scaler_std = None
        
    def build_model(self) -> NeuralNet:
        """Build the neural network architecture"""
        model = NeuralNet(
            feature_dim=self.feature_dim,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate
        )
        return model.to(self.device)
    
    def fit(self, 
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 50,
            batch_size: int = 256,
            learning_rate: float = 0.001,
            early_stopping_patience: int = 10,
            verbose: int = 1) -> 'NeuralRecommender':
        """Train the neural network"""
        
        self.scaler_mean = np.mean(X_train, axis=0)
        self.scaler_std = np.std(X_train, axis=0) + 1e-8
        
        X_train_scaled = (X_train - self.scaler_mean) / self.scaler_std
        
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val is not None:
            X_val_scaled = (X_val - self.scaler_mean) / self.scaler_std
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        self.model = self.build_model()
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=self.l2_reg)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            train_preds = []
            train_targets = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(batch_X)
                train_mae += torch.abs(predictions - batch_y).sum().item()
                train_preds.extend(predictions.detach().cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())
            
            train_loss /= len(train_dataset)
            train_mae /= len(train_dataset)
            train_rmse = np.sqrt(train_loss)
            
            # Calculate R² for training
            train_preds_np = np.array(train_preds)
            train_targets_np = np.array(train_targets)
            ss_res = np.sum((train_targets_np - train_preds_np) ** 2)
            ss_tot = np.sum((train_targets_np - np.mean(train_targets_np)) ** 2)
            train_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            self.history['loss'].append(train_loss)
            self.history['mae'].append(train_mae)
            
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_mae = 0.0
                val_preds = []
                val_targets = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        predictions = self.model(batch_X)
                        loss = criterion(predictions, batch_y)
                        
                        val_loss += loss.item() * len(batch_X)
                        val_mae += torch.abs(predictions - batch_y).sum().item()
                        val_preds.extend(predictions.cpu().numpy())
                        val_targets.extend(batch_y.cpu().numpy())
                
                val_loss /= len(val_dataset)
                val_mae /= len(val_dataset)
                val_rmse = np.sqrt(val_loss)
                
                # Calculate R² for validation
                val_preds_np = np.array(val_preds)
                val_targets_np = np.array(val_targets)
                ss_res = np.sum((val_targets_np - val_preds_np) ** 2)
                ss_tot = np.sum((val_targets_np - np.mean(val_targets_np)) ** 2)
                val_r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                self.history['val_loss'].append(val_loss)
                self.history['val_mae'].append(val_mae)
                
                scheduler.step(val_loss)
                
                if verbose > 0:
                    print(f'Epoch {epoch+1}/{epochs} - '
                          f'loss: {train_loss:.4f} - mae: {train_mae:.4f} - rmse: {train_rmse:.4f} - r2: {train_r2:.4f} - '
                          f'val_loss: {val_loss:.4f} - val_mae: {val_mae:.4f} - val_rmse: {val_rmse:.4f} - val_r2: {val_r2:.4f}')
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose > 0:
                            print(f'Early stopping at epoch {epoch+1}')
                        break
            else:
                if verbose > 0:
                    print(f'Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - mae: {train_mae:.4f} - rmse: {train_rmse:.4f} - r2: {train_r2:.4f}')
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ratings"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def extract_embeddings(self, X: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from the neural network (output of embedding layer)
        This can be used for computing cosine similarity between items
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        self.model.eval()
        
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Extract embeddings from the first few layers (up to embedding layer)
        # We'll use the output after the first ReLU activation
        embeddings_list = []
        
        with torch.no_grad():
            x = X_tensor
            # Pass through embedding layer + BatchNorm + ReLU
            for i in range(4):  # Linear, BatchNorm, ReLU, Dropout
                x = self.model.network[i](x)
            embeddings_list = x.cpu().numpy()
        
        return embeddings_list
    
    def recommend(self, 
                  feature_matrix: np.ndarray,
                  product_ids: np.ndarray,
                  top_k: int = 10,
                  exclude_indices: Optional[List[int]] = None) -> List[Tuple[str, float]]:
        """Get top-k recommendations based on predicted ratings"""
        predictions = self.predict(feature_matrix)
        
        if exclude_indices:
            predictions[exclude_indices] = -np.inf
        
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        recommendations = [
            (product_ids[i], float(predictions[i]))
            for i in top_indices
        ]
        
        return recommendations


class HybridRecommender:
    """Hybrid recommender combining content-based and neural approaches"""
    
    def __init__(self, 
                 content_weight: float = 0.5,
                 neural_weight: float = 0.5):
        self.content_weight = content_weight
        self.neural_weight = neural_weight
        self.content_model = None
        self.neural_model = None
        
    def fit(self,
            content_features: csr_matrix,
            neural_features: np.ndarray,
            product_ids: np.ndarray,
            ratings: np.ndarray,
            **neural_kwargs) -> 'HybridRecommender':
        """Fit both models"""
        
        self.content_model = ContentBasedRecommender()
        self.content_model.fit(content_features, product_ids)
        
        init_params = {
            'embedding_dim': neural_kwargs.pop('embedding_dim', 128),
            'hidden_dims': neural_kwargs.pop('hidden_dims', [256, 128, 64]),
            'dropout_rate': neural_kwargs.pop('dropout_rate', 0.3),
            'l2_reg': neural_kwargs.pop('l2_reg', 0.001)
        }
        
        self.neural_model = NeuralRecommender(
            feature_dim=neural_features.shape[1],
            **init_params
        )
        self.neural_model.fit(neural_features, ratings, **neural_kwargs)
        
        return self
    
    def recommend(self,
                  product_id: str,
                  content_features: Optional[csr_matrix] = None,
                  neural_features: Optional[np.ndarray] = None,
                  product_ids: Optional[np.ndarray] = None,
                  top_k: int = 10) -> List[Tuple[str, float]]:
        """Get hybrid recommendations"""
        
        content_recs = self.content_model.recommend(product_id, top_k=top_k * 2)
        content_scores = {pid: score for pid, score in content_recs}
        
        if self.neural_model and neural_features is not None:
            neural_recs = self.neural_model.recommend(
                neural_features, product_ids, top_k=top_k * 2
            )
            neural_scores = {pid: score for pid, score in neural_recs}
        else:
            neural_scores = {}
        
        all_products = set(content_scores.keys()) | set(neural_scores.keys())
        
        hybrid_scores = {}
        for pid in all_products:
            content_score = content_scores.get(pid, 0)
            neural_score = neural_scores.get(pid, 0)
            hybrid_scores[pid] = (
                self.content_weight * content_score +
                self.neural_weight * neural_score
            )
        
        sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:top_k]
