"""
Unified GNN Trainer

Provides a consistent training interface for all GNN models.
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from typing import Callable, Dict, List, Optional, Tuple
import time


class GNNTrainer:
    """
    Unified trainer for GNN clustering models.
    
    Features:
    - Consistent training loop
    - Early stopping
    - Learning rate scheduling
    - Logging and callbacks
    """
    
    def __init__(
        self,
        model: nn.Module,
        data: Data,
        loss_fn: Callable,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch GNN model
            data: PyG Data object
            loss_fn: Loss function (data, model_output) -> loss
            lr: Learning rate
            weight_decay: L2 regularization
            device: Device to train on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.data = data.to(self.device)
        self.loss_fn = loss_fn
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.history = {
            'loss': [],
            'time': []
        }
    
    def train_epoch(self) -> float:
        """Single training epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        output = self.model(self.data.x, self.data.edge_index)
        loss = self.loss_fn(self.data, output)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(
        self,
        epochs: int = 200,
        patience: int = 20,
        min_delta: float = 1e-4,
        verbose: bool = False
    ) -> Dict[str, List]:
        """
        Full training loop.
        
        Args:
            epochs: Maximum epochs
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            verbose: Print progress
            
        Returns:
            Training history
        """
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            t1 = time.time()
            loss = self.train_epoch()
            t2 = time.time()
            
            self.history['loss'].append(loss)
            self.history['time'].append(t2 - t1)
            
            # Early stopping check
            if loss < best_loss - min_delta:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1:03d}, Loss: {loss:.4f}')
            
            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch + 1}')
                break
        
        return self.history
    
    def get_clusters(self) -> torch.Tensor:
        """Get cluster assignments from trained model."""
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.data.x, self.data.edge_index)
            
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]  # Take first element (usually cluster logits)
            
            if output.dim() == 2:
                return output.argmax(dim=-1)
            else:
                return output

