"""
GCN (Graph Convolutional Network) - Supervised Version

Original implementation from Kipf & Welling (2017).
Uses ground truth labels for training (cross-entropy loss).

Use cases:
- Semi-supervised node classification
- Upper-bound performance comparison
- Transfer learning (train on labeled, test on unlabeled)

NOTE: For fair community detection benchmarking, prefer the unsupervised
version (gcn_cluster.py) which doesn't use ground truth during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.gnn.base_gnn import BaseGNNClustering
from clusternet.registry import register_algorithm


class GCNSupervisedModel(nn.Module):
    """
    Supervised GCN for node classification/community detection.
    
    Architecture from Kipf & Welling (2017):
    - Stack of GCN layers with ReLU activation
    - Dropout for regularization
    - Final softmax for class probabilities
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            
        Returns:
            Log probabilities [N, num_classes]
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get embeddings from second-to-last layer."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        return x


@register_algorithm('gcn_supervised', aliases=['gcn_sup', 'gcn_semi'])
class GCNSupervised(BaseGNNClustering):
    """
    Supervised GCN for community detection.
    
    Uses ground truth labels during training (cross-entropy loss).
    This provides an "upper bound" on performance - what's achievable
    when the model has access to ground truth information.
    
    Parameters:
        num_clusters: Number of communities/classes
        labels: Ground truth labels for training (REQUIRED)
        train_mask: Boolean mask for training nodes (optional)
        hidden_channels: Hidden layer dimension (default: 64)
        num_layers: Number of GCN layers (default: 3)
        epochs: Training epochs (default: 200)
        lr: Learning rate (default: 0.01)
        
    Reference:
        Kipf & Welling (2017). "Semi-Supervised Classification with 
        Graph Convolutional Networks"
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    REQUIRES_FEATURES = True
    IS_TRAINABLE = True
    IS_SUPERVISED = True  # Key difference!
    
    def __init__(
        self,
        G,
        num_clusters: int,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
        train_mask: Optional[np.ndarray] = None,
        hidden_channels: int = 64,
        num_layers: int = 3,
        epochs: int = 200,
        lr: float = 0.01,
        dropout: float = 0.5,
        **kwargs
    ):
        # Store labels before parent init
        self._labels = torch.tensor(labels, dtype=torch.long)
        
        # Train mask (if None, use all nodes)
        if train_mask is None:
            self._train_mask = torch.ones(len(labels), dtype=torch.bool)
        else:
            self._train_mask = torch.tensor(train_mask, dtype=torch.bool)
        
        super().__init__(
            G,
            num_clusters=num_clusters,
            features=features,
            hidden_channels=hidden_channels,
            epochs=epochs,
            lr=lr,
            dropout=dropout,
            **kwargs
        )
        
        self.num_layers = num_layers
        
        # Move labels and mask to device
        self._labels = self._labels.to(self.device)
        self._train_mask = self._train_mask.to(self.device)
    
    def _build_model(self) -> nn.Module:
        """Build supervised GCN model."""
        return GCNSupervisedModel(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.hidden_channels,
            out_channels=self.num_clusters,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
    
    def _train_step(self, data: Data) -> float:
        """Single training step with cross-entropy loss."""
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index)
        
        # Cross-entropy loss on training nodes only
        loss = F.nll_loss(out[self._train_mask], self._labels[self._train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _get_cluster_assignments(self, data: Data) -> np.ndarray:
        """Get cluster assignments (predictions)."""
        out = self.model(data.x, data.edge_index)
        labels = out.argmax(dim=-1).cpu().numpy()
        return labels
    
    def _get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings."""
        return self.model.get_embeddings(data.x, data.edge_index)
    
    def compute_accuracy(self, mask: Optional[torch.Tensor] = None) -> float:
        """
        Compute classification accuracy.
        
        Args:
            mask: Optional mask for subset of nodes
            
        Returns:
            Accuracy score
        """
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=-1)
            
            if mask is None:
                mask = torch.ones(len(pred), dtype=torch.bool, device=self.device)
            
            correct = (pred[mask] == self._labels[mask]).sum().item()
            total = mask.sum().item()
            
        return correct / total if total > 0 else 0.0


# Standalone function
def gcn_supervised_clustering(
    G,
    num_clusters: int,
    labels: np.ndarray,
    features: Optional[np.ndarray] = None,
    train_ratio: float = 0.8,
    hidden_channels: int = 64,
    num_layers: int = 3,
    epochs: int = 200,
    lr: float = 0.01,
    seed: int = 42,
    **kwargs
) -> Tuple[List[List], float, float]:
    """
    Run supervised GCN clustering.
    
    Args:
        G: NetworkX graph
        num_clusters: Number of clusters
        labels: Ground truth labels
        features: Node features
        train_ratio: Fraction of nodes for training
        hidden_channels: Hidden dimension
        num_layers: Number of GCN layers
        epochs: Training epochs
        lr: Learning rate
        seed: Random seed
        
    Returns:
        communities: List of communities
        runtime: Training time
        train_accuracy: Training accuracy
    """
    import time
    
    np.random.seed(seed)
    
    # Create train/test split
    n = len(labels)
    perm = np.random.permutation(n)
    train_size = int(n * train_ratio)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[perm[:train_size]] = True
    
    t1 = time.time()
    algo = GCNSupervised(
        G,
        num_clusters=num_clusters,
        labels=labels,
        features=features,
        train_mask=train_mask,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        epochs=epochs,
        lr=lr,
        **kwargs
    )
    communities = algo.run()
    t2 = time.time()
    
    # Compute accuracy
    train_acc = algo.compute_accuracy(torch.tensor(train_mask, device=algo.device))
    
    return communities, t2 - t1, train_acc









