"""
DMoN: Differentiable Modularity Networks for Community Detection

Adapted from RobustGNN with modifications for ClusterNet integration.
Original paper: Tsitsulin et al. "Graph Clustering with Graph Neural Networks" (2020)

Key Features:
- Unsupervised learning via modularity optimization
- Soft cluster assignments
- Differentiable pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, DenseGCNConv, DMoNPooling
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Data
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.gnn.base_gnn import BaseGNNClustering
from clusternet.registry import register_algorithm


class DMoNModel(nn.Module):
    """
    DMoN Model Architecture.
    
    Two-stage architecture:
    1. GCN encoder for node embeddings
    2. DMoN pooling for cluster assignments
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_clusters: int,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_clusters = num_clusters
        self.dropout = dropout
        
        # GCN encoder with batch norm
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Skip connection: DMoNPooling sees [GCN output || raw features]
        pool_input_dim = hidden_channels + in_channels
        self.pool = DMoNPooling(
            channels=[pool_input_dim, pool_input_dim],
            k=num_clusters
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            
        Returns:
            cluster_assignments: [N, K]
            total_loss: Scalar loss for training
        """
        # GCN encoding with batch norm + skip connection
        x_raw = x
        x = self.bn1(F.selu(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(F.selu(self.conv2(x, edge_index)))
        x = torch.cat([x, x_raw], dim=-1)
        
        # Convert to dense format for pooling
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x_dense, mask = to_dense_batch(x, batch=batch)
        adj = to_dense_adj(edge_index, batch=batch)
        
        # DMoN pooling
        cluster_assignments, x_pooled, adj_pooled, spectral_loss, ortho_loss, cluster_loss = self.pool(
            x_dense, adj, mask
        )
        
        # Total loss
        total_loss = spectral_loss + ortho_loss + cluster_loss
        
        return cluster_assignments.squeeze(0), total_loss
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings before pooling."""
        x_raw = x
        x = self.bn1(F.selu(self.conv1(x, edge_index)))
        x = self.bn2(F.selu(self.conv2(x, edge_index)))
        return torch.cat([x, x_raw], dim=-1)


@register_algorithm('dmon', aliases=['dmon_clustering', 'modularity_gnn'])
class DMoNCluster(BaseGNNClustering):
    """
    DMoN-based community detection.
    
    Uses differentiable modularity optimization to learn cluster assignments.
    Unsupervised - no ground truth labels needed.
    
    Parameters:
        num_clusters: Number of clusters (auto-detected if None)
        hidden_channels: Hidden layer dimension (default: 32)
        epochs: Training epochs (default: 200)
        lr: Learning rate (default: 0.001)
        
    Reference:
        Tsitsulin et al. (2020). "Graph Clustering with Graph Neural Networks"
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    REQUIRES_FEATURES = True
    IS_TRAINABLE = True
    
    def __init__(
        self,
        G,
        num_clusters: Optional[int] = None,
        features: Optional[np.ndarray] = None,
        hidden_channels: int = 32,
        epochs: int = 200,
        lr: float = 0.001,
        dropout: float = 0.5,
        **kwargs
    ):
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
        self.dropout = dropout
    
    def _build_model(self) -> nn.Module:
        """Build DMoN model."""
        return DMoNModel(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.hidden_channels,
            num_clusters=self.num_clusters,
            dropout=self.dropout
        )
    
    def _train_step(self, data: Data) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        cluster_assignments, loss = self.model(data.x, data.edge_index)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _get_cluster_assignments(self, data: Data) -> np.ndarray:
        """Get cluster assignments."""
        cluster_assignments, _ = self.model(data.x, data.edge_index)
        labels = cluster_assignments.argmax(dim=-1).cpu().numpy()
        return labels
    
    def _get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings."""
        return self.model.get_embeddings(data.x, data.edge_index)


# Standalone function for compatibility
def dmon_clustering(
    G,
    num_clusters: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    hidden_channels: int = 32,
    epochs: int = 200,
    lr: float = 0.001,
    **kwargs
) -> Tuple[List[List], float]:
    """
    Run DMoN clustering.
    
    Args:
        G: NetworkX graph
        num_clusters: Number of clusters
        features: Node features
        hidden_channels: Hidden dimension
        epochs: Training epochs
        lr: Learning rate
        
    Returns:
        communities: List of communities
        runtime: Training time
    """
    import time
    
    t1 = time.time()
    algo = DMoNCluster(
        G,
        num_clusters=num_clusters,
        features=features,
        hidden_channels=hidden_channels,
        epochs=epochs,
        lr=lr,
        **kwargs
    )
    communities = algo.run()
    t2 = time.time()
    
    return communities, t2 - t1









