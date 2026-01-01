"""
MinCut Pooling for Community Detection

Adapted from RobustGNN with modifications.
Based on: Bianchi et al. "Spectral Clustering with Graph Neural Networks" (2020)

Key Features:
- Spectral clustering in GNN framework
- MinCut + orthogonality objectives
- Soft differentiable clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, dense_mincut_pool
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.gnn.base_gnn import BaseGNNClustering
from clusternet.registry import register_algorithm


class MinCutModel(nn.Module):
    """
    MinCut Pooling Model.
    
    Uses GCN encoder with MinCut pooling objective.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_clusters: int,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.dropout = dropout
        
        # GCN encoder
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Cluster assignment layer
        self.pool = nn.Linear(hidden_channels, num_clusters)
    
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            
        Returns:
            cluster_assignments: [N, K]
            mincut_loss: MinCut objective
            ortho_loss: Orthogonality regularization
        """
        # GCN encoding
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        # Cluster assignments (soft)
        s = self.pool(x)
        
        # Compute MinCut loss
        adj = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0)
        
        # Add batch dimension for dense_mincut_pool
        x_batch = x.unsqueeze(0)
        adj_batch = adj.unsqueeze(0)
        s_batch = s.unsqueeze(0)
        
        _, _, mincut_loss, ortho_loss = dense_mincut_pool(x_batch, adj_batch, s_batch)
        
        return s, mincut_loss, ortho_loss
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings."""
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x


@register_algorithm('mincut', aliases=['mincut_pool', 'mincut_clustering', 'spectral_gnn'])
class MinCutCluster(BaseGNNClustering):
    """
    MinCut pooling-based community detection.
    
    Uses spectral clustering objective within GNN framework.
    Minimizes cut between clusters while maintaining balanced sizes.
    
    Parameters:
        num_clusters: Number of clusters (auto-detected if None)
        hidden_channels: Hidden layer dimension (default: 32)
        epochs: Training epochs (default: 200)
        lr: Learning rate (default: 0.01)
        mincut_weight: Weight for mincut loss (default: 1.0)
        ortho_weight: Weight for orthogonality loss (default: 1.0)
        
    Reference:
        Bianchi et al. (2020). "Spectral Clustering with Graph Neural Networks"
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
        lr: float = 0.01,
        dropout: float = 0.5,
        mincut_weight: float = 1.0,
        ortho_weight: float = 1.0,
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
        self.mincut_weight = mincut_weight
        self.ortho_weight = ortho_weight
    
    def _build_model(self) -> nn.Module:
        """Build MinCut model."""
        return MinCutModel(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.hidden_channels,
            num_clusters=self.num_clusters,
            dropout=self.dropout
        )
    
    def _train_step(self, data: Data) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        _, mincut_loss, ortho_loss = self.model(data.x, data.edge_index)
        
        loss = self.mincut_weight * mincut_loss + self.ortho_weight * ortho_loss
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _get_cluster_assignments(self, data: Data) -> np.ndarray:
        """Get cluster assignments."""
        cluster_logits, _, _ = self.model(data.x, data.edge_index)
        labels = cluster_logits.argmax(dim=-1).cpu().numpy()
        return labels
    
    def _get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings."""
        return self.model.get_embeddings(data.x, data.edge_index)


# Standalone function
def mincut_clustering(
    G,
    num_clusters: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    hidden_channels: int = 32,
    epochs: int = 200,
    lr: float = 0.01,
    **kwargs
) -> Tuple[List[List], float]:
    """
    Run MinCut clustering.
    
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
    algo = MinCutCluster(
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

