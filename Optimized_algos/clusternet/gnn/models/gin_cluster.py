"""
GIN (Graph Isomorphism Network) for Community Detection

NEW implementation - not from RobustGNN.
Based on: Xu et al. "How Powerful are Graph Neural Networks?" (ICLR 2019)

Key Features:
- Most expressive GNN architecture (WL-test equivalent)
- MLP-based aggregation
- Better at capturing structural patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.gnn.base_gnn import BaseGNNClustering, modularity_loss, orthogonality_loss
from clusternet.registry import register_algorithm


class GINEncoder(nn.Module):
    """
    GIN Encoder with MLP aggregation.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(mlp, train_eps=True))
        self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            
        Returns:
            Node embeddings [N, hidden_channels]
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GINClusterModel(nn.Module):
    """
    GIN model with clustering head.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_clusters: int,
        num_layers: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.encoder = GINEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Clustering head
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_clusters)
        )
        
        self.num_clusters = num_clusters
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            
        Returns:
            Cluster logits [N, K]
        """
        embeddings = self.encoder(x, edge_index)
        cluster_logits = self.cluster_head(embeddings)
        return cluster_logits
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings."""
        return self.encoder(x, edge_index)


@register_algorithm('gin_cluster', aliases=['gin_clustering', 'gin'])
class GINCluster(BaseGNNClustering):
    """
    GIN-based community detection.
    
    Uses Graph Isomorphism Network with unsupervised clustering loss.
    Most expressive GNN architecture, equivalent to WL-test.
    
    Parameters:
        num_clusters: Number of clusters (auto-detected if None)
        hidden_channels: Hidden layer dimension (default: 64)
        num_layers: Number of GIN layers (default: 3)
        epochs: Training epochs (default: 200)
        lr: Learning rate (default: 0.01)
        modularity_weight: Weight for modularity loss (default: 1.0)
        ortho_weight: Weight for orthogonality loss (default: 0.1)
        
    Reference:
        Xu et al. (2019). "How Powerful are Graph Neural Networks?"
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
        hidden_channels: int = 64,
        num_layers: int = 3,
        epochs: int = 200,
        lr: float = 0.01,
        dropout: float = 0.5,
        modularity_weight: float = 1.0,
        ortho_weight: float = 0.1,
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
        self.num_layers = num_layers
        self.modularity_weight = modularity_weight
        self.ortho_weight = ortho_weight
        
        # Pre-compute adjacency for loss
        self.adj = to_dense_adj(self.data.edge_index).squeeze(0)
    
    def _build_model(self) -> nn.Module:
        """Build GIN model."""
        return GINClusterModel(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.hidden_channels,
            num_clusters=self.num_clusters,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
    
    def _train_step(self, data: Data) -> float:
        """Single training step with unsupervised loss."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get cluster assignments
        cluster_logits = self.model(data.x, data.edge_index)
        
        # Compute losses
        mod_loss = modularity_loss(self.adj, cluster_logits)
        ortho_loss_val = orthogonality_loss(cluster_logits)
        
        # Total loss
        loss = self.modularity_weight * mod_loss + self.ortho_weight * ortho_loss_val
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _get_cluster_assignments(self, data: Data) -> np.ndarray:
        """Get cluster assignments."""
        cluster_logits = self.model(data.x, data.edge_index)
        labels = cluster_logits.argmax(dim=-1).cpu().numpy()
        return labels
    
    def _get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings."""
        return self.model.get_embeddings(data.x, data.edge_index)


# Standalone function
def gin_clustering(
    G,
    num_clusters: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    hidden_channels: int = 64,
    num_layers: int = 3,
    epochs: int = 200,
    lr: float = 0.01,
    **kwargs
) -> Tuple[List[List], float]:
    """
    Run GIN clustering.
    
    Args:
        G: NetworkX graph
        num_clusters: Number of clusters
        features: Node features
        hidden_channels: Hidden dimension
        num_layers: Number of GIN layers
        epochs: Training epochs
        lr: Learning rate
        
    Returns:
        communities: List of communities
        runtime: Training time
    """
    import time
    
    t1 = time.time()
    algo = GINCluster(
        G,
        num_clusters=num_clusters,
        features=features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        epochs=epochs,
        lr=lr,
        **kwargs
    )
    communities = algo.run()
    t2 = time.time()
    
    return communities, t2 - t1









