"""
GAT (Graph Attention Network) for Unsupervised Community Detection

Based on Veličković et al. "Graph Attention Networks" (ICLR 2018)
Adapted for unsupervised clustering via modularity loss.

Key Features:
- Multi-head attention over neighbors
- Learns to weight important neighbors
- Adapted for unsupervised clustering via modularity loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.gnn.base_gnn import BaseGNNClustering, modularity_loss, orthogonality_loss
from clusternet.registry import register_algorithm


class GATEncoder(nn.Module):
    """
    GAT Encoder for node embeddings.
    
    Based on Veličković et al. (2018) architecture with multi-head attention.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        
        # Input layer: in_channels -> hidden_channels (with heads)
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels // heads, heads=heads)
            )
        
        # Output layer (single head for final embedding)
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=False))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            
        Returns:
            Node embeddings [N, hidden_channels]
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x


class GATClusterModel(nn.Module):
    """
    GAT model with clustering head for unsupervised community detection.
    
    Uses Graph Attention Networks with modularity-based loss.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_clusters: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.encoder = GATEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout
        )
        
        # Clustering head
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
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
        """Get node embeddings before clustering head."""
        return self.encoder(x, edge_index)


@register_algorithm('gat_cluster', aliases=['gat_clustering', 'gat_unsup'])
class GATCluster(BaseGNNClustering):
    """
    GAT-based unsupervised community detection.
    
    Uses Graph Attention Networks with modularity loss for 
    unsupervised community detection.
    
    Parameters:
        num_clusters: Number of clusters (auto-detected if None)
        hidden_channels: Hidden layer dimension (default: 64)
        num_layers: Number of GAT layers (default: 3)
        heads: Number of attention heads (default: 4)
        epochs: Training epochs (default: 200)
        lr: Learning rate (default: 0.005)
        modularity_weight: Weight for modularity loss (default: 1.0)
        ortho_weight: Weight for orthogonality loss (default: 0.1)
        
    Reference:
        Veličković et al. (2018). "Graph Attention Networks"
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
        heads: int = 4,
        epochs: int = 200,
        lr: float = 0.005,
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
        self.heads = heads
        self.modularity_weight = modularity_weight
        self.ortho_weight = ortho_weight
        
        # Pre-compute adjacency for modularity loss
        self.adj = to_dense_adj(self.data.edge_index).squeeze(0)
    
    def _build_model(self) -> nn.Module:
        """Build GAT model."""
        return GATClusterModel(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.hidden_channels,
            num_clusters=self.num_clusters,
            num_layers=self.num_layers,
            heads=self.heads,
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


# Standalone function for compatibility
def gat_clustering(
    G,
    num_clusters: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    hidden_channels: int = 64,
    num_layers: int = 3,
    heads: int = 4,
    epochs: int = 200,
    lr: float = 0.005,
    **kwargs
) -> Tuple[List[List], float]:
    """
    Run GAT clustering.
    
    Args:
        G: NetworkX graph
        num_clusters: Number of clusters
        features: Node features
        hidden_channels: Hidden dimension
        num_layers: Number of GAT layers
        heads: Number of attention heads
        epochs: Training epochs
        lr: Learning rate
        
    Returns:
        communities: List of communities
        runtime: Training time
    """
    import time
    
    t1 = time.time()
    algo = GATCluster(
        G,
        num_clusters=num_clusters,
        features=features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
        epochs=epochs,
        lr=lr,
        **kwargs
    )
    communities = algo.run()
    t2 = time.time()
    
    return communities, t2 - t1
