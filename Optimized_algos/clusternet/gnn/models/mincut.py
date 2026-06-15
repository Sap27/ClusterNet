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
from torch_geometric.nn import GCNConv
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
        dropout: float = 0.5,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.dropout = dropout
        self.temperature = temperature
        
        # GCN encoder
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        
        # Skip connection: pool sees [GCN output || raw features]
        # to prevent over-smoothing from destroying discriminative signal
        pool_input_dim = hidden_channels + in_channels
        self.pool = nn.Linear(pool_input_dim, num_clusters)
        nn.init.xavier_uniform_(self.pool.weight, gain=5.0)
        nn.init.uniform_(self.pool.bias, -1.0, 1.0)
    
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
        # GCN encoding with batch norm
        x_raw = x
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        
        # Skip connection: concatenate raw features to prevent over-smoothing
        x_cat = torch.cat([x, x_raw], dim=-1)
        
        # Cluster assignments — temperature < 1 sharpens softmax to break
        # uniform-assignment collapse where gradients vanish
        s_logits = self.pool(x_cat) / self.temperature
        s = F.softmax(s_logits, dim=-1)
        
        # Compute MinCut + orthogonality loss manually (Bianchi et al., 2020)
        adj = to_dense_adj(edge_index, max_num_nodes=x.size(0)).squeeze(0)
        d = adj.sum(dim=-1)
        
        # MinCut loss: -Tr(S^T A S) / Tr(S^T D S)
        st_a_s = torch.matmul(s.t(), torch.matmul(adj, s))
        st_d_s = torch.matmul(s.t(), s * d.unsqueeze(-1))
        mincut_loss = -torch.trace(st_a_s) / (torch.trace(st_d_s) + 1e-8)
        
        # Orthogonality loss: ||S^T S / ||S^T S||_F - I_K / sqrt(K)||_F
        ss = torch.matmul(s.t(), s)
        ss_norm = ss / (torch.norm(ss, p='fro') + 1e-8)
        eye_k = torch.eye(self.num_clusters, device=s.device) / (self.num_clusters ** 0.5)
        ortho_loss = torch.norm(ss_norm - eye_k, p='fro')
        
        return s, mincut_loss, ortho_loss
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings."""
        x_raw = x
        x = self.bn1(F.relu(self.conv1(x, edge_index)))
        x = self.bn2(F.relu(self.conv2(x, edge_index)))
        return torch.cat([x, x_raw], dim=-1)


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
        ortho_weight: float = 5.0,
        temperature: float = 0.3,
        **kwargs
    ):
        self.temperature = temperature
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
            dropout=self.dropout,
            temperature=self.temperature,
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









