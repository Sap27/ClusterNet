"""
DGI: Deep Graph Infomax for Community Detection

Original paper: Velickovic et al. "Deep Graph Infomax" (ICLR 2019)

Two-stage approach:
1. Train encoder by maximizing mutual information between node embeddings
   and a graph-level summary, using a corruption-based contrastive objective
2. Cluster learned embeddings with KMeans

Loss: BCE on bilinear discriminator D(h_i, s) = sigmoid(h_i^T W s)
  distinguishing real node-graph pairs from corrupted (shuffled) pairs.
  This is the Jensen-Shannon MI estimator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, DeepGraphInfomax
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.gnn.base_gnn import BaseGNNClustering
from clusternet.registry import register_algorithm


class DGIEncoder(nn.Module):
    """Two-layer GCN encoder for DGI."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.prelu = nn.PReLU(hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x, edge_index))
        x = self.prelu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.bn2(self.conv2(x, edge_index))
        return x


def _corruption(x: torch.Tensor, edge_index: torch.Tensor):
    """Corrupt the graph by shuffling node features (standard DGI corruption)."""
    perm = torch.randperm(x.size(0), device=x.device)
    return x[perm], edge_index


def _summary(z: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Graph-level readout: sigmoid of mean pooling (standard DGI readout)."""
    return torch.sigmoid(z.mean(dim=0))


@register_algorithm('dgi', aliases=['dgi_clustering', 'dgi_kmeans', 'deep_graph_infomax'])
class DGICluster(BaseGNNClustering):
    """
    DGI-based community detection.

    Learns node embeddings by maximizing mutual information between local
    node representations and a global graph summary. Clusters embeddings
    with KMeans.

    Parameters:
        num_clusters: Number of clusters (auto-detected if None)
        hidden_channels: Hidden layer dimension (default: 64)
        epochs: Training epochs (default: 300)
        lr: Learning rate (default: 0.001)

    Reference:
        Velickovic et al. (2019). "Deep Graph Infomax" (ICLR)
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
        epochs: int = 300,
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

    def _build_model(self) -> nn.Module:
        encoder = DGIEncoder(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.hidden_channels,
            dropout=self.dropout,
        )
        return DeepGraphInfomax(
            hidden_channels=self.hidden_channels,
            encoder=encoder,
            summary=_summary,
            corruption=_corruption,
        )

    def _train_step(self, data: Data) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        pos_z, neg_z, summary = self.model(data.x, data.edge_index)
        loss = self.model.loss(pos_z, neg_z, summary)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _get_cluster_assignments(self, data: Data) -> np.ndarray:
        z = self.model.encoder(data.x, data.edge_index)
        embeddings = z.cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=42)
        return kmeans.fit_predict(embeddings)

    def _get_embeddings(self, data: Data) -> torch.Tensor:
        return self.model.encoder(data.x, data.edge_index)


def dgi_clustering(
    G,
    num_clusters: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    hidden_channels: int = 64,
    epochs: int = 300,
    lr: float = 0.001,
    **kwargs
) -> Tuple[List[List], float]:
    """
    Run DGI clustering.

    Returns:
        communities: List of communities
        runtime: Training time in seconds
    """
    import time

    t1 = time.time()
    algo = DGICluster(
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
