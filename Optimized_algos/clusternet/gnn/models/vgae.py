"""
VGAE: Variational Graph Auto-Encoder for Community Detection

Original paper: Kipf & Welling "Variational Graph Auto-Encoders" (2016)

Two-stage approach:
1. Train VGAE to reconstruct adjacency via learned latent embeddings
2. Cluster embeddings with KMeans

Loss: ELBO = E_q[log p(A|Z)] - KL(q(Z|X,A) || p(Z))
  - Reconstruction: BCE on adjacency matrix
  - Regularization: KL divergence to standard Gaussian prior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, VGAE as PyGVGAE
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.gnn.base_gnn import BaseGNNClustering
from clusternet.registry import register_algorithm


class VGAEEncoder(nn.Module):
    """
    Two-layer GCN encoder producing mu and log-sigma for the latent space.

    Architecture from Kipf & Welling (2016): shared first GCN layer,
    then separate GCN layers for mu and log-sigma.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5
    ):
        super().__init__()
        self.dropout = dropout
        self.conv_shared = GCNConv(in_channels, hidden_channels)
        self.bn = nn.BatchNorm1d(hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.bn(F.relu(self.conv_shared(x, edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


@register_algorithm('vgae', aliases=['vgae_clustering', 'vgae_kmeans'])
class VGAECluster(BaseGNNClustering):
    """
    VGAE-based community detection.

    Learns node embeddings by reconstructing the graph adjacency via a
    variational autoencoder, then clusters embeddings with KMeans.

    Parameters:
        num_clusters: Number of clusters (auto-detected if None)
        hidden_channels: Hidden layer dimension (default: 32)
        embedding_dim: Latent embedding dimension (default: 16)
        epochs: Training epochs (default: 200)
        lr: Learning rate (default: 0.01)

    Reference:
        Kipf & Welling (2016). "Variational Graph Auto-Encoders"
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
        embedding_dim: int = 16,
        epochs: int = 200,
        lr: float = 0.01,
        dropout: float = 0.5,
        **kwargs
    ):
        self.embedding_dim = embedding_dim
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
        encoder = VGAEEncoder(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.hidden_channels,
            out_channels=self.embedding_dim,
            dropout=self.dropout,
        )
        return PyGVGAE(encoder)

    def _train_step(self, data: Data) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        z = self.model.encode(data.x, data.edge_index)

        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
        )
        recon_loss = self.model.recon_loss(z, data.edge_index, neg_edge_index)
        kl_loss = (1 / data.num_nodes) * self.model.kl_loss()
        loss = recon_loss + kl_loss

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _get_cluster_assignments(self, data: Data) -> np.ndarray:
        z = self.model.encode(data.x, data.edge_index)
        embeddings = z.cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=42)
        return kmeans.fit_predict(embeddings)

    def _get_embeddings(self, data: Data) -> torch.Tensor:
        return self.model.encode(data.x, data.edge_index)


def vgae_clustering(
    G,
    num_clusters: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    hidden_channels: int = 32,
    embedding_dim: int = 16,
    epochs: int = 200,
    lr: float = 0.01,
    **kwargs
) -> Tuple[List[List], float]:
    """
    Run VGAE clustering.

    Returns:
        communities: List of communities
        runtime: Training time in seconds
    """
    import time

    t1 = time.time()
    algo = VGAECluster(
        G,
        num_clusters=num_clusters,
        features=features,
        hidden_channels=hidden_channels,
        embedding_dim=embedding_dim,
        epochs=epochs,
        lr=lr,
        **kwargs
    )
    communities = algo.run()
    t2 = time.time()
    return communities, t2 - t1
