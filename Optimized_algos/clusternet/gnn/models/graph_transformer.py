"""
Graph Transformer for Community Detection

NEW implementation - not from RobustGNN.
Based on: Dwivedi et al. "A Generalization of Transformer Networks to Graphs" (2021)

Key Features:
- Self-attention over graph structure
- Positional encodings for graphs
- State-of-the-art on many graph tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.utils import to_dense_adj, degree
from torch_geometric.data import Data
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.gnn.base_gnn import BaseGNNClustering, modularity_loss
from clusternet.registry import register_algorithm


class GraphMultiHeadAttention(nn.Module):
    """Multi-head attention for graphs with edge-aware attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with graph-masked attention.
        
        Args:
            x: Node features [N, D]
            adj: Adjacency matrix [N, N]
            mask: Optional attention mask
            
        Returns:
            Updated node features [N, D]
        """
        N = x.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)
        
        # Compute attention scores
        attn = torch.einsum('nhd,mhd->nmh', Q, K) / self.scale
        
        # Apply graph structure mask (only attend to neighbors + self)
        graph_mask = adj + torch.eye(N, device=adj.device)
        graph_mask = graph_mask.unsqueeze(-1).expand(-1, -1, self.num_heads)
        attn = attn.masked_fill(graph_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.einsum('nmh,mhd->nhd', attn, V)
        out = out.reshape(N, self.hidden_dim)
        
        return self.out_proj(out)


class GraphTransformerLayer(nn.Module):
    """Single Graph Transformer layer."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        
        # Multi-head attention
        self.attention = GraphMultiHeadAttention(hidden_dim, num_heads, dropout)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Attention with residual
        x = x + self.dropout(self.attention(self.norm1(x), adj))
        
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        
        return x


class GraphTransformerModel(nn.Module):
    """
    Graph Transformer for node clustering.
    
    Architecture:
    1. Input projection
    2. Positional encoding (Laplacian eigenvectors)
    3. Stack of Graph Transformer layers
    4. Clustering head
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_clusters: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        pos_enc_dim: int = 8
    ):
        super().__init__()
        
        self.pos_enc_dim = pos_enc_dim
        
        # Input projection
        self.input_proj = nn.Linear(in_channels + pos_enc_dim, hidden_channels)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_channels, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Clustering head
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_clusters)
        )
        
        self.num_clusters = num_clusters
    
    def _compute_positional_encoding(
        self,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Compute Laplacian positional encodings."""
        device = edge_index.device
        
        # Build adjacency and degree matrices
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        deg = adj.sum(dim=1)
        
        # Compute normalized Laplacian
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        L = torch.eye(num_nodes, device=device) - D_inv_sqrt @ adj @ D_inv_sqrt
        
        # Compute eigenvectors
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
            # Take smallest k eigenvectors (skip first which is constant)
            pos_enc = eigenvectors[:, 1:self.pos_enc_dim + 1]
            
            # Pad if needed
            if pos_enc.size(1) < self.pos_enc_dim:
                pad = torch.zeros(num_nodes, self.pos_enc_dim - pos_enc.size(1), device=device)
                pos_enc = torch.cat([pos_enc, pad], dim=1)
        except:
            # Fallback: random positional encoding
            pos_enc = torch.randn(num_nodes, self.pos_enc_dim, device=device)
        
        return pos_enc
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            
        Returns:
            Cluster logits [N, K]
        """
        num_nodes = x.size(0)
        
        # Get positional encoding
        pos_enc = self._compute_positional_encoding(edge_index, num_nodes)
        
        # Concatenate features with positional encoding
        x = torch.cat([x, pos_enc], dim=-1)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        
        # Get adjacency for attention masking
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, adj)
        
        # Clustering
        return self.cluster_head(x)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings before clustering head."""
        num_nodes = x.size(0)
        pos_enc = self._compute_positional_encoding(edge_index, num_nodes)
        x = torch.cat([x, pos_enc], dim=-1)
        x = self.input_proj(x)
        
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        
        for layer in self.layers:
            x = layer(x, adj)
        
        return x


@register_algorithm('graph_transformer', aliases=['transformer', 'gt_cluster'])
class GraphTransformerCluster(BaseGNNClustering):
    """
    Graph Transformer-based community detection.
    
    Uses self-attention mechanism over graph structure with
    Laplacian positional encodings for clustering.
    
    Parameters:
        num_clusters: Number of clusters (auto-detected if None)
        hidden_channels: Hidden dimension (default: 64)
        num_layers: Number of transformer layers (default: 3)
        num_heads: Number of attention heads (default: 4)
        epochs: Training epochs (default: 200)
        lr: Learning rate (default: 0.001)
        
    Reference:
        Dwivedi et al. (2021). "A Generalization of Transformer Networks to Graphs"
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = False  # Standard transformer doesn't use edge weights
    REQUIRES_FEATURES = True
    IS_TRAINABLE = True
    
    def __init__(
        self,
        G,
        num_clusters: Optional[int] = None,
        features: Optional[np.ndarray] = None,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        epochs: int = 200,
        lr: float = 0.001,
        dropout: float = 0.1,
        pos_enc_dim: int = 8,
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
        self.num_heads = num_heads
        self.pos_enc_dim = pos_enc_dim
        
        # Pre-compute adjacency for loss
        self.adj = to_dense_adj(self.data.edge_index).squeeze(0)
    
    def _build_model(self) -> nn.Module:
        """Build Graph Transformer model."""
        return GraphTransformerModel(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.hidden_channels,
            num_clusters=self.num_clusters,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            pos_enc_dim=self.pos_enc_dim
        )
    
    def _train_step(self, data: Data) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        cluster_logits = self.model(data.x, data.edge_index)
        
        # Modularity-based loss
        loss = modularity_loss(self.adj, cluster_logits)
        
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
def graph_transformer_clustering(
    G,
    num_clusters: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    hidden_channels: int = 64,
    num_layers: int = 3,
    epochs: int = 200,
    lr: float = 0.001,
    **kwargs
) -> Tuple[List[List], float]:
    """
    Run Graph Transformer clustering.
    
    Args:
        G: NetworkX graph
        num_clusters: Number of clusters
        features: Node features
        hidden_channels: Hidden dimension
        num_layers: Number of layers
        epochs: Training epochs
        lr: Learning rate
        
    Returns:
        communities: List of communities
        runtime: Training time
    """
    import time
    
    t1 = time.time()
    algo = GraphTransformerCluster(
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

