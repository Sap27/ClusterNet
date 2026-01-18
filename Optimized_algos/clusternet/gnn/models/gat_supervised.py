"""
GAT (Graph Attention Network) - Supervised Version

Adapted from RobustGNN based on Veličković et al. (2018).
Uses attention mechanism to weight neighbor contributions.

Key Features:
- Multi-head attention over neighbors
- Learns to weight important neighbors
- More expressive than GCN for heterogeneous graphs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.gnn.base_gnn import BaseGNNClustering
from clusternet.registry import register_algorithm


class GATSupervisedModel(nn.Module):
    """
    Supervised GAT for node classification.
    
    Architecture from Veličković et al. (2018):
    - Multi-head attention layers
    - Concatenation (hidden) / averaging (output) of heads
    - ELU activation
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        
        # Input layer: in_channels -> hidden_channels * heads
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        
        # Hidden layers: hidden_channels * heads -> hidden_channels * heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels * heads, hidden_channels, heads=heads)
            )
        
        # Output layer: hidden_channels * heads -> out_channels (single head, no concat)
        self.convs.append(
            GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        )
    
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
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get embeddings from second-to-last layer."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
        return x
    
    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get attention weights from each layer.
        
        Returns:
            List of attention weight tensors
        """
        attention_weights = []
        
        for i, conv in enumerate(self.convs):
            x, (edge_index_out, alpha) = conv(
                x, edge_index, return_attention_weights=True
            )
            attention_weights.append(alpha)
            
            if i < self.num_layers - 1:
                x = F.elu(x)
        
        return attention_weights


@register_algorithm('gat_supervised', aliases=['gat_sup', 'gat'])
class GATSupervised(BaseGNNClustering):
    """
    Supervised GAT for community detection.
    
    Uses attention mechanism to learn importance of different neighbors.
    Requires ground truth labels for training.
    
    Parameters:
        num_clusters: Number of communities/classes
        labels: Ground truth labels for training (REQUIRED)
        train_mask: Boolean mask for training nodes (optional)
        hidden_channels: Hidden layer dimension (default: 64)
        num_layers: Number of GAT layers (default: 3)
        heads: Number of attention heads (default: 8)
        epochs: Training epochs (default: 200)
        lr: Learning rate (default: 0.005)
        
    Reference:
        Veličković et al. (2018). "Graph Attention Networks"
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    REQUIRES_FEATURES = True
    IS_TRAINABLE = True
    IS_SUPERVISED = True
    
    def __init__(
        self,
        G,
        num_clusters: int,
        labels: np.ndarray,
        features: Optional[np.ndarray] = None,
        train_mask: Optional[np.ndarray] = None,
        hidden_channels: int = 64,
        num_layers: int = 3,
        heads: int = 8,
        epochs: int = 200,
        lr: float = 0.005,
        dropout: float = 0.6,
        **kwargs
    ):
        # Store labels before parent init
        self._labels = torch.tensor(labels, dtype=torch.long)
        
        # Train mask
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
        self.heads = heads
        
        # Move to device
        self._labels = self._labels.to(self.device)
        self._train_mask = self._train_mask.to(self.device)
    
    def _build_model(self) -> nn.Module:
        """Build supervised GAT model."""
        return GATSupervisedModel(
            in_channels=self.data.x.shape[1],
            hidden_channels=self.hidden_channels,
            out_channels=self.num_clusters,
            num_layers=self.num_layers,
            heads=self.heads,
            dropout=self.dropout
        )
    
    def _train_step(self, data: Data) -> float:
        """Single training step with cross-entropy loss."""
        self.model.train()
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index)
        
        # Cross-entropy loss on training nodes
        loss = F.nll_loss(out[self._train_mask], self._labels[self._train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _get_cluster_assignments(self, data: Data) -> np.ndarray:
        """Get cluster assignments."""
        out = self.model(data.x, data.edge_index)
        labels = out.argmax(dim=-1).cpu().numpy()
        return labels
    
    def _get_embeddings(self, data: Data) -> torch.Tensor:
        """Get node embeddings."""
        return self.model.get_embeddings(data.x, data.edge_index)
    
    def compute_accuracy(self, mask: Optional[torch.Tensor] = None) -> float:
        """Compute classification accuracy."""
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
def gat_supervised_clustering(
    G,
    num_clusters: int,
    labels: np.ndarray,
    features: Optional[np.ndarray] = None,
    train_ratio: float = 0.8,
    hidden_channels: int = 64,
    num_layers: int = 3,
    heads: int = 8,
    epochs: int = 200,
    lr: float = 0.005,
    seed: int = 42,
    **kwargs
) -> Tuple[List[List], float, float]:
    """
    Run supervised GAT clustering.
    
    Args:
        G: NetworkX graph
        num_clusters: Number of clusters
        labels: Ground truth labels
        features: Node features
        train_ratio: Fraction of nodes for training
        hidden_channels: Hidden dimension
        num_layers: Number of GAT layers
        heads: Number of attention heads
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
    algo = GATSupervised(
        G,
        num_clusters=num_clusters,
        labels=labels,
        features=features,
        train_mask=train_mask,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        heads=heads,
        epochs=epochs,
        lr=lr,
        **kwargs
    )
    communities = algo.run()
    t2 = time.time()
    
    train_acc = algo.compute_accuracy(torch.tensor(train_mask, device=algo.device))
    
    return communities, t2 - t1, train_acc









