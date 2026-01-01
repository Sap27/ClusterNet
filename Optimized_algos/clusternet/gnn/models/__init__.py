"""
GNN Models for Community Detection

UNSUPERVISED (for fair benchmarking - no labels during training):
- GCN: Graph Convolutional Network with modularity loss
- DMoN: Differentiable Modularity Networks
- MinCutPool: Spectral clustering with GNN
- GIN: Graph Isomorphism Network
- GraphTransformer: Attention-based clustering

SUPERVISED (uses labels - for upper bound / semi-supervised):
- GCNSupervised: GCN with cross-entropy loss
- GATSupervised: Graph Attention Network
"""

# Unsupervised models (use these for fair benchmarking)
from .gcn_cluster import GCNCluster, gcn_clustering
from .dmon import DMoNCluster, dmon_clustering
from .gin_cluster import GINCluster, gin_clustering
from .mincut import MinCutCluster, mincut_clustering
from .graph_transformer import GraphTransformerCluster, graph_transformer_clustering

# Supervised models (use these for upper-bound comparison)
from .gcn_supervised import GCNSupervised, gcn_supervised_clustering
from .gat_supervised import GATSupervised, gat_supervised_clustering

__all__ = [
    # Unsupervised Classes
    'GCNCluster',
    'DMoNCluster',
    'GINCluster', 
    'MinCutCluster',
    'GraphTransformerCluster',
    # Supervised Classes
    'GCNSupervised',
    'GATSupervised',
    # Unsupervised Functions
    'gcn_clustering',
    'dmon_clustering',
    'gin_clustering',
    'mincut_clustering',
    'graph_transformer_clustering',
    # Supervised Functions
    'gcn_supervised_clustering',
    'gat_supervised_clustering',
]
