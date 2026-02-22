"""
GNN Models for Community Detection

UNSUPERVISED (for fair benchmarking - no labels during training):
- GCN: Graph Convolutional Network with modularity loss
- GAT: Graph Attention Network with modularity loss
- GraphSAGE: GraphSAGE with modularity loss
- DMoN: Differentiable Modularity Networks
- MinCutPool: Spectral clustering with GNN
- GIN: Graph Isomorphism Network
- GraphTransformer: Attention-based clustering

SUPERVISED (uses labels - for upper bound / semi-supervised):
- GCNSupervised: GCN with cross-entropy loss
- GATSupervised: Graph Attention Network with cross-entropy loss
"""

# Unsupervised models (use these for fair benchmarking)
from .gcn_cluster import GCNCluster, GCNClusterModel, gcn_clustering
from .gat_cluster import GATCluster, GATClusterModel, gat_clustering
from .sage_cluster import SAGECluster, SAGEClusterModel, sage_clustering
from .dmon import DMoNCluster, DMoNModel, dmon_clustering
from .gin_cluster import GINCluster, gin_clustering
from .mincut import MinCutCluster, MinCutModel, mincut_clustering
from .graph_transformer import GraphTransformerCluster, graph_transformer_clustering

# Supervised models (use these for upper-bound comparison)
from .gcn_supervised import GCNSupervised, gcn_supervised_clustering
from .gat_supervised import GATSupervised, gat_supervised_clustering

__all__ = [
    # Unsupervised Classes
    'GCNCluster',
    'GATCluster',
    'SAGECluster',
    'DMoNCluster',
    'GINCluster', 
    'MinCutCluster',
    'GraphTransformerCluster',
    # Unsupervised Model Classes (for direct use)
    'GCNClusterModel',
    'GATClusterModel',
    'SAGEClusterModel',
    'DMoNModel',
    'MinCutModel',
    # Supervised Classes
    'GCNSupervised',
    'GATSupervised',
    # Unsupervised Functions
    'gcn_clustering',
    'gat_clustering',
    'sage_clustering',
    'dmon_clustering',
    'gin_clustering',
    'mincut_clustering',
    'graph_transformer_clustering',
    # Supervised Functions
    'gcn_supervised_clustering',
    'gat_supervised_clustering',
]
