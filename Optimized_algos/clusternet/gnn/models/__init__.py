"""
GNN Models for Community Detection

SOTA UNSUPERVISED (published methods, standard loss functions):
- DMoN: Differentiable Modularity Networks (Tsitsulin et al., JMLR 2023)
- MinCutPool: Spectral clustering with GNN (Bianchi et al., ICML 2020)
- VGAE: Variational Graph Auto-Encoder + KMeans (Kipf & Welling, 2016)
- DGI: Deep Graph Infomax + KMeans (Velickovic et al., ICLR 2019)

SEMI-SUPERVISED (uses partial labels - upper bound):
- GCNSupervised: GCN with cross-entropy loss (Kipf & Welling, ICLR 2017)
- GATSupervised: GAT with cross-entropy loss (Brody et al., ICLR 2022)

BASELINES (non-GNN, for ablation):
- MLPKMeansBaseline: Features-only clustering (no graph structure)
- Node2VecCluster: Topology-only embeddings + KMeans (Grover & Leskovec, 2016)

LEGACY UNSUPERVISED (custom variants, kept for backward compatibility):
- GCNCluster, GATCluster, SAGECluster, GINCluster, GraphTransformerCluster
"""

# SOTA unsupervised models (use these for benchmarking)
from .dmon import DMoNCluster, DMoNModel, dmon_clustering
from .mincut import MinCutCluster, MinCutModel, mincut_clustering
from .vgae import VGAECluster, vgae_clustering
from .dgi import DGICluster, dgi_clustering

# Semi-supervised models (upper-bound comparison)
from .gcn_supervised import GCNSupervised, gcn_supervised_clustering
from .gat_supervised import GATSupervised, gat_supervised_clustering

# Non-GNN baselines
from .mlp_baseline import MLPKMeansBaseline, mlp_kmeans_clustering
from .node2vec_baseline import Node2VecCluster, node2vec_clustering

# Legacy unsupervised models (custom variants, backward compatible)
from .gcn_cluster import GCNCluster, GCNClusterModel, gcn_clustering
from .gat_cluster import GATCluster, GATClusterModel, gat_clustering
from .sage_cluster import SAGECluster, SAGEClusterModel, sage_clustering
from .gin_cluster import GINCluster, gin_clustering
from .graph_transformer import GraphTransformerCluster, graph_transformer_clustering

__all__ = [
    # SOTA Unsupervised
    'DMoNCluster',
    'DMoNModel',
    'MinCutCluster',
    'MinCutModel',
    'VGAECluster',
    'DGICluster',
    # Semi-supervised
    'GCNSupervised',
    'GATSupervised',
    # Baselines
    'MLPKMeansBaseline',
    'Node2VecCluster',
    # SOTA functions
    'dmon_clustering',
    'mincut_clustering',
    'vgae_clustering',
    'dgi_clustering',
    'gcn_supervised_clustering',
    'gat_supervised_clustering',
    'mlp_kmeans_clustering',
    'node2vec_clustering',
    # Legacy classes (backward compatible)
    'GCNCluster',
    'GCNClusterModel',
    'GATCluster',
    'GATClusterModel',
    'SAGECluster',
    'SAGEClusterModel',
    'GINCluster',
    'GraphTransformerCluster',
    # Legacy functions
    'gcn_clustering',
    'gat_clustering',
    'sage_clustering',
    'gin_clustering',
    'graph_transformer_clustering',
]
