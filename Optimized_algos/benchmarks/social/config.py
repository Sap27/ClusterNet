"""
Configuration for social/citation network benchmarks.

Datasets: Cora, Citeseer, Amazon Photo
All auto-downloaded via PyTorch Geometric.
"""
from pathlib import Path

# =============================================================================
# PATHS  (change SOCIAL_DATA_DIR to match your machine)
# =============================================================================
SOCIAL_DATA_DIR = Path('/mnt/d/cluster/SocialNetworks')

OUTPUT_DIR = SOCIAL_DATA_DIR / 'data'
RESULTS_DIR = SOCIAL_DATA_DIR / 'results'

BENCHMARK_DIR = Path(__file__).resolve().parent

# =============================================================================
# NETWORKS
# =============================================================================
NETWORKS = {
    'cora': {
        'pyg_dataset': 'Planetoid',
        'pyg_name': 'Cora',
        'description': 'Cora citation network (bag-of-words features, 7 topic labels)',
    },
    'citeseer': {
        'pyg_dataset': 'Planetoid',
        'pyg_name': 'CiteSeer',
        'description': 'CiteSeer citation network (bag-of-words features, 6 topic labels)',
    },
    'amazon_photo': {
        'pyg_dataset': 'Amazon',
        'pyg_name': 'Photo',
        'description': 'Amazon Photo co-purchase network (product features, 8 categories)',
    },
}

NETWORK_ORDER = ['cora', 'citeseer', 'amazon_photo']

# =============================================================================
# ALGORITHMS  (same pool as LFR / biological)
# =============================================================================
CLASSICAL_ALGORITHMS = [
    'louvain', 'leiden', 'label_propagation', 'fastgreedy',
    'spectral', 'infomap', 'spinglass', 'leading_eigen',
    'em', 'sbm', 'nested_sbm', 'nested_sbm_coarse',
    'cpm', 'rb_pots', 'rber_pots', 'scan', 'agdl', 'async_fluid', 'surprise',
    'score', 'teamcs', 'bigs2', 'csbio_iitm2',
    'angel', 'demon', 'kclique',
]

GNN_UNSUPERVISED = ['dmon_gnn', 'mincut_gnn', 'vgae_gnn', 'dgi_gnn']
GNN_BASELINES    = ['mlp_kmeans', 'node2vec_gnn']
GNN_SUPERVISED   = ['gcn_supervised', 'gat_supervised']

ALL_GNN = GNN_UNSUPERVISED + GNN_BASELINES + GNN_SUPERVISED
ALL_ALGORITHMS = CLASSICAL_ALGORITHMS + ALL_GNN

# =============================================================================
# FEATURE ABLATION SETTINGS
# =============================================================================
FEATURE_SETTINGS = ['real', 'structural', 'random']

STRUCTURAL_FEATURE_DIM = 16   # degree + clustering coeff + pagerank + spectral embedding dims

# =============================================================================
# BENCHMARK PARAMETERS
# =============================================================================
GNN_EPOCHS = 600
GNN_SEEDS = 3
ALGO_TIMEOUT = 600  # seconds per algorithm
