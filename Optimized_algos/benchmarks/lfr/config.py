"""
LFR Benchmark Configuration
===========================

Configuration for all LFR benchmark experiments.
Designed for minimal testing on Mac before scaling to cluster.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LFR_BINARY_DIR = os.path.join(os.path.dirname(BASE_DIR), '..', '..', '..', 'LFRbenchmarks')

# Network generation parameters
NETWORK_PARAMS = {
    'base': {
        'k': 15,           # average degree
        'maxk': 50,        # maximum degree
        't1': 2,           # degree distribution exponent
        't2': 1,           # community size distribution exponent
        'minc': 20,        # minimum community size
        'maxc': 50,        # maximum community size
    }
}

# ============= ACCURACY BENCHMARK =============
# Vary mixing parameter (μ) to test detection difficulty
ACCURACY_CONFIG = {
    'name': 'accuracy',
    'description': 'Test algorithm accuracy as community structure weakens (increasing μ)',
    'network_size': 1000,  # Fixed N for accuracy test
    'mu_values': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # μ = 0.8, 0.9 often fail to generate
    'realizations': 10,    # 10 network realizations per μ
    'binary': 'unweighted_undirected/benchmark',
}

# ============= SCALABILITY BENCHMARK =============
# Vary network size to test computational scaling
SCALABILITY_CONFIG = {
    'name': 'scalability',
    'description': 'Test algorithm runtime as network size increases',
    'size_values': [ 50000],  # N values (Mac-friendly)
    'mu': 0.15,             # Fixed μ for scalability test
    'realizations': 10,    # 10 network realizations per N
    'binary': 'unweighted_undirected/benchmark',
}

# ============= HIERARCHICAL BENCHMARK =============
# Test detection of hierarchical community structure
HIERARCHICAL_CONFIG = {
    'name': 'hierarchical',
    'description': 'Test detection of hierarchical (nested) community structure',
    'network_size': 1000,  # Reduced from 5000 for faster testing
    'mu1_values': [0.1, 0.2, 0.3],  # macro-community mixing
    'mu2_values': [0.1, 0.2, 0.3],  # micro-community mixing
    'realizations': 10,
    'binary': 'hierarchical/hbenchmark',
    'params': {
        'k': 15,         # Adjusted for smaller network
        'maxk': 50,
        'minc': 10,      # micro community min
        'maxc': 50,      # micro community max
        'minC': 100,     # macro community min
        'maxC': 300,     # macro community max
    }
}

# ============= OVERLAPPING BENCHMARK =============
# Test detection of overlapping communities
OVERLAPPING_CONFIG = {
    'name': 'overlapping',
    'description': 'Test detection of overlapping community structure',
    'network_size': 1000,
    'mu': 0.3,
    'overlap_fractions': [0.0, 0.1, 0.2, 0.3, 0.4],  # fraction of overlapping nodes
    'om': 2,               # number of memberships for overlapping nodes
    'realizations': 10,
    'binary': 'unweighted_undirected/benchmark',
}

# ============= ALGORITHMS TO TEST =============
# Categorized for systematic testing

ALGORITHMS = {
    # Classical disjoint algorithms
    'disjoint': [
        'louvain',
        'leiden',
        #'walktrap',
        'fastgreedy',
        'label_propagation',
        'spectral',
        'spinglass',
        'leading_eigen',
    ],
    
    # Statistical inference
    'statistical': [
        'sbm',
        'nested_sbm',
        'em',
    ],
    
    # CDlib additional
    'cdlib': [
        'cpm',
        'rb_pots',
        'rber_pots',
        'scan',
        'agdl',
        'async_fluid',
        #'gdmp2',
        #'der',
        'surprise'
    ],
    
    # Custom implementations
    'custom': [
        'score',
        #'svt',
        'teamcs',
        #'simnet',
        #'tusk',
        'bigs2',
        'csbio_iitm2',
    ],
    
    # Overlapping algorithms (for overlapping benchmark)
    'overlapping': [
        'angel',
        'demon',
        'kclique',
    ],
    
    # GNN models (SOTA published methods)
    'gnn_unsupervised': [
        'dmon_gnn',
        'mincut_gnn',
        'vgae_gnn',
        'dgi_gnn',
    ],

    # GNN baselines (ablation: features-only and topology-only)
    'gnn_baselines': [
        'mlp_kmeans',
        'node2vec_gnn',
    ],

    # GNN semi-supervised (need ground truth labels)
    'gnn_supervised': [
        'gcn_supervised',
        'gat_supervised',
    ],

    # Semi-supervised label-fraction ablations
    'gnn_supervised_40': ['gcn_supervised_40', 'gat_supervised_40'],
    'gnn_supervised_60': ['gcn_supervised_60', 'gat_supervised_60'],
    'gnn_supervised_80': ['gcn_supervised_80', 'gat_supervised_80'],

    'hierarchical': ['louvain', 'leiden', 'nested_sbm', 'nested_sbm_coarse', 'infomap', 'walktrap', 'csbio_iitm2'],

    # GNNs for hierarchical benchmark (unsupervised, accept num_clusters)
    'gnn_hierarchical': [
        'dmon_gnn',
        'mincut_gnn',
        'vgae_gnn',
        'dgi_gnn',
    ],
}

# Select which algorithm categories to run
# For minimal testing, can reduce this
ACTIVE_CATEGORIES = ['disjoint', 'statistical', 'cdlib', 'custom', 'overlapping']
OVERLAPPING_CATEGORIES = ['overlapping', 'disjoint']  # For overlapping benchmark
HIERARCHICAL_CATEGORIES = ['hierarchical']

# GNN categories (unsupervised + baselines run without labels;
# supervised needs ground truth passed separately)
GNN_CATEGORIES = ['gnn_unsupervised', 'gnn_baselines']
GNN_SUPERVISED_CATEGORIES = ['gnn_supervised', 'gnn_supervised_40',
                             'gnn_supervised_60', 'gnn_supervised_80']

# ============= METRICS =============
METRICS = {
    'disjoint': ['ami', 'nmi', 'ari', 'modularity', 'num_communities'],
    'overlapping': ['onmi', 'omega_index', 'f1', 'modularity', 'num_communities'],
    'all': ['runtime'],
}

# ============= OUTPUT SETTINGS =============
SAVE_NETWORKS = True      # Save generated networks for reproducibility
SAVE_COMMUNITIES = True   # Save detected communities
VERBOSE = True            # Print progress
NUM_WORKERS = 4           # Parallel workers (adjust for your Mac)

