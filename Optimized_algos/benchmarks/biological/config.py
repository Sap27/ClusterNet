"""
Configuration for biological network benchmarks.
"""
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BIO_DATA_DIR = Path('/mnt/d/cluster/BIologicalNetworks')

RAW_NETWORKS = {
    'gtex': BIO_DATA_DIR / 'GTEx_Full_Benchmark_2 (1).csv',
    'grn': BIO_DATA_DIR / 'Human_GRN_Weighted.csv',
    'biogrid': BIO_DATA_DIR / 'BioGRID_Experimental.csv',
    'signor': BIO_DATA_DIR / 'SIGNOR_Directed.csv',
}

GTEx_TPM_PATH = BIO_DATA_DIR / 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm (2).gct'
GO_GAF_PATH = BIO_DATA_DIR / 'goa_human.gaf.gz'
GO_OBO_PATH = BIO_DATA_DIR / 'go-basic.obo'

BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BIO_DATA_DIR / 'data'
RESULTS_DIR = BIO_DATA_DIR / 'results'

# =============================================================================
# NETWORK CONFIG
# =============================================================================
NETWORKS = {
    'gtex_0.9': {
        'source': 'gtex',
        'threshold': 0.9,
        'directed': False,
        'signed': False,
        'id_format': 'symbol',
        'feature_source': 'expression',
        'description': 'GTEx gene co-expression (Pearson >= 0.9)',
    },
    'grn': {
        'source': 'grn',
        'threshold': None,
        'directed': True,               # TF -> target (kept for FFL evaluation)
        'signed': False,
        'id_format': 'uniprot',         # UniProt accessions (O00716, P05412, ...)
        'feature_source': 'expression', # TPM profiles via UniProt->symbol mapping
        'description': 'Human gene regulatory network (TF-target)',
    },
    'biogrid': {
        'source': 'biogrid',
        'threshold': None,
        'directed': False,
        'signed': False,
        'id_format': 'symbol',          # gene symbols (STAT3, PML, ...)
        'feature_source': 'go_terms',   # GO term binary vectors
        'description': 'BioGRID protein-protein interaction network',
    },
    'signor': {
        'source': 'signor',
        'threshold': None,
        'directed': True,               # kept for sign coherence evaluation
        'signed': True,                 # +1 activation, -1 inhibition
        'id_format': 'symbol',          # gene symbols (TBK1, ERN1, ...)
        'feature_source': 'go_terms',   # GO terms + sign-aware degree features
        'description': 'SIGNOR directed signaling network',
    },
}

# =============================================================================
# FEATURE CONFIG
# =============================================================================
EXPRESSION_PCA_DIM = 128
GO_TERM_MIN_GENES = 10       # drop GO terms annotating fewer genes
GO_TERM_MAX_GENES = 1000     # drop GO terms annotating more genes

# =============================================================================
# ALGORITHMS
# =============================================================================
CLASSICAL_ALGORITHMS = [
    # Disjoint
    'louvain', 'leiden', 'label_propagation', 'fastgreedy',
    'spectral', 'infomap', 'spinglass', 'leading_eigen',
    # Statistical inference
    'em', 'sbm', 'nested_sbm', 'nested_sbm_coarse',
    # CDlib
    'cpm', 'rb_pots', 'rber_pots', 'scan', 'agdl', 'async_fluid', 'surprise',
    # Custom
    'score', 'teamcs', 'bigs2', 'csbio_iitm2',
    # Overlapping
    'angel', 'demon', 'kclique',
]

# Memory-safe subset (kept for reference on low-RAM machines)
CLASSICAL_ALGORITHMS_MEMSAFE = [
    'louvain', 'leiden', 'label_propagation', 'infomap',
    'cpm', 'rb_pots', 'rber_pots', 'async_fluid', 'surprise',
]

# Order networks by edge count (ascending) for adaptive pruning
NETWORK_ORDER = ['signor', 'grn', 'biogrid', 'gtex_0.9']
MEMSAFE_THRESHOLD = 500000  # edge count above which we use MEMSAFE list

GNN_UNSUPERVISED = ['dmon_gnn', 'mincut_gnn', 'vgae_gnn', 'dgi_gnn']
GNN_BASELINES = ['mlp_kmeans', 'node2vec_gnn']
GNN_SUPERVISED = ['gcn_supervised', 'gat_supervised']

ALL_GNN = GNN_UNSUPERVISED + GNN_BASELINES + GNN_SUPERVISED
ALL_ALGORITHMS = CLASSICAL_ALGORITHMS + ALL_GNN

# =============================================================================
# BENCHMARK PARAMETERS
# =============================================================================
GNN_EPOCHS = 600
GNN_SEEDS = 3               # random seeds for variance estimates
PERTURBATION_SEEDS = 3

PERTURBATION_LEVELS = {
    'edge_dropout': [0.05, 0.10, 0.20, 0.30],
    'node_removal_random': [0.01, 0.02, 0.05],
    'node_removal_hub': [0.01, 0.02, 0.05],
    'edge_noise': [0.05, 0.10],
}

TF_KNOCKOUT_LEVELS = [5, 10]  # top-K TFs by out-degree (GRN only)
