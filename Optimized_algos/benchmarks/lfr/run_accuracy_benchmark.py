#!/usr/bin/env python3
"""
LFR Accuracy Benchmark
======================

Test algorithm accuracy as community structure weakens (increasing μ).

X-axis: Mixing parameter μ (0.1 to 0.7)
Y-axis: AMI score
N realizations per μ value

Output:
- CSV with all results (classical algorithms)
- CSV with GNN results (homophily ablation)
- CSV with GNN results (node feature types)
- Plots showing AMI vs μ for each algorithm
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from config import ACCURACY_CONFIG, NETWORK_PARAMS, ALGORITHMS, ACTIVE_CATEGORIES
from lfr_generator import LFRGenerator, save_network, load_network
from algorithm_runner import AlgorithmRunner
from metrics import compute_all_metrics

# Import actual GNN models from clusternet
from clusternet.gnn.models.dmon import DMoNModel
from clusternet.gnn.models.mincut import MinCutModel
from clusternet.gnn.models.gat_cluster import GATClusterModel
from clusternet.gnn.models.sage_cluster import SAGEClusterModel
from clusternet.gnn.models.gcn_cluster import GCNClusterModel
from clusternet.gnn.models.gin_cluster import GINClusterModel
from clusternet.gnn.models.graph_transformer import GraphTransformerModel

# PyTorch/PyG imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, dense_mincut_pool, DMoNPooling
    from torch_geometric.utils import to_dense_adj, to_dense_batch
    from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
    from sklearn.cluster import KMeans
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import diags
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    HAS_TORCH = False
    DEVICE = None


# =============================================================================
# GNN MODELS
# =============================================================================

if HAS_TORCH:
    class GCNModel(nn.Module):
        """GCN for semi-supervised community detection."""
        def __init__(self, in_dim, hidden_dim, num_classes):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return self.classifier(x)
    
    class GATModel(nn.Module):
        """GAT for semi-supervised community detection."""
        def __init__(self, in_dim, hidden_dim, num_classes):
            super().__init__()
            self.conv1 = GATConv(in_dim, hidden_dim, heads=4, concat=False)
            self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x, edge_index):
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return self.classifier(x)
    
    class SAGEModel(nn.Module):
        """GraphSAGE for semi-supervised community detection."""
        def __init__(self, in_dim, hidden_dim, num_classes):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.classifier = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return self.classifier(x)
    
    class DMoNModelLegacy(nn.Module):
        """Legacy DMoN for semi-supervised (kept for backwards compatibility)."""
        def __init__(self, in_dim, hidden_dim, num_clusters):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.cluster = nn.Linear(hidden_dim, num_clusters)
            nn.init.xavier_uniform_(self.cluster.weight, gain=2.0)
        
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.relu(self.conv2(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.relu(self.conv3(x, edge_index))
            return F.softmax(self.cluster(x) * 2.0, dim=-1)

    # GNN algorithm registry (legacy models for semi-supervised)
    GNN_ALGORITHMS = {
        'gcn': GCNModel,
        'gat': GATModel,
        'sage': SAGEModel,
        'dmon': DMoNModelLegacy,
    }


# =============================================================================
# NODE FEATURES
# =============================================================================

def dmon_loss(s, edge_index, num_nodes):
    """
    DMoN loss from paper (NO orthogonality - that's MinCut!):
    
    L = -1/(2m) * Tr(C^T B C) + sqrt(k)/N * ||cluster_sizes||_2 - 1
    
    Term 1: Modularity maximization
    Term 2: Collapse regularizer (penalizes putting all nodes in one cluster)
    """
    row, col = edge_index[0], edge_index[1]
    m = edge_index.size(1) / 2
    k = s.size(1)
    n = num_nodes
    
    # Degree vector
    deg = torch.zeros(n, device=s.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=s.device))
    
    # === MODULARITY: Tr(C^T B C) / 2m ===
    # B = A - dd^T / 2m (modularity matrix)
    adj_term = (s[row] * s[col]).sum()  # Tr(C^T A C)
    cluster_degrees = torch.mm(deg.unsqueeze(0), s).squeeze()
    degree_term = (cluster_degrees ** 2).sum()  # Tr(C^T dd^T C)
    modularity = (adj_term - degree_term / (2 * m)) / (2 * m)
    
    # === COLLAPSE REGULARIZER (DMoN paper) ===
    # sqrt(k)/N * ||cluster_sizes||_2 - 1
    # This is 0 when balanced, positive when collapsed
    cluster_sizes = s.sum(dim=0)
    collapse_reg = (np.sqrt(k) / n) * torch.norm(cluster_sizes, p=2) - 1
    
    # Scale collapse_reg to be effective (paper coefficient is tiny for large N)
    # When collapsed: collapse_reg ≈ sqrt(k) - 1 ≈ 6 for k=50
    # When balanced: collapse_reg = 0
    # Modularity is in range [0, ~0.5], so scale collapse_reg down
    
    return -modularity + 0.5 * collapse_reg


def modularity_loss(s, edge_index, num_nodes):
    """Wrapper that uses DMoN loss."""
    return dmon_loss(s, edge_index, num_nodes)


def get_edge_index(G):
    """Convert graph to edge index tensor."""
    import networkx as nx
    edges = list(G.edges())
    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([
        [e[0] for e in edges] + [e[1] for e in edges],
        [e[1] for e in edges] + [e[0] for e in edges]
    ], dtype=torch.long)


def random_features(n, dim=32):
    """Random Gaussian features (no class info)."""
    return torch.randn(n, dim) * 0.1


def structural_features(G, dim=32):
    """Structural features based on graph topology."""
    import networkx as nx
    n = G.number_of_nodes()
    degrees = dict(G.degree())
    clustering = nx.clustering(G)
    
    try:
        pagerank = nx.pagerank(G, max_iter=50)
    except:
        pagerank = {i: 1/n for i in range(n)}
    
    features = []
    for i in range(n):
        neighbors = list(G.neighbors(i))
        f = [
            degrees.get(i, 0),
            degrees.get(i, 0) / max(1, max(degrees.values())),
            clustering.get(i, 0),
            pagerank.get(i, 0),
            len(neighbors),
            np.mean([degrees.get(j, 0) for j in neighbors]) if neighbors else 0,
            np.std([degrees.get(j, 0) for j in neighbors]) if len(neighbors) > 1 else 0,
        ]
        features.append(f)
    
    features = np.array(features)
    if features.shape[1] < dim:
        features = np.hstack([features, np.zeros((n, dim - features.shape[1]))])
    
    return torch.tensor(features[:, :dim], dtype=torch.float)


def spectral_features(G, dim=32):
    """Spectral features (Laplacian eigenvectors)."""
    import networkx as nx
    n = G.number_of_nodes()
    
    try:
        A = nx.adjacency_matrix(G).astype(np.float64)
        degrees = np.array(A.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
        L_norm = diags(np.ones(n)) - D_inv_sqrt @ A @ D_inv_sqrt
        
        k = min(dim, n - 2)
        eigenvalues, eigenvectors = eigsh(L_norm, k=k+1, which='SM', tol=1e-6)
        
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx][:, 1:k+1]
        eigenvectors = eigenvectors / (np.linalg.norm(eigenvectors, axis=0, keepdims=True) + 1e-8)
        
        if eigenvectors.shape[1] < dim:
            padding = np.zeros((n, dim - eigenvectors.shape[1]))
            eigenvectors = np.hstack([eigenvectors, padding])
        
        return torch.tensor(eigenvectors[:, :dim], dtype=torch.float)
    
    except Exception as e:
        return structural_features(G, dim)


def homophily_features(n, gt, k, feature_dim=32, feature_distance=1.0, noise_std=0.8, seed=42):
    """Generate cluster-correlated features for homophily testing."""
    np.random.seed(seed)
    centers = np.random.randn(k, feature_dim) * feature_distance
    features = np.array([
        centers[gt[i]] + np.random.randn(feature_dim) * noise_std 
        for i in range(n)
    ])
    return torch.tensor(features, dtype=torch.float)


# =============================================================================
# GNN TRAINING
# =============================================================================

def train_gnn_semisupervised(
    G, gt, k, features, model_class,
    labeled_ratio=0.1, epochs=200, hidden_dim=64
):
    """Train GNN in semi-supervised mode."""
    n = G.number_of_nodes()
    edge_index = get_edge_index(G).to(DEVICE)
    features = features.to(DEVICE)
    y = torch.tensor(gt, dtype=torch.long, device=DEVICE)
    
    # Create train mask
    train_mask = np.zeros(n, dtype=bool)
    for c in range(k):
        c_nodes = np.where(gt == c)[0]
        if len(c_nodes) > 0:
            n_label = max(1, int(len(c_nodes) * labeled_ratio))
            train_mask[np.random.choice(c_nodes, min(n_label, len(c_nodes)), replace=False)] = True
    train_mask = torch.tensor(train_mask, device=DEVICE)
    
    model = model_class(features.size(1), hidden_dim, k).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        out = model(features, edge_index)
        return out.argmax(dim=1).cpu().numpy()


def train_dmon_unsupervised(G, k, features, epochs=300, hidden_dim=64):
    """Train DMoN in unsupervised mode with modularity loss."""
    n = G.number_of_nodes()
    edge_index = get_edge_index(G).to(DEVICE)
    features = features.to(DEVICE)
    
    # Build adjacency
    adj = torch.zeros(n, n, device=DEVICE)
    adj[edge_index[0], edge_index[1]] = 1
    d = adj.sum(dim=1, keepdim=True)
    m = adj.sum() / 2
    B = adj - torch.mm(d, d.t()) / (2 * m + 1e-10)
    
    model = DMoNModel(features.size(1), hidden_dim, k).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        s = model(features, edge_index)
        
        # Modularity loss
        mod_loss = -torch.trace(torch.mm(torch.mm(s.t(), B), s)) / (2 * m + 1e-10)
        
        # Entropy regularization
        cluster_sizes = s.sum(dim=0) / n
        entropy = -torch.sum(cluster_sizes * torch.log(cluster_sizes + 1e-10))
        collapse_loss = -entropy / np.log(k)
        
        loss = mod_loss + 0.5 * collapse_loss
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        s = model(features, edge_index)
        return s.argmax(dim=1).cpu().numpy()


# =============================================================================
# NETWORK GENERATION
# =============================================================================

def generate_accuracy_networks(output_dir: str, config: dict, lfr_dir: str, verbose: bool = True):
    """Generate all LFR networks for accuracy benchmark."""
    generator = LFRGenerator(lfr_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    networks_generated = []
    
    for mu in config['mu_values']:
        if verbose:
            print(f"\nGenerating networks for μ={mu}...")
        
        for realization in range(config['realizations']):
            name = f"mu{mu:.1f}_r{realization}"
            seed = int(mu * 1000) + realization
            
            try:
                G, communities = generator.generate_standard(
                    N=config['network_size'],
                    mu=mu,
                    k=NETWORK_PARAMS['base']['k'],
                    maxk=NETWORK_PARAMS['base']['maxk'],
                    t1=NETWORK_PARAMS['base']['t1'],
                    t2=NETWORK_PARAMS['base']['t2'],
                    minc=NETWORK_PARAMS['base']['minc'],
                    maxc=NETWORK_PARAMS['base']['maxc'],
                    seed=seed
                )
                
                save_network(G, communities, str(output_path), name)
                
                networks_generated.append({
                    'name': name,
                    'mu': mu,
                    'realization': realization,
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'communities': len(communities),
                })
                
                if verbose:
                    print(f"  ✓ {name}: {G.number_of_nodes()} nodes, {len(communities)} communities")
                    
            except Exception as e:
                print(f"  ✗ {name}: Failed - {e}")
    
    metadata = {
        'benchmark': 'accuracy',
        'config': config,
        'networks': networks_generated,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return networks_generated


# =============================================================================
# CLASSICAL BENCHMARK
# =============================================================================

def run_classical_benchmark(
    networks_dir: str,
    results_dir: str,
    algorithms: list,
    verbose: bool = True
):
    """Run classical algorithms benchmark."""
    networks_path = Path(networks_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(networks_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    runner = AlgorithmRunner(verbose=False)
    all_results = []
    
    for network_info in metadata['networks']:
        name = network_info['name']
        mu = network_info['mu']
        
        if verbose:
            print(f"\nProcessing {name} (μ={mu})...")
        
        G, true_communities = load_network(str(networks_path), name)
        num_true_comms = len(true_communities)
        
        for algo_name in algorithms:
            if verbose:
                print(f"  Running {algo_name}...", end=' ', flush=True)
            
            result = runner.run_algorithm(G, algo_name)
            
            if result.success and result.communities:
                metrics = compute_all_metrics(G, true_communities, result.communities)
                
                all_results.append({
                    'network': name,
                    'mu': mu,
                    'realization': network_info['realization'],
                    'algorithm': algo_name,
                    'algorithm_type': 'classical',
                    'success': True,
                    'runtime': result.runtime,
                    'num_detected': len(result.communities),
                    'num_true': num_true_comms,
                    **metrics
                })
                
                if verbose:
                    print(f"✓ AMI={metrics['ami']:.3f}")
            else:
                all_results.append({
                    'network': name,
                    'mu': mu,
                    'realization': network_info['realization'],
                    'algorithm': algo_name,
                    'algorithm_type': 'classical',
                    'success': False,
                    'runtime': result.runtime,
                    'error': result.error_message
                })
                if verbose:
                    print(f"✗")
    
    df = pd.DataFrame(all_results)
    df.to_csv(results_path / 'classical_accuracy_results.csv', index=False)
    
    # Summary
    summary = df[df['success'] == True].groupby(['mu', 'algorithm']).agg({
        'ami': ['mean', 'std'],
        'nmi': ['mean', 'std'],
        'runtime': ['mean', 'std'],
        'num_detected': 'mean'
    }).round(4)
    summary.to_csv(results_path / 'classical_accuracy_summary.csv')
    
    if verbose:
        print(f"\nClassical results saved to: {results_path / 'classical_accuracy_results.csv'}")
    
    return df


# =============================================================================
# GLOBAL GNN TRAINING (for zero-shot inference)
# =============================================================================

class GlobalGNNEncoder(nn.Module):
    """GNN encoder for global training via link prediction."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


# =============================================================================
# MULTIPLE GNN ARCHITECTURES
# =============================================================================

class GCNUnsupervised(nn.Module):
    """
    GCN wrapper for benchmark - Uses actual GCNClusterModel.
    
    This is the REAL GCN implementation for unsupervised clustering
    using Graph Convolutional Networks with modularity loss.
    
    Reference: Kipf & Welling (2017) "Semi-Supervised Classification with GCN"
    """
    def __init__(self, in_dim, hidden_dim, max_clusters=50):
        super().__init__()
        # Use actual GCNClusterModel from clusternet
        self.model = GCNClusterModel(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_clusters=max_clusters,
            num_layers=3,
            dropout=0.5
        )
    
    def forward(self, x, edge_index):
        # GCNClusterModel returns cluster logits
        cluster_logits = self.model(x, edge_index)
        return F.softmax(cluster_logits, dim=-1)
    
    def get_communities(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            s = self.forward(x, edge_index)
            assignments = s.argmax(dim=1).cpu().numpy()
            unique = np.unique(assignments)
            mapping = {old: new for new, old in enumerate(unique)}
            return np.array([mapping[a] for a in assignments])


class DMoNGNN(nn.Module):
    """
    DMoN wrapper for benchmark - Uses actual DMoNModel with DMoNPooling.
    
    This is the REAL DMoN implementation using PyG's DMoNPooling layer,
    not just a GCN with custom modularity loss.
    """
    def __init__(self, in_dim, hidden_dim, max_clusters=50):
        super().__init__()
        # Use actual DMoNModel from clusternet
        self.model = DMoNModel(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_clusters=max_clusters,
            dropout=0.3
        )
    
    def forward(self, x, edge_index, return_loss=False):
        # DMoNModel returns (cluster_assignments, total_loss)
        cluster_assignments, total_loss = self.model(x, edge_index)
        if return_loss:
            return cluster_assignments, total_loss
        return cluster_assignments
    
    def get_communities(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            s = self.forward(x, edge_index, return_loss=False)
            assignments = s.argmax(dim=1).cpu().numpy()
            unique = np.unique(assignments)
            mapping = {old: new for new, old in enumerate(unique)}
            return np.array([mapping[a] for a in assignments])


class MinCutGNN(nn.Module):
    """
    MinCut wrapper for benchmark - Uses actual MinCutModel with dense_mincut_pool.
    
    This is the REAL MinCut implementation using PyG's dense_mincut_pool,
    based on Bianchi et al. "Spectral Clustering with Graph Neural Networks".
    """
    def __init__(self, in_dim, hidden_dim, max_clusters=50):
        super().__init__()
        # Use actual MinCutModel from clusternet
        self.model = MinCutModel(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_clusters=max_clusters,
            dropout=0.3
        )
    
    def forward(self, x, edge_index, return_loss=False):
        # MinCutModel returns (s, mincut_loss, ortho_loss)
        s, mincut_loss, ortho_loss = self.model(x, edge_index)
        if return_loss:
            return F.softmax(s, dim=-1), mincut_loss + ortho_loss
        return F.softmax(s, dim=-1)
    
    def get_communities(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            s = self.forward(x, edge_index, return_loss=False)
            assignments = s.argmax(dim=1).cpu().numpy()
            unique = np.unique(assignments)
            mapping = {old: new for new, old in enumerate(unique)}
            return np.array([mapping[a] for a in assignments])


class GATGNN(nn.Module):
    """
    GAT wrapper for benchmark - Uses actual GATClusterModel.
    
    This is the REAL GAT implementation for unsupervised clustering
    using Graph Attention Networks with modularity loss.
    
    Reference: Veličković et al. (2018) "Graph Attention Networks"
    """
    def __init__(self, in_dim, hidden_dim, max_clusters=50, heads=4):
        super().__init__()
        # Use actual GATClusterModel from clusternet
        self.model = GATClusterModel(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_clusters=max_clusters,
            num_layers=3,
            heads=heads,
            dropout=0.3
        )
    
    def forward(self, x, edge_index):
        # GATClusterModel returns cluster logits
        cluster_logits = self.model(x, edge_index)
        return F.softmax(cluster_logits, dim=-1)
    
    def get_communities(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            s = self.forward(x, edge_index)
            assignments = s.argmax(dim=1).cpu().numpy()
            unique = np.unique(assignments)
            mapping = {old: new for new, old in enumerate(unique)}
            return np.array([mapping[a] for a in assignments])


class GraphSAGEGNN(nn.Module):
    """
    GraphSAGE wrapper for benchmark - Uses actual SAGEClusterModel.
    
    This is the REAL GraphSAGE implementation for unsupervised clustering
    using GraphSAGE convolutions with modularity loss.
    
    Reference: Hamilton et al. (2017) "Inductive Representation Learning on Large Graphs"
    """
    def __init__(self, in_dim, hidden_dim, max_clusters=50):
        super().__init__()
        # Use actual SAGEClusterModel from clusternet
        self.model = SAGEClusterModel(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_clusters=max_clusters,
            num_layers=3,
            dropout=0.3,
            aggr='mean'
        )
    
    def forward(self, x, edge_index):
        # SAGEClusterModel returns cluster logits
        cluster_logits = self.model(x, edge_index)
        return F.softmax(cluster_logits, dim=-1)
    
    def get_communities(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            s = self.forward(x, edge_index)
            assignments = s.argmax(dim=1).cpu().numpy()
            unique = np.unique(assignments)
            mapping = {old: new for new, old in enumerate(unique)}
            return np.array([mapping[a] for a in assignments])


class GINGNN(nn.Module):
    """
    GIN wrapper for benchmark - Uses actual GINClusterModel.
    
    This is the REAL GIN (Graph Isomorphism Network) implementation for 
    unsupervised clustering. GIN is the most expressive GNN architecture,
    equivalent to the Weisfeiler-Lehman graph isomorphism test.
    
    Reference: Xu et al. (2019) "How Powerful are Graph Neural Networks?"
    """
    def __init__(self, in_dim, hidden_dim, max_clusters=50):
        super().__init__()
        # Use actual GINClusterModel from clusternet
        self.model = GINClusterModel(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_clusters=max_clusters,
            num_layers=3,
            dropout=0.5
        )
    
    def forward(self, x, edge_index):
        # GINClusterModel returns cluster logits
        cluster_logits = self.model(x, edge_index)
        return F.softmax(cluster_logits, dim=-1)
    
    def get_communities(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            s = self.forward(x, edge_index)
            assignments = s.argmax(dim=1).cpu().numpy()
            unique = np.unique(assignments)
            mapping = {old: new for new, old in enumerate(unique)}
            return np.array([mapping[a] for a in assignments])


class GraphTransformerGNN(nn.Module):
    """
    Graph Transformer wrapper for benchmark - Uses actual GraphTransformerModel.
    
    This is the REAL Graph Transformer implementation for unsupervised clustering
    using self-attention over graph structure with Laplacian positional encodings.
    
    Reference: Dwivedi et al. (2021) "A Generalization of Transformer Networks to Graphs"
    """
    def __init__(self, in_dim, hidden_dim, max_clusters=50):
        super().__init__()
        # Use actual GraphTransformerModel from clusternet
        self.model = GraphTransformerModel(
            in_channels=in_dim,
            hidden_channels=hidden_dim,
            num_clusters=max_clusters,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            pos_enc_dim=8
        )
    
    def forward(self, x, edge_index):
        # GraphTransformerModel returns cluster logits
        cluster_logits = self.model(x, edge_index)
        return F.softmax(cluster_logits, dim=-1)
    
    def get_communities(self, x, edge_index):
        self.eval()
        with torch.no_grad():
            s = self.forward(x, edge_index)
            assignments = s.argmax(dim=1).cpu().numpy()
            unique = np.unique(assignments)
            mapping = {old: new for new, old in enumerate(unique)}
            return np.array([mapping[a] for a in assignments])


# Semi-supervised models (per-network training with labels)
class GCNSemiSupervised(nn.Module):
    """GCN for semi-supervised node classification."""
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return self.classifier(x)


class GATSemiSupervised(nn.Module):
    """GAT for semi-supervised node classification."""
    def __init__(self, in_dim, hidden_dim, num_classes, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        return self.classifier(x)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def mincut_loss(s, edge_index, num_nodes):
    """
    Proper MinCutPool loss from paper.
    
    L = L_cut + L_ortho
    
    L_cut = -Tr(S^T A S) / Tr(S^T D S)
          = -sum of intra-cluster edges / sum of intra-cluster degrees
          (Minimizing this maximizes normalized association)
    
    L_ortho = ||S^T S / ||S^T S||_F - I_K / sqrt(K)||_F
            (Encourages orthogonal, balanced clusters)
    """
    row, col = edge_index[0], edge_index[1]
    n = num_nodes
    k = s.size(1)
    
    # Compute degrees
    deg = torch.zeros(n, device=s.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=s.device))
    
    # === CUT LOSS ===
    # Tr(S^T A S) = sum over edges of s[i]^T * s[j]
    # This counts (weighted) intra-cluster edges
    numerator = (s[row] * s[col]).sum()
    
    # Tr(S^T D S) = sum_i d_i * ||s_i||^2 = sum_i d_i * sum_k s_ik^2
    # For soft assignments with softmax, ||s_i||^2 ≈ max(s_i)
    # Simpler: Tr(S^T D S) = d^T (S * S) summed = sum_k (d^T s_k)^2 / cluster_size
    # Actually: Tr(S^T D S) = sum_i d_i * (s_i^T s_i) 
    s_squared = (s ** 2).sum(dim=1)  # ||s_i||^2 for each node
    denominator = (deg * s_squared).sum()
    
    # Normalized cut loss (minimize = maximize association)
    cut_loss = -numerator / (denominator + 1e-10)
    
    # === ORTHOGONALITY LOSS ===
    # S^T S should be close to (N/K) * I_K for balanced orthogonal clusters
    # L_ortho = ||S^T S / ||S^T S||_F - I_K / sqrt(K)||_F
    sts = torch.mm(s.t(), s)  # K x K
    sts_norm = sts / (torch.norm(sts, p='fro') + 1e-10)
    identity_norm = torch.eye(k, device=s.device) / np.sqrt(k)
    ortho_loss = torch.norm(sts_norm - identity_norm, p='fro')
    
    return cut_loss + ortho_loss


# Keep old class for compatibility
GlobalCommunityGNN = DMoNGNN


def generate_training_networks(n_networks, n_nodes, mu_min, mu_max, verbose=True):
    """Generate training networks with random μ values."""
    import networkx as nx
    
    train_nets = []
    for i in range(n_networks):
        mu = np.random.uniform(mu_min, mu_max)
        seed = i + 1000
        
        try:
            G = nx.LFR_benchmark_graph(
                n=n_nodes, tau1=2.5, tau2=1.5, mu=mu,
                average_degree=15, max_degree=min(50, n_nodes // 5),
                min_community=max(10, n_nodes // 20),
                max_community=min(100, n_nodes // 3),
                seed=seed
            )
            
            gt = np.zeros(n_nodes, dtype=int)
            comm_to_id = {}
            for node in G.nodes():
                comm = frozenset(G.nodes[node]['community'])
                if comm not in comm_to_id:
                    comm_to_id[comm] = len(comm_to_id)
                gt[node] = comm_to_id[comm]
            
            for node in G.nodes():
                del G.nodes[node]['community']
            
            train_nets.append({'G': G, 'gt': gt, 'k': len(comm_to_id), 'mu': mu})
            
        except Exception as e:
            if verbose:
                print(f"  Network {i} failed: {e}")
    
    return train_nets


def train_global_encoder(train_nets, feature_type, epochs=100, verbose=True):
    """Train encoder globally via link prediction on multiple networks."""
    encoder = GlobalGNNEncoder(32, 64, 32).to(DEVICE)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    
    # Precompute features for all networks
    for net in train_nets:
        G = net['G']
        n = G.number_of_nodes()
        if feature_type == 'random':
            net['features'] = random_features(n, 32)
        elif feature_type == 'structural':
            net['features'] = structural_features(G, 32)
        elif feature_type == 'spectral':
            net['features'] = spectral_features(G, 32)
        net['edge_index'] = get_edge_index(G)
    
    encoder.train()
    for epoch in range(epochs):
        np.random.shuffle(train_nets)
        total_loss = 0
        
        for net in train_nets:
            features = net['features'].to(DEVICE)
            edge_index = net['edge_index'].to(DEVICE)
            n = features.size(0)
            
            optimizer.zero_grad()
            z = encoder(features, edge_index)
            
            # Link prediction loss
            num_edges = min(edge_index.size(1), 2000)
            pos = (z[edge_index[0, :num_edges]] * z[edge_index[1, :num_edges]]).sum(1)
            neg_i = torch.randint(0, n, (num_edges,), device=DEVICE)
            neg_j = torch.randint(0, n, (num_edges,), device=DEVICE)
            neg = (z[neg_i] * z[neg_j]).sum(1)
            
            loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + \
                   F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: Loss = {total_loss/len(train_nets):.4f}")
    
    return encoder


def zero_shot_inference(encoder, G, k, feature_type, custom_features=None):
    """Run zero-shot inference on a network using trained encoder."""
    n = G.number_of_nodes()
    
    if custom_features is not None:
        features = custom_features
    elif feature_type == 'random':
        features = random_features(n, 32)
    elif feature_type == 'structural':
        features = structural_features(G, 32)
    elif feature_type == 'spectral':
        features = spectral_features(G, 32)
    
    edge_index = get_edge_index(G)
    
    encoder.eval()
    with torch.no_grad():
        features = features.to(DEVICE)
        edge_index = edge_index.to(DEVICE)
        embeddings = encoder(features, edge_index).cpu().numpy()
    
    # Cluster embeddings with K-Means
    pred = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)
    return pred


# =============================================================================
# GNN BENCHMARKS (ZERO-SHOT)
# =============================================================================

def train_global_encoder_homophily(train_nets, feature_distances, epochs=100, verbose=True):
    """Train encoder globally with homophily features (varied feature distances)."""
    encoder = GlobalGNNEncoder(32, 64, 32).to(DEVICE)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    
    # Precompute features for all networks (sample feature distance randomly)
    for net in train_nets:
        G = net['G']
        gt = net['gt']
        k = net['k']
        n = G.number_of_nodes()
        
        # Random feature distance for this network
        feat_dist = np.random.choice(feature_distances)
        net['features'] = homophily_features(n, gt, k, feature_distance=feat_dist, seed=np.random.randint(10000))
        net['edge_index'] = get_edge_index(G)
    
    encoder.train()
    for epoch in range(epochs):
        np.random.shuffle(train_nets)
        total_loss = 0
        
        for net in train_nets:
            features = net['features'].to(DEVICE)
            edge_index = net['edge_index'].to(DEVICE)
            n = features.size(0)
            
            optimizer.zero_grad()
            z = encoder(features, edge_index)
            
            # Link prediction loss
            num_edges = min(edge_index.size(1), 2000)
            pos = (z[edge_index[0, :num_edges]] * z[edge_index[1, :num_edges]]).sum(1)
            neg_i = torch.randint(0, n, (num_edges,), device=DEVICE)
            neg_j = torch.randint(0, n, (num_edges,), device=DEVICE)
            neg = (z[neg_i] * z[neg_j]).sum(1)
            
            loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + \
                   F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: Loss = {total_loss/len(train_nets):.4f}")
    
    return encoder


def generate_lfr_training_networks_with_homophily(n_networks, n_nodes, mu_min, mu_max, feature_distances, verbose=True):
    """Generate LFR training networks with homophily features."""
    import networkx as nx
    
    train_nets = []
    for i in range(n_networks):
        mu = np.random.uniform(mu_min, mu_max)
        feat_dist = np.random.choice(feature_distances)
        seed = i + 2000
        
        try:
            # Generate LFR network
            G = nx.LFR_benchmark_graph(
                n=n_nodes, tau1=2.5, tau2=1.5, mu=mu,
                average_degree=15, max_degree=min(50, n_nodes // 5),
                min_community=max(10, n_nodes // 20),
                max_community=min(100, n_nodes // 3),
                seed=seed
            )
            
            # Extract ground truth
            gt = np.zeros(n_nodes, dtype=int)
            comm_to_id = {}
            for node in G.nodes():
                comm = frozenset(G.nodes[node]['community'])
                if comm not in comm_to_id:
                    comm_to_id[comm] = len(comm_to_id)
                gt[node] = comm_to_id[comm]
            
            k = len(comm_to_id)
            
            # Remove community attribute
            for node in G.nodes():
                del G.nodes[node]['community']
            
            # Create homophily features based on ground truth
            features = homophily_features(n_nodes, gt, k, feature_distance=feat_dist, seed=seed)
            
            train_nets.append({
                'G': G, 'gt': gt, 'k': k, 'mu': mu,
                'features': features
            })
            
        except Exception as e:
            if verbose:
                print(f"  LFR network {i} failed: {e}")
    
    return train_nets


def train_unsupervised_gnn(model, train_nets, loss_type='modularity', epochs=100, verbose=True):
    """Train a community detection GNN with unsupervised loss (modularity or mincut)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    model.train()
    best_loss = float('inf')
    patience = 0
    
    # Check if model has built-in loss (DMoNGNN and MinCutGNN now both support return_loss)
    has_builtin_loss = isinstance(model, (DMoNGNN, MinCutGNN))
    
    for epoch in range(epochs):
        np.random.shuffle(train_nets)
        total_loss = 0
        total_ami = 0
        
        for net in train_nets:
            features = net['features'].to(DEVICE)
            edge_index = net['edge_index'].to(DEVICE)
            n = features.size(0)
            gt = net['gt']
            
            optimizer.zero_grad()
            
            if has_builtin_loss:
                # DMoNGNN and MinCutGNN return (soft_assignments, loss) when return_loss=True
                s, loss = model(features, edge_index, return_loss=True)
            else:
                s = model(features, edge_index)
                if loss_type == 'modularity':
                    loss = modularity_loss(s, edge_index, n)
                else:  # mincut (fallback for non-MinCutGNN models)
                    loss = mincut_loss(s, edge_index, n)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            with torch.no_grad():
                pred = s.argmax(dim=1).cpu().numpy()
                total_ami += adjusted_mutual_info_score(gt, pred)
        
        avg_loss = total_loss / len(train_nets)
        avg_ami = total_ami / len(train_nets)
        
        if verbose and (epoch + 1) % 30 == 0:
            print(f"      Epoch {epoch+1}: Loss = {avg_loss:.4f}, Train AMI = {avg_ami:.3f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience = 0
        else:
            patience += 1
            if patience >= 30:
                if verbose:
                    print(f"      Early stopping at epoch {epoch+1}")
                break
    
    return model


def train_semisupervised_gnn(model, G, gt, features, train_ratio=0.5, epochs=200):
    """Train semi-supervised GNN per-network with partial labels."""
    n = G.number_of_nodes()
    edge_index = get_edge_index(G).to(DEVICE)
    features = features.to(DEVICE)
    labels = torch.tensor(gt, dtype=torch.long, device=DEVICE)
    
    # Create train/test split
    n_train = int(n * train_ratio)
    perm = torch.randperm(n)
    train_mask = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    train_mask[perm[:n_train]] = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(features, edge_index)
        loss = F.cross_entropy(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
    
    # Predict
    model.eval()
    with torch.no_grad():
        out = model(features, edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
    
    return pred


def run_gnn_homophily_benchmark(
    networks_dir: str,
    feature_distances: list,
    results_dir: str,
    n_train_networks: int = 100,
    verbose: bool = True
):
    """
    Run GNN homophily ablation benchmark with 4 GNN architectures.
    
    UNSUPERVISED (global training, zero-shot testing):
    - DMoN (GCN + modularity loss)
    - MinCut (GCN + mincut loss)
    - GAT (GAT + modularity loss)
    - SAGE (GraphSAGE + modularity loss)
    
    SEMI-SUPERVISED (per-network training with labels):
    - GCN semi-supervised
    - GAT semi-supervised
    """
    if not HAS_TORCH:
        print("PyTorch not available, skipping GNN benchmark")
        return None
    
    networks_path = Path(networks_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(networks_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    all_results = []
    mu_values = sorted(list(set([net['mu'] for net in metadata['networks']])))
    n_nodes = metadata['networks'][0]['nodes']
    
    print("\n" + "="*60)
    print("GNN HOMOPHILY BENCHMARK - 7 ARCHITECTURES")
    print(f"Training on {n_train_networks} LFR networks")
    print(f"Testing on SAME networks as classical algorithms")
    print(f"Feature distances: {feature_distances}")
    print("="*60)
    
    # Generate training networks
    print(f"\nGenerating {n_train_networks} LFR training networks...")
    mu_min, mu_max = min(mu_values), max(mu_values)
    train_nets = generate_lfr_training_networks_with_homophily(
        n_train_networks, n_nodes, mu_min, mu_max, feature_distances, verbose=False
    )
    print(f"  Generated {len(train_nets)} training networks")
    
    for net in train_nets:
        net['edge_index'] = get_edge_index(net['G'])
    
    # =========================================================================
    # UNSUPERVISED GNNs (Global Training)
    # =========================================================================
    unsupervised_models = {
        #'GCN_unsup': (GCNUnsupervised(32, 64, max_clusters=50), 'modularity'),  # GCN with modularity loss
        'DMoN': (DMoNGNN(32, 64, max_clusters=50), 'modularity'),  # DMoN with DMoNPooling
        #'MinCut': (MinCutGNN(32, 64, max_clusters=50), 'mincut'),  # MinCut with dense_mincut_pool
        #'GAT_unsup': (GATGNN(32, 64, max_clusters=50), 'modularity'),  # GAT with modularity loss
        #'SAGE_unsup': (GraphSAGEGNN(32, 64, max_clusters=50), 'modularity'),  # GraphSAGE with modularity loss
        'GIN_unsup': (GINGNN(32, 64, max_clusters=50), 'modularity'),  # GIN with modularity loss
          # Graph Transformer
    }
    
    trained_models = {}
    for name, (model, loss_type) in unsupervised_models.items():
        print(f"\n[UNSUPERVISED] Training {name}...")
        model = model.to(DEVICE)
        trained_models[name] = train_unsupervised_gnn(
            model, train_nets, loss_type=loss_type, epochs=100, verbose=verbose
        )
    
    # =========================================================================
    # TEST ON SAME NETWORKS AS CLASSICAL
    # =========================================================================
    print("\n" + "-"*60)
    print("TESTING ON SAME NETWORKS AS CLASSICAL")
    print("-"*60)
    
    for network_info in metadata['networks']:
        name = network_info['name']
        mu = network_info['mu']
        
        if verbose:
            print(f"\nNetwork: {name} (μ={mu})")
        
        G, true_communities = load_network(str(networks_path), name)
        n = G.number_of_nodes()
        k = len(true_communities)
        
        gt = np.zeros(n, dtype=int)
        for i, comm in enumerate(true_communities):
            for node in comm:
                gt[node] = i
        
        edge_index = get_edge_index(G).to(DEVICE)
        
        for feat_dist in feature_distances:
            features = homophily_features(n, gt, k, feature_distance=feat_dist, seed=network_info['realization'])
            features_gpu = features.to(DEVICE)
            
            # -----------------------------------------------------------------
            # UNSUPERVISED GNNs (zero-shot inference)
            # -----------------------------------------------------------------
            for model_name, model in trained_models.items():
                start_time = time.time()
                try:
                    model.eval()
                    pred = model.get_communities(features_gpu, edge_index)
                    runtime = time.time() - start_time
                    ami = adjusted_mutual_info_score(gt, pred)
                    nmi = normalized_mutual_info_score(gt, pred)
                    
                    all_results.append({
                        'network': name,
                        'mu': mu,
                        'realization': network_info['realization'],
                        'feature_distance': feat_dist,
                        'algorithm': model_name,
                        'algorithm_type': 'gnn_unsupervised',
                        'training': 'global_zeroshot',
                        'success': True,
                        'runtime': runtime,
                        'ami': ami,
                        'nmi': nmi,
                        'num_detected': len(set(pred)),
                        'num_true': k
                    })
                    
                    if verbose:
                        print(f"  {model_name} (fd={feat_dist}): AMI={ami:.3f}")
                        
                except Exception as e:
                    all_results.append({
                        'network': name,
                        'mu': mu,
                        'realization': network_info['realization'],
                        'feature_distance': feat_dist,
                        'algorithm': model_name,
                        'algorithm_type': 'gnn_unsupervised',
                        'training': 'global_zeroshot',
                        'success': False,
                        'error': str(e)
                    })
            
            # -----------------------------------------------------------------
            # SEMI-SUPERVISED GNNs (per-network training)
            # -----------------------------------------------------------------
            for ss_name, ss_class in [('GCN_semisup', GCNSemiSupervised), 
                                       ('GAT_semisup', GATSemiSupervised)]:
                start_time = time.time()
                try:
                    model = ss_class(32, 64, k).to(DEVICE)
                    pred = train_semisupervised_gnn(model, G, gt, features, train_ratio=0.2, epochs=200)
                    runtime = time.time() - start_time
                    ami = adjusted_mutual_info_score(gt, pred)
                    nmi = normalized_mutual_info_score(gt, pred)
                    
                    all_results.append({
                        'network': name,
                        'mu': mu,
                        'realization': network_info['realization'],
                        'feature_distance': feat_dist,
                        'algorithm': ss_name,
                        'algorithm_type': 'gnn_semisupervised',
                        'training': 'per_network',
                        'success': True,
                        'runtime': runtime,
                        'ami': ami,
                        'nmi': nmi,
                        'num_detected': len(set(pred)),
                        'num_true': k
                    })
                    
                    if verbose:
                        print(f"  {ss_name} (fd={feat_dist}): AMI={ami:.3f}")
                        
                except Exception as e:
                    all_results.append({
                        'network': name,
                        'mu': mu,
                        'realization': network_info['realization'],
                        'feature_distance': feat_dist,
                        'algorithm': ss_name,
                        'algorithm_type': 'gnn_semisupervised',
                        'training': 'per_network',
                        'success': False,
                        'error': str(e)
                    })
            
            # -----------------------------------------------------------------
            # K-Means baseline
            # -----------------------------------------------------------------
            start_time = time.time()
            pred_km = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features.numpy())
            ami_km = adjusted_mutual_info_score(gt, pred_km)
            nmi_km = normalized_mutual_info_score(gt, pred_km)
            
            all_results.append({
                'network': name,
                'mu': mu,
                'realization': network_info['realization'],
                'feature_distance': feat_dist,
                'algorithm': 'kmeans_homophily',
                'algorithm_type': 'baseline',
                'training': 'none',
                'success': True,
                'runtime': time.time() - start_time,
                'ami': ami_km,
                'nmi': nmi_km,
                'num_detected': len(set(pred_km)),
                'num_true': k
            })
            
            if verbose:
                print(f"  KMeans (fd={feat_dist}): AMI={ami_km:.3f}")
    
    df = pd.DataFrame(all_results)
    df.to_csv(results_path / 'gnn_homophily_results.csv', index=False)
    
    df_success = df[df['success'] == True]
    summary = df_success.groupby(['mu', 'feature_distance', 'algorithm']).agg({
        'ami': ['mean', 'std'],
        'nmi': ['mean', 'std'],
        'runtime': ['mean', 'std']
    }).round(4)
    summary.to_csv(results_path / 'gnn_homophily_summary.csv')
    
    if verbose:
        print(f"\nHomophily results saved to: {results_path / 'gnn_homophily_results.csv'}")
    
    return df


def run_gnn_features_benchmark(
    networks_dir: str,
    results_dir: str,
    n_train_networks: int = 100,
    verbose: bool = True
):
    """
    Run GNN node features benchmark with 4 GNN architectures.
    
    Feature types: random, structural, spectral
    GNNs: DMoN, MinCut, GAT, SAGE (all unsupervised with modularity/mincut loss)
    
    1. Train globally on LFR networks (separate model per feature type per GNN)
    2. Test on SAME LFR networks as classical algorithms (zero-shot)
    """
    if not HAS_TORCH:
        print("PyTorch not available, skipping GNN benchmark")
        return None
    
    networks_path = Path(networks_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(networks_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    all_results = []
    feature_types = ['random', 'structural', 'spectral']
    
    # GNN architectures to test (UNSUPERVISED - global training)
    gnn_configs = {
        'GCN_unsup': (GCNUnsupervised, 'modularity'),  # GCN with modularity loss
        'DMoN': (DMoNGNN, 'modularity'),  # DMoN with DMoNPooling
        'MinCut': (MinCutGNN, 'mincut'),  # MinCut with dense_mincut_pool
        'GAT_unsup': (GATGNN, 'modularity'),  # GAT with modularity loss
        'SAGE_unsup': (GraphSAGEGNN, 'modularity'),  # GraphSAGE with modularity loss
        'GIN_unsup': (GINGNN, 'modularity'),  # GIN with modularity loss
      # Graph Transformer
    }
    
    print("\n" + "="*60)
    print("GNN NODE FEATURES BENCHMARK - 7 ARCHITECTURES")
    print(f"Training on {n_train_networks} networks")
    print(f"Testing on SAME networks as classical algorithms")
    print(f"Feature types: {feature_types}")
    print(f"GNN architectures: {list(gnn_configs.keys())}")
    print("="*60)
    
    # Get mu range from test networks
    mu_values = list(set([net['mu'] for net in metadata['networks']]))
    mu_min, mu_max = min(mu_values), max(mu_values)
    n_nodes = metadata['networks'][0]['nodes']
    
    # Generate training networks
    print(f"\nGenerating {n_train_networks} training networks...")
    train_nets = generate_training_networks(n_train_networks, n_nodes, mu_min, mu_max, verbose=False)
    print(f"  Generated {len(train_nets)} training networks")
    
    # Precompute edge indices
    for net in train_nets:
        net['edge_index'] = get_edge_index(net['G'])
    
    # Train models for each (feature_type, gnn_architecture) combination
    trained_models = {}
    
    for feat_type in feature_types:
        print(f"\n{'='*40}")
        print(f"FEATURE TYPE: {feat_type.upper()}")
        print(f"{'='*40}")
        
        # Compute features for training networks
        for net in train_nets:
            G = net['G']
            n = G.number_of_nodes()
            if feat_type == 'random':
                net['features'] = random_features(n, 32)
            elif feat_type == 'structural':
                net['features'] = structural_features(G, 32)
            elif feat_type == 'spectral':
                net['features'] = spectral_features(G, 32)
        
        # Train each GNN architecture
        for gnn_name, (gnn_class, loss_type) in gnn_configs.items():
            print(f"\n  Training {gnn_name} ({loss_type} loss)...")
            model = gnn_class(32, 64, max_clusters=50).to(DEVICE)
            trained_model = train_unsupervised_gnn(
                model, train_nets, loss_type=loss_type, epochs=80, verbose=False
            )
            trained_models[(feat_type, gnn_name)] = trained_model
            print(f"    ✓ {gnn_name} trained")
    
    # =========================================================================
    # TEST ON SAME NETWORKS AS CLASSICAL
    # =========================================================================
    print("\n" + "="*60)
    print("TESTING ON SAME NETWORKS AS CLASSICAL")
    print("="*60)
    
    for network_info in metadata['networks']:
        name = network_info['name']
        mu = network_info['mu']
        
        if verbose:
            print(f"\nNetwork: {name} (μ={mu})")
        
        G, true_communities = load_network(str(networks_path), name)
        n = G.number_of_nodes()
        k = len(true_communities)
        
        gt = np.zeros(n, dtype=int)
        for i, comm in enumerate(true_communities):
            for node in comm:
                gt[node] = i
        
        edge_index = get_edge_index(G).to(DEVICE)
        
        for feat_type in feature_types:
            # Generate features for test network
            if feat_type == 'random':
                features = random_features(n, 32)
            elif feat_type == 'structural':
                features = structural_features(G, 32)
            elif feat_type == 'spectral':
                features = spectral_features(G, 32)
            
            features_gpu = features.to(DEVICE)
            
            # Test each GNN architecture
            for gnn_name in gnn_configs.keys():
                model = trained_models[(feat_type, gnn_name)]
                
                start_time = time.time()
                try:
                    model.eval()
                    pred = model.get_communities(features_gpu, edge_index)
                    runtime = time.time() - start_time
                    ami = adjusted_mutual_info_score(gt, pred)
                    nmi = normalized_mutual_info_score(gt, pred)
                    
                    all_results.append({
                        'network': name,
                        'mu': mu,
                        'realization': network_info['realization'],
                        'feature_type': feat_type,
                        'algorithm': gnn_name,
                        'algorithm_type': 'gnn_unsupervised',
                        'training': 'global_zeroshot',
                        'success': True,
                        'runtime': runtime,
                        'ami': ami,
                        'nmi': nmi,
                        'num_detected': len(set(pred)),
                        'num_true': k
                    })
                    
                    if verbose:
                        print(f"  {gnn_name} ({feat_type}): AMI={ami:.3f}")
                        
                except Exception as e:
                    all_results.append({
                        'network': name,
                        'mu': mu,
                        'realization': network_info['realization'],
                        'feature_type': feat_type,
                        'algorithm': gnn_name,
                        'algorithm_type': 'gnn_unsupervised',
                        'training': 'global_zeroshot',
                        'success': False,
                        'error': str(e)
                    })
                    if verbose:
                        print(f"  {gnn_name} ({feat_type}): FAILED - {e}")
            
            # SEMI-SUPERVISED GNNs (per-network training with 50% labels)
            for ss_name, ss_class in [('GCN_semisup', GCNSemiSupervised), 
                                       ('GAT_semisup', GATSemiSupervised)]:
                start_time = time.time()
                try:
                    model = ss_class(32, 64, k).to(DEVICE)
                    pred = train_semisupervised_gnn(model, G, gt, features, train_ratio=0.2, epochs=200)
                    runtime = time.time() - start_time
                    ami = adjusted_mutual_info_score(gt, pred)
                    nmi = normalized_mutual_info_score(gt, pred)
                    
                    all_results.append({
                        'network': name,
                        'mu': mu,
                        'realization': network_info['realization'],
                        'feature_type': feat_type,
                        'algorithm': ss_name,
                        'algorithm_type': 'gnn_semisupervised',
                        'training': 'per_network',
                        'success': True,
                        'runtime': runtime,
                        'ami': ami,
                        'nmi': nmi,
                        'num_detected': len(set(pred)),
                        'num_true': k
                    })
                    
                    if verbose:
                        print(f"  {ss_name} ({feat_type}): AMI={ami:.3f}")
                        
                except Exception as e:
                    all_results.append({
                        'network': name,
                        'mu': mu,
                        'realization': network_info['realization'],
                        'feature_type': feat_type,
                        'algorithm': ss_name,
                        'algorithm_type': 'gnn_semisupervised',
                        'training': 'per_network',
                        'success': False,
                        'error': str(e)
                    })
                    if verbose:
                        print(f"  {ss_name} ({feat_type}): FAILED - {e}")
            
            # K-Means baseline on raw features
            start_time = time.time()
            pred_km = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features.numpy())
            ami_km = adjusted_mutual_info_score(gt, pred_km)
            nmi_km = normalized_mutual_info_score(gt, pred_km)
            
            all_results.append({
                'network': name,
                'mu': mu,
                'realization': network_info['realization'],
                'feature_type': feat_type,
                'algorithm': 'KMeans',
                'algorithm_type': 'baseline',
                'training': 'none',
                'success': True,
                'runtime': time.time() - start_time,
                'ami': ami_km,
                'nmi': nmi_km,
                'num_detected': len(set(pred_km)),
                'num_true': k
            })
            
            if verbose:
                print(f"  KMeans ({feat_type}): AMI={ami_km:.3f}")
    
    df = pd.DataFrame(all_results)
    df.to_csv(results_path / 'gnn_features_results.csv', index=False)
    
    df_success = df[df['success'] == True]
    summary = df_success.groupby(['mu', 'feature_type', 'algorithm']).agg({
        'ami': ['mean', 'std'],
        'nmi': ['mean', 'std'],
        'runtime': ['mean', 'std']
    }).round(4)
    summary.to_csv(results_path / 'gnn_features_summary.csv')
    
    if verbose:
        print(f"\nFeatures results saved to: {results_path / 'gnn_features_results.csv'}")
    
    return df


# =============================================================================
# PLOTTING
# =============================================================================

def plot_accuracy_results(results_dir: str):
    """Generate plots from accuracy results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not available for plotting")
        return
    
    results_path = Path(results_dir)
    
    # Plot classical results
    classical_file = results_path / 'classical_accuracy_results.csv'
    if classical_file.exists():
        df = pd.read_csv(classical_file)
        df_success = df[df['success'] == True]
        
        plt.figure(figsize=(12, 8))
        for algo in df_success['algorithm'].unique():
            algo_data = df_success[df_success['algorithm'] == algo]
            summary = algo_data.groupby('mu')['ami'].agg(['mean', 'std']).reset_index()
            plt.errorbar(summary['mu'], summary['mean'], yerr=summary['std'],
                        marker='o', label=algo, capsize=3)
        
        plt.xlabel('Mixing Parameter (μ)', fontsize=12)
        plt.ylabel('Adjusted Mutual Information (AMI)', fontsize=12)
        plt.title('Classical Algorithms: AMI vs Mixing Parameter', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_path / 'classical_ami_vs_mu.png', dpi=150)
        plt.close()
    
    # Plot GNN features results
    features_file = results_path / 'gnn_features_results.csv'
    if features_file.exists():
        df = pd.read_csv(features_file)
        df_success = df[df['success'] == True]
        
        for feat_type in df_success['feature_type'].unique():
            plt.figure(figsize=(10, 6))
            feat_data = df_success[df_success['feature_type'] == feat_type]
            
            for algo in feat_data['algorithm'].unique():
                algo_data = feat_data[feat_data['algorithm'] == algo]
                summary = algo_data.groupby('mu')['ami'].agg(['mean', 'std']).reset_index()
                plt.errorbar(summary['mu'], summary['mean'], yerr=summary['std'],
                            marker='o', label=algo, capsize=3)
            
            plt.xlabel('Mixing Parameter (μ)', fontsize=12)
            plt.ylabel('AMI', fontsize=12)
            plt.title(f'GNN with {feat_type.capitalize()} Features', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(results_path / f'gnn_{feat_type}_ami_vs_mu.png', dpi=150)
            plt.close()
    
    # Plot homophily results
    homophily_file = results_path / 'gnn_homophily_results.csv'
    if homophily_file.exists():
        df = pd.read_csv(homophily_file)
        df_success = df[df['success'] == True]
        
        # Heatmap for each algorithm
        for algo in df_success['algorithm'].unique():
            algo_data = df_success[df_success['algorithm'] == algo]
            pivot = algo_data.pivot_table(
                values='ami', index='feature_distance', columns='mu', aggfunc='mean'
            )
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
            plt.title(f'{algo.upper()}: AMI by Feature Distance and μ')
            plt.xlabel('Mixing Parameter (μ)')
            plt.ylabel('Feature Distance')
            plt.tight_layout()
            plt.savefig(results_path / f'gnn_{algo}_homophily_heatmap.png', dpi=150)
            plt.close()
    
    print(f"Plots saved to {results_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='LFR Accuracy Benchmark')
    parser.add_argument('--generate', action='store_true', help='Generate networks')
    parser.add_argument('--run-classical', action='store_true', help='Run classical algorithms')
    parser.add_argument('--run-gnn-homophily', action='store_true', help='Run GNN homophily ablation (zero-shot)')
    parser.add_argument('--run-gnn-features', action='store_true', help='Run GNN node features benchmark (zero-shot)')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--all', action='store_true', help='Run everything')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--mu-values', type=str, default=None,
                        help='Comma-separated μ values (e.g., "0.1,0.3,0.5")')
    parser.add_argument('--realizations', type=int, default=None,
                        help='Number of realizations per μ')
    parser.add_argument('--feature-distances', type=str, default='0.2,0.4,0.6,0.8,1.2',
                        help='Comma-separated feature distances for homophily test')
    parser.add_argument('--train-networks', type=int, default=100,
                        help='Number of training networks for global GNN training')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    lfr_dir = script_dir.parent.parent.parent.parent / 'LFRbenchmarks'
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / 'accuracy'
    
    networks_dir = output_dir / 'networks'
    results_dir = output_dir / 'results'
    
    # Update config
    config = ACCURACY_CONFIG.copy()
    if args.mu_values:
        config['mu_values'] = [float(x) for x in args.mu_values.split(',')]
    if args.realizations:
        config['realizations'] = args.realizations
    
    feature_distances = [float(x) for x in args.feature_distances.split(',')]
    
    # Get algorithms
    algorithms = []
    for category in ACTIVE_CATEGORIES:
        algorithms.extend(ALGORITHMS.get(category, []))
    
    # Generate networks
    if args.all or args.generate:
        print("="*60)
        print("GENERATING LFR NETWORKS")
        print("="*60)
        print(f"μ values: {config['mu_values']}")
        print(f"Realizations: {config['realizations']}")
        
        generate_accuracy_networks(
            str(networks_dir), config, str(lfr_dir), verbose=True
        )
    
    # Run classical benchmark
    if args.all or args.run_classical:
        print("\n" + "="*60)
        print("RUNNING CLASSICAL ALGORITHMS BENCHMARK")
        print("="*60)
        
        run_classical_benchmark(
            str(networks_dir), str(results_dir), algorithms, verbose=True
        )
    
    # Run GNN homophily benchmark (zero-shot on same networks as classical)
    if args.all or args.run_gnn_homophily:
        print("\n" + "="*60)
        print("RUNNING GNN HOMOPHILY BENCHMARK (ZERO-SHOT)")
        print("="*60)
        print(f"Training on {args.train_networks} networks")
        print(f"Testing on SAME networks as classical algorithms")
        
        run_gnn_homophily_benchmark(
            networks_dir=str(networks_dir),
            feature_distances=feature_distances,
            results_dir=str(results_dir),
            n_train_networks=args.train_networks,
            verbose=True
        )
    
    # Run GNN features benchmark (zero-shot on same networks as classical)
    if args.all or args.run_gnn_features:
        print("\n" + "="*60)
        print("RUNNING GNN NODE FEATURES BENCHMARK (ZERO-SHOT)")
        print("="*60)
        print(f"Training on {args.train_networks} networks, testing on same LFR networks as classical")
        
        run_gnn_features_benchmark(
            str(networks_dir), str(results_dir), 
            n_train_networks=args.train_networks,
            verbose=True
        )
    
    # Generate plots
    if args.all or args.plot:
        print("\n" + "="*60)
        print("GENERATING PLOTS")
        print("="*60)
        
        plot_accuracy_results(str(results_dir))
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
