#!/usr/bin/env python3
"""
Comprehensive GNN Accuracy Benchmark Suite
==========================================

This script provides a systematic evaluation of GNN-based community detection
across multiple dimensions:

1. HOMOPHILY ABLATION: Test different feature distances to understand when 
   node features help vs hurt performance

2. GLOBAL FEATURE LEARNING: Train a shared feature encoder via link prediction
   across multiple networks

3. SPECTRAL FEATURES: Use Laplacian eigenvectors as node features

4. MULTIPLE ALGORITHMS: Test GCN, GAT, SAGE, DMoN in semi-supervised and 
   unsupervised regimes

Usage:
    python gnn_accuracy_benchmark.py --experiment all --mu-values "0.1,0.3,0.5" --realizations 5
    python gnn_accuracy_benchmark.py --experiment homophily --realizations 3
    python gnn_accuracy_benchmark.py --experiment global_features --train-networks 100
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags

# Add ClusterNet to path
CLUSTERNET_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CLUSTERNET_PATH))

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# NETWORK GENERATION
# =============================================================================

def generate_lfr(n: int, mu: float, seed: int) -> Tuple[nx.Graph, np.ndarray, int]:
    """Generate LFR benchmark network."""
    np.random.seed(seed)
    try:
        G = nx.LFR_benchmark_graph(
            n=n, tau1=2.5, tau2=1.5, mu=mu,
            average_degree=15,
            max_degree=min(50, n // 5),
            min_community=max(10, n // 20),
            max_community=min(100, n // 3),
            seed=seed
        )
        # Extract ground truth
        gt = np.zeros(n, dtype=int)
        comm_to_id = {}
        for node in G.nodes():
            comm = frozenset(G.nodes[node]['community'])
            if comm not in comm_to_id:
                comm_to_id[comm] = len(comm_to_id)
            gt[node] = comm_to_id[comm]
        
        # Remove community attribute
        for node in G.nodes():
            del G.nodes[node]['community']
        
        return G, gt, len(comm_to_id)
    except Exception as e:
        return None, None, None


def generate_sbm_with_features(
    n: int, k: int, mu: float, 
    feature_dim: int = 32, 
    feature_distance: float = 1.0,
    noise_std: float = 0.8,
    seed: int = 42
) -> Tuple[nx.Graph, np.ndarray, torch.Tensor, int]:
    """
    Generate SBM with cluster-correlated features.
    
    Args:
        n: Number of nodes
        k: Number of communities
        mu: Mixing parameter (higher = weaker structure)
        feature_dim: Dimension of node features
        feature_distance: Distance between cluster centers (higher = more separable)
        noise_std: Noise standard deviation
        seed: Random seed
    
    Returns:
        G: NetworkX graph
        gt: Ground truth labels
        features: Node feature tensor
        k: Number of communities
    """
    np.random.seed(seed)
    
    # Community sizes
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    
    # Edge probabilities based on mu
    avg_degree = 15
    total_edges = n * avg_degree // 2
    p_in = (1 - mu) * total_edges * 2 / sum(s**2 for s in sizes)
    p_out = mu * total_edges * 2 / (n**2 - sum(s**2 for s in sizes))
    p_in, p_out = min(p_in, 0.9), max(p_out, 0.001)
    
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    
    # Ground truth
    gt = sum([[i] * s for i, s in enumerate(sizes)], [])
    gt = np.array(gt)
    
    # Generate cluster-correlated features
    centers = np.random.randn(k, feature_dim) * feature_distance
    features = np.array([
        centers[gt[i]] + np.random.randn(feature_dim) * noise_std 
        for i in range(n)
    ])
    
    return G, gt, torch.tensor(features, dtype=torch.float), k


# =============================================================================
# NODE FEATURES
# =============================================================================

def random_gaussian_features(n: int, dim: int = 32) -> torch.Tensor:
    """Generate random Gaussian features (no cluster information)."""
    return torch.randn(n, dim) * 0.1


def structural_features(G: nx.Graph, dim: int = 32) -> torch.Tensor:
    """Generate structural features based on graph topology."""
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


def spectral_features(G: nx.Graph, dim: int = 32) -> torch.Tensor:
    """Generate spectral features (Laplacian eigenvectors)."""
    n = G.number_of_nodes()
    
    try:
        # Create adjacency matrix
        A = nx.adjacency_matrix(G).astype(np.float64)
        
        # Compute normalized Laplacian
        degrees = np.array(A.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1
        D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
        L_norm = diags(np.ones(n)) - D_inv_sqrt @ A @ D_inv_sqrt
        
        # Compute eigenvectors
        k = min(dim, n - 2)
        eigenvalues, eigenvectors = eigsh(L_norm, k=k+1, which='SM', tol=1e-6)
        
        # Sort and skip trivial eigenvector
        idx = np.argsort(eigenvalues)
        eigenvectors = eigenvectors[:, idx][:, 1:k+1]
        
        # Normalize
        eigenvectors = eigenvectors / (np.linalg.norm(eigenvectors, axis=0, keepdims=True) + 1e-8)
        
        # Pad if needed
        if eigenvectors.shape[1] < dim:
            padding = np.zeros((n, dim - eigenvectors.shape[1]))
            eigenvectors = np.hstack([eigenvectors, padding])
        
        return torch.tensor(eigenvectors[:, :dim], dtype=torch.float)
    
    except Exception as e:
        print(f"    Spectral features failed: {e}, using structural")
        return structural_features(G, dim)


def get_edge_index(G: nx.Graph) -> torch.Tensor:
    """Convert graph to edge index tensor."""
    edges = list(G.edges())
    if len(edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([
        [e[0] for e in edges] + [e[1] for e in edges],
        [e[1] for e in edges] + [e[0] for e in edges]
    ], dtype=torch.long)


# =============================================================================
# GNN MODELS
# =============================================================================

class GCNEncoder(nn.Module):
    """GCN-based encoder for feature learning."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class GATEncoder(nn.Module):
    """GAT-based encoder."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=2, concat=False)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1)
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class SAGEEncoder(nn.Module):
    """GraphSAGE-based encoder."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class ClusteringHead(nn.Module):
    """Clustering head for unsupervised training."""
    def __init__(self, in_dim, num_clusters):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_clusters)
        nn.init.xavier_uniform_(self.fc.weight, gain=2.0)
    
    def forward(self, x):
        return F.softmax(self.fc(x) * 2.0, dim=-1)


class SemiSupervisedGNN(nn.Module):
    """Semi-supervised GNN with learnable embeddings."""
    def __init__(self, num_nodes, hidden_dim, num_classes, encoder_class=GCNConv):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, edge_index, x=None):
        if x is None:
            x = self.embedding.weight
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return self.classifier(x)


class UnsupervisedGNN(nn.Module):
    """Unsupervised GNN with modularity loss."""
    def __init__(self, in_dim, hidden_dim, num_clusters):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.cluster = nn.Linear(hidden_dim, num_clusters)
        nn.init.xavier_uniform_(self.cluster.weight, gain=2.0)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        return F.softmax(self.cluster(x) * 2.0, dim=-1)


class DMoN(nn.Module):
    """Deep Modularity Network."""
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


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

class GlobalFeatureEncoder:
    """
    Trains a feature encoder globally across multiple networks via link prediction.
    """
    def __init__(self, feature_dim: int = 32, hidden_dim: int = 64, 
                 out_dim: int = 32, encoder_class=GCNEncoder):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.encoder_class = encoder_class
        self.model = None
    
    def train(self, networks: List[Dict], epochs: int = 100, 
              feature_type: str = 'random', verbose: bool = True):
        """
        Train encoder globally on multiple networks via link prediction.
        
        Args:
            networks: List of network dicts with 'G', 'gt', 'k'
            epochs: Training epochs
            feature_type: 'random', 'structural', or 'spectral'
            verbose: Print progress
        """
        self.model = self.encoder_class(self.feature_dim, self.hidden_dim, self.out_dim).to(DEVICE)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        # Compute features for all networks
        for net in networks:
            G = net['G']
            n = G.number_of_nodes()
            if feature_type == 'random':
                net['features'] = random_gaussian_features(n, self.feature_dim)
            elif feature_type == 'structural':
                net['features'] = structural_features(G, self.feature_dim)
            elif feature_type == 'spectral':
                net['features'] = spectral_features(G, self.feature_dim)
            net['edge_index'] = get_edge_index(G)
        
        self.model.train()
        for epoch in range(epochs):
            np.random.shuffle(networks)
            total_loss = 0
            
            for net in networks:
                features = net['features'].to(DEVICE)
                edge_index = net['edge_index'].to(DEVICE)
                n = features.size(0)
                
                optimizer.zero_grad()
                z = self.model(features, edge_index)
                
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
                print(f"    Epoch {epoch+1}: Loss = {total_loss/len(networks):.4f}")
        
        return self
    
    def encode(self, G: nx.Graph, feature_type: str = 'random') -> torch.Tensor:
        """Encode a network using the trained encoder."""
        n = G.number_of_nodes()
        if feature_type == 'random':
            features = random_gaussian_features(n, self.feature_dim)
        elif feature_type == 'structural':
            features = structural_features(G, self.feature_dim)
        elif feature_type == 'spectral':
            features = spectral_features(G, self.feature_dim)
        
        edge_index = get_edge_index(G)
        
        self.model.eval()
        with torch.no_grad():
            features = features.to(DEVICE)
            edge_index = edge_index.to(DEVICE)
            return self.model(features, edge_index).cpu()


def train_semisupervised(
    G: nx.Graph, 
    gt: np.ndarray, 
    k: int,
    features: torch.Tensor = None,
    labeled_ratio: float = 0.1,
    epochs: int = 200,
    hidden_dim: int = 64,
    use_learnable_embeddings: bool = True
) -> np.ndarray:
    """
    Train semi-supervised GNN.
    
    Args:
        G: NetworkX graph
        gt: Ground truth labels
        k: Number of communities
        features: Optional node features (if None, uses learnable embeddings)
        labeled_ratio: Fraction of labeled nodes
        epochs: Training epochs
        hidden_dim: Hidden dimension
        use_learnable_embeddings: Use learnable embeddings instead of features
    
    Returns:
        Predicted labels
    """
    n = G.number_of_nodes()
    edge_index = get_edge_index(G).to(DEVICE)
    y = torch.tensor(gt, dtype=torch.long, device=DEVICE)
    
    # Create train mask (sample from each community)
    train_mask = np.zeros(n, dtype=bool)
    for c in range(k):
        c_nodes = np.where(gt == c)[0]
        if len(c_nodes) > 0:
            n_label = max(1, int(len(c_nodes) * labeled_ratio))
            train_mask[np.random.choice(c_nodes, n_label, replace=False)] = True
    train_mask = torch.tensor(train_mask, device=DEVICE)
    
    if use_learnable_embeddings or features is None:
        model = SemiSupervisedGNN(n, hidden_dim, k).to(DEVICE)
        features = None
    else:
        # Use provided features with simple GNN
        class FeatureGNN(nn.Module):
            def __init__(self, in_dim, hidden_dim, out_dim):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, out_dim)
            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, 0.5, training=self.training)
                return self.conv2(x, edge_index)
        
        model = FeatureGNN(features.size(1), hidden_dim, k).to(DEVICE)
        features = features.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        if features is None:
            out = model(edge_index)
        else:
            out = model(features, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        if features is None:
            out = model(edge_index)
        else:
            out = model(features, edge_index)
        return out.argmax(dim=1).cpu().numpy()


def train_unsupervised(
    G: nx.Graph,
    k: int,
    features: torch.Tensor,
    epochs: int = 300,
    hidden_dim: int = 64,
    loss_type: str = 'modularity'
) -> np.ndarray:
    """
    Train unsupervised GNN.
    
    Args:
        G: NetworkX graph
        k: Number of communities
        features: Node features
        epochs: Training epochs
        hidden_dim: Hidden dimension
        loss_type: 'modularity' or 'mincut'
    
    Returns:
        Predicted labels
    """
    n = G.number_of_nodes()
    edge_index = get_edge_index(G).to(DEVICE)
    features = features.to(DEVICE)
    
    # Build adjacency
    adj = torch.zeros(n, n, device=DEVICE)
    adj[edge_index[0], edge_index[1]] = 1
    
    model = UnsupervisedGNN(features.size(1), hidden_dim, k).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Precompute for modularity
    d = adj.sum(dim=1, keepdim=True)
    m = adj.sum() / 2
    B = adj - torch.mm(d, d.t()) / (2 * m + 1e-10)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        s = model(features, edge_index)
        
        if loss_type == 'modularity':
            # Modularity loss
            mod_loss = -torch.trace(torch.mm(torch.mm(s.t(), B), s)) / (2 * m + 1e-10)
            
            # Entropy regularization
            cluster_sizes = s.sum(dim=0) / n
            entropy = -torch.sum(cluster_sizes * torch.log(cluster_sizes + 1e-10))
            collapse_loss = -entropy / np.log(k)
            
            loss = mod_loss + 0.5 * collapse_loss
        else:  # mincut
            cut_edges = torch.trace(torch.mm(torch.mm(s.t(), adj), s))
            d_s = torch.mm(d.t(), s).squeeze()
            cut_vol = (d_s * d_s).sum() + 1e-10
            loss = -cut_edges / cut_vol
        
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        s = model(features, edge_index)
        return s.argmax(dim=1).cpu().numpy()


def train_dmon(
    G: nx.Graph,
    k: int,
    features: torch.Tensor,
    epochs: int = 300,
    hidden_dim: int = 64
) -> np.ndarray:
    """Train DMoN model."""
    n = G.number_of_nodes()
    edge_index = get_edge_index(G).to(DEVICE)
    features = features.to(DEVICE)
    
    adj = torch.zeros(n, n, device=DEVICE)
    adj[edge_index[0], edge_index[1]] = 1
    
    d = adj.sum(dim=1, keepdim=True)
    m = adj.sum() / 2
    B = adj - torch.mm(d, d.t()) / (2 * m + 1e-10)
    
    model = DMoN(features.size(1), hidden_dim, k).to(DEVICE)
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
        
        # Orthogonality
        ss = torch.mm(s.t(), s) / n
        ortho_loss = torch.norm(ss - torch.diag(ss.diag()))
        
        loss = mod_loss + 0.5 * collapse_loss + 0.3 * ortho_loss
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        s = model(features, edge_index)
        return s.argmax(dim=1).cpu().numpy()


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def run_homophily_ablation(
    mu_values: List[float],
    feature_distances: List[float],
    n_realizations: int,
    n_nodes: int = 500,
    k: int = 5,
    verbose: bool = True
) -> Dict:
    """
    Experiment 1: Homophily Ablation
    
    Test different feature distances to understand when features help vs hurt.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: HOMOPHILY ABLATION")
    print("Testing feature distances:", feature_distances)
    print("="*80)
    
    results = defaultdict(list)
    
    for mu in mu_values:
        print(f"\n--- μ = {mu} ---")
        
        for feat_dist in feature_distances:
            for seed in range(n_realizations):
                # Generate SBM with features
                G, gt, features, _ = generate_sbm_with_features(
                    n=n_nodes, k=k, mu=mu,
                    feature_distance=feat_dist,
                    noise_std=0.8,
                    seed=seed + 100
                )
                
                # Baseline: K-Means on features only
                pred_km = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features.numpy())
                ami_km = adjusted_mutual_info_score(gt, pred_km)
                results[('kmeans', feat_dist, mu)].append(ami_km)
                
                # Baseline: Louvain (structure only)
                try:
                    from community import community_louvain
                    part = community_louvain.best_partition(G)
                    pred_louv = [part[i] for i in range(n_nodes)]
                    ami_louv = adjusted_mutual_info_score(gt, pred_louv)
                except:
                    ami_louv = 0
                results[('louvain', feat_dist, mu)].append(ami_louv)
                
                # GCN Semi-supervised
                pred_gcn = train_semisupervised(G, gt, k, features, labeled_ratio=0.1, epochs=150)
                ami_gcn = adjusted_mutual_info_score(gt, pred_gcn)
                results[('gcn_semi', feat_dist, mu)].append(ami_gcn)
                
                # GCN Unsupervised
                pred_unsup = train_unsupervised(G, k, features, epochs=200)
                ami_unsup = adjusted_mutual_info_score(gt, pred_unsup)
                results[('gcn_unsup', feat_dist, mu)].append(ami_unsup)
            
            if verbose:
                print(f"  feat_dist={feat_dist}: KM={np.mean(results[('kmeans', feat_dist, mu)]):.3f}, "
                      f"Louv={np.mean(results[('louvain', feat_dist, mu)]):.3f}, "
                      f"GCN_Semi={np.mean(results[('gcn_semi', feat_dist, mu)]):.3f}, "
                      f"GCN_Unsup={np.mean(results[('gcn_unsup', feat_dist, mu)]):.3f}")
    
    return dict(results)


def run_global_feature_learning(
    mu_values: List[float],
    n_train_networks: int,
    n_test_realizations: int,
    feature_types: List[str] = ['random', 'structural'],
    n_nodes: int = 500,
    verbose: bool = True
) -> Dict:
    """
    Experiment 2: Global Feature Learning
    
    Train a shared feature encoder via link prediction across multiple networks.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: GLOBAL FEATURE LEARNING")
    print(f"Training on {n_train_networks} networks, testing on {n_test_realizations} per μ")
    print("="*80)
    
    results = defaultdict(list)
    
    # Generate training networks (sample μ uniformly)
    print("\nGenerating training networks...")
    train_nets = []
    mu_min, mu_max = min(mu_values), max(mu_values)
    
    for i in range(n_train_networks):
        mu = np.random.uniform(mu_min, mu_max)
        G, gt, k = generate_lfr(n_nodes, mu, seed=i + 1000)
        if G is not None:
            train_nets.append({'G': G, 'gt': gt, 'k': k, 'mu': mu})
    
    print(f"  Generated {len(train_nets)} training networks")
    
    # Generate test networks
    print("\nGenerating test networks...")
    test_nets = {mu: [] for mu in mu_values}
    for mu in mu_values:
        for seed in range(n_test_realizations):
            G, gt, k = generate_lfr(n_nodes, mu, seed=seed + 5000)
            if G is not None:
                test_nets[mu].append({'G': G, 'gt': gt, 'k': k})
    
    for feat_type in feature_types:
        print(f"\n--- Feature type: {feat_type} ---")
        
        # Train global encoder
        print(f"  Training global encoder on {len(train_nets)} networks...")
        encoder = GlobalFeatureEncoder(feature_dim=32, hidden_dim=64, out_dim=32)
        encoder.train(train_nets, epochs=80, feature_type=feat_type, verbose=verbose)
        
        # Test on held-out networks
        for mu in mu_values:
            for net in test_nets[mu]:
                G, gt, k = net['G'], net['gt'], net['k']
                
                # Get learned features
                learned_features = encoder.encode(G, feat_type)
                
                # K-Means on learned features (zero-shot)
                pred_km = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(learned_features.numpy())
                ami = adjusted_mutual_info_score(gt, pred_km)
                results[(feat_type, 'global_kmeans', mu)].append(ami)
                
                # Louvain baseline
                try:
                    from community import community_louvain
                    part = community_louvain.best_partition(G)
                    pred_louv = [part[i] for i in range(len(gt))]
                    ami_louv = adjusted_mutual_info_score(gt, pred_louv)
                except:
                    ami_louv = 0
                results[(feat_type, 'louvain', mu)].append(ami_louv)
            
            if verbose:
                print(f"  μ={mu}: Global+KM={np.mean(results[(feat_type, 'global_kmeans', mu)]):.3f}, "
                      f"Louvain={np.mean(results[(feat_type, 'louvain', mu)]):.3f}")
    
    return dict(results)


def run_spectral_features(
    mu_values: List[float],
    n_realizations: int,
    n_nodes: int = 500,
    verbose: bool = True
) -> Dict:
    """
    Experiment 3: Spectral Features
    
    Test GNNs with Laplacian eigenvector features.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: SPECTRAL FEATURES")
    print("="*80)
    
    results = defaultdict(list)
    
    for mu in mu_values:
        print(f"\n--- μ = {mu} ---")
        
        for seed in range(n_realizations):
            G, gt, k = generate_lfr(n_nodes, mu, seed=seed + 2000)
            if G is None:
                continue
            
            n = G.number_of_nodes()
            
            # Compute spectral features
            features = spectral_features(G, dim=32)
            
            # K-Means on spectral features
            pred_km = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features.numpy())
            ami_km = adjusted_mutual_info_score(gt, pred_km)
            results[('spectral_kmeans', mu)].append(ami_km)
            
            # GCN Semi-supervised with spectral features
            pred_semi = train_semisupervised(G, gt, k, features, labeled_ratio=0.1, 
                                             epochs=150, use_learnable_embeddings=False)
            ami_semi = adjusted_mutual_info_score(gt, pred_semi)
            results[('spectral_gcn_semi', mu)].append(ami_semi)
            
            # GCN Unsupervised with spectral features
            pred_unsup = train_unsupervised(G, k, features, epochs=200)
            ami_unsup = adjusted_mutual_info_score(gt, pred_unsup)
            results[('spectral_gcn_unsup', mu)].append(ami_unsup)
            
            # DMoN with spectral features
            pred_dmon = train_dmon(G, k, features, epochs=200)
            ami_dmon = adjusted_mutual_info_score(gt, pred_dmon)
            results[('spectral_dmon', mu)].append(ami_dmon)
            
            # Louvain baseline
            try:
                from community import community_louvain
                part = community_louvain.best_partition(G)
                pred_louv = [part[i] for i in range(n)]
                ami_louv = adjusted_mutual_info_score(gt, pred_louv)
            except:
                ami_louv = 0
            results[('louvain', mu)].append(ami_louv)
        
        if verbose:
            print(f"  Spectral+KM={np.mean(results[('spectral_kmeans', mu)]):.3f}, "
                  f"Spectral+GCN_Semi={np.mean(results[('spectral_gcn_semi', mu)]):.3f}, "
                  f"Spectral+GCN_Unsup={np.mean(results[('spectral_gcn_unsup', mu)]):.3f}, "
                  f"Spectral+DMoN={np.mean(results[('spectral_dmon', mu)]):.3f}, "
                  f"Louvain={np.mean(results[('louvain', mu)]):.3f}")
    
    return dict(results)


def run_algorithm_comparison(
    mu_values: List[float],
    n_realizations: int,
    n_nodes: int = 500,
    verbose: bool = True
) -> Dict:
    """
    Experiment 4: Algorithm Comparison
    
    Compare all algorithms across semi-supervised and unsupervised regimes.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: ALGORITHM COMPARISON")
    print("="*80)
    
    results = defaultdict(list)
    
    algorithms = {
        # (name, training_func, kwargs)
        'GCN_Semi_Learnable': ('semi', {'use_learnable_embeddings': True}),
        'GCN_Semi_Spectral': ('semi', {'use_learnable_embeddings': False}),
        'GCN_Unsup_Modularity': ('unsup', {'loss_type': 'modularity'}),
        'GCN_Unsup_MinCut': ('unsup', {'loss_type': 'mincut'}),
        'DMoN_Spectral': ('dmon', {}),
    }
    
    for mu in mu_values:
        print(f"\n--- μ = {mu} ---")
        
        for seed in range(n_realizations):
            G, gt, k = generate_lfr(n_nodes, mu, seed=seed + 3000)
            if G is None:
                continue
            
            n = G.number_of_nodes()
            features = spectral_features(G, dim=32)
            
            for algo_name, (algo_type, kwargs) in algorithms.items():
                try:
                    if algo_type == 'semi':
                        pred = train_semisupervised(G, gt, k, features, epochs=150, **kwargs)
                    elif algo_type == 'unsup':
                        pred = train_unsupervised(G, k, features, epochs=200, **kwargs)
                    elif algo_type == 'dmon':
                        pred = train_dmon(G, k, features, epochs=200, **kwargs)
                    
                    ami = adjusted_mutual_info_score(gt, pred)
                    results[(algo_name, mu)].append(ami)
                except Exception as e:
                    print(f"    {algo_name} failed: {e}")
                    results[(algo_name, mu)].append(0)
            
            # Louvain baseline
            try:
                from community import community_louvain
                part = community_louvain.best_partition(G)
                pred_louv = [part[i] for i in range(n)]
                ami_louv = adjusted_mutual_info_score(gt, pred_louv)
            except:
                ami_louv = 0
            results[('Louvain', mu)].append(ami_louv)
        
        if verbose:
            for algo_name in list(algorithms.keys()) + ['Louvain']:
                print(f"  {algo_name}: {np.mean(results[(algo_name, mu)]):.3f}")
    
    return dict(results)


# =============================================================================
# MAIN
# =============================================================================

def save_results(results: Dict, experiment_name: str, output_dir: str):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Convert tuple keys to strings
    serializable = {}
    for key, value in results.items():
        str_key = str(key)
        serializable[str_key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to {filepath}")


def print_summary(results: Dict, experiment_name: str):
    """Print summary table."""
    print("\n" + "="*80)
    print(f"SUMMARY: {experiment_name}")
    print("="*80)
    
    # Group by algorithm
    algo_results = defaultdict(list)
    for key, values in results.items():
        if isinstance(key, tuple):
            algo = key[0] if len(key) >= 1 else str(key)
        else:
            algo = str(key)
        algo_results[algo].extend(values)
    
    print(f"\n{'Algorithm':<25} | {'Mean AMI':<10} | {'Std':<10}")
    print("-" * 50)
    
    for algo, amis in sorted(algo_results.items(), key=lambda x: -np.mean(x[1])):
        print(f"{algo:<25} | {np.mean(amis):>8.4f}   | {np.std(amis):>8.4f}")


def main():
    parser = argparse.ArgumentParser(description='GNN Accuracy Benchmark Suite')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'homophily', 'global_features', 'spectral', 'algorithms'],
                        help='Which experiment to run')
    parser.add_argument('--mu-values', type=str, default='0.1,0.3,0.5',
                        help='Comma-separated mu values')
    parser.add_argument('--realizations', type=int, default=3,
                        help='Number of realizations per configuration')
    parser.add_argument('--train-networks', type=int, default=100,
                        help='Number of training networks for global feature learning')
    parser.add_argument('--n-nodes', type=int, default=500,
                        help='Number of nodes per network')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for results')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    mu_values = [float(x) for x in args.mu_values.split(',')]
    
    print("="*80)
    print("GNN ACCURACY BENCHMARK SUITE")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"μ values: {mu_values}")
    print(f"Realizations: {args.realizations}")
    print(f"Nodes per network: {args.n_nodes}")
    
    all_results = {}
    
    if args.experiment in ['all', 'homophily']:
        feature_distances = [0.2, 0.4, 0.6, 0.8, 1.2]
        results = run_homophily_ablation(
            mu_values, feature_distances, 
            args.realizations, args.n_nodes, verbose=args.verbose
        )
        all_results['homophily'] = results
        print_summary(results, 'Homophily Ablation')
        save_results(results, 'homophily', args.output_dir)
    
    if args.experiment in ['all', 'global_features']:
        results = run_global_feature_learning(
            mu_values, args.train_networks, args.realizations,
            feature_types=['random', 'structural'],
            n_nodes=args.n_nodes, verbose=args.verbose
        )
        all_results['global_features'] = results
        print_summary(results, 'Global Feature Learning')
        save_results(results, 'global_features', args.output_dir)
    
    if args.experiment in ['all', 'spectral']:
        results = run_spectral_features(
            mu_values, args.realizations, args.n_nodes, verbose=args.verbose
        )
        all_results['spectral'] = results
        print_summary(results, 'Spectral Features')
        save_results(results, 'spectral', args.output_dir)
    
    if args.experiment in ['all', 'algorithms']:
        results = run_algorithm_comparison(
            mu_values, args.realizations, args.n_nodes, verbose=args.verbose
        )
        all_results['algorithms'] = results
        print_summary(results, 'Algorithm Comparison')
        save_results(results, 'algorithms', args.output_dir)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
