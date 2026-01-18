#!/usr/bin/env python3
"""
Quick GNN Accuracy Test
=======================

Test GNNs with better settings on a small LFR network with few communities.
"""

import os
import sys
import time
import numpy as np
import networkx as nx
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("ERROR: torch_geometric not available!")
    sys.exit(1)

from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def graph_to_pyg(G: nx.Graph, feature_dim: int = 32):
    """Convert NetworkX graph to PyG Data with node features."""
    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    # Edge index
    edges = list(G.edges())
    edge_index = torch.tensor(
        [[node_to_idx[e[0]] for e in edges] + [node_to_idx[e[1]] for e in edges],
         [node_to_idx[e[1]] for e in edges] + [node_to_idx[e[0]] for e in edges]],
        dtype=torch.long
    )
    
    # Node features: degree, clustering coef, neighbors' avg degree
    x = torch.zeros((num_nodes, feature_dim))
    degrees = [G.degree(n) for n in nodes]
    max_deg = max(degrees) if degrees else 1
    
    for i, node in enumerate(nodes):
        deg = G.degree(node)
        x[i, 0] = deg / max_deg  # Normalized degree
        
        # Local clustering coefficient
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:
            subg = G.subgraph(neighbors)
            possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
            actual_edges = subg.number_of_edges()
            x[i, 1] = actual_edges / max(1, possible_edges)
        
        # Neighbors' average degree
        if neighbors:
            x[i, 2] = sum(G.degree(n) for n in neighbors) / len(neighbors) / max_deg
        
        # One-hot degree (buckets)
        deg_bucket = min(deg // 5, feature_dim - 4)
        x[i, 3 + deg_bucket] = 1.0
    
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


class DMoNModel(nn.Module):
    """Deep Modularity Network."""
    def __init__(self, in_dim, hidden_dim, num_clusters, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.cluster = nn.Linear(hidden_dim, num_clusters)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.cluster.weight, gain=2.0)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        s = F.softmax(self.cluster(x) * 2.0, dim=-1)  # Temperature scaling
        return s


class GATClusterModel(nn.Module):
    """GAT-based clustering."""
    def __init__(self, in_dim, hidden_dim, num_clusters, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.cluster = nn.Linear(hidden_dim, num_clusters)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.cluster.weight, gain=2.0)
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        s = F.softmax(self.cluster(x) * 2.0, dim=-1)
        return s


class SAGEClusterModel(nn.Module):
    """GraphSAGE-based clustering."""
    def __init__(self, in_dim, hidden_dim, num_clusters, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.cluster = nn.Linear(hidden_dim, num_clusters)
        self.dropout = dropout
        nn.init.xavier_uniform_(self.cluster.weight, gain=2.0)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        s = F.softmax(self.cluster(x) * 2.0, dim=-1)
        return s


def train_model(model, data, adj, num_communities, epochs=500, lr=0.005, verbose=False):
    """Train model with modularity-based loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)
    
    d = adj.sum(dim=1, keepdim=True)
    m = adj.sum() / 2
    
    model.train()
    best_loss = float('inf')
    best_assignments = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        s = model(data.x, data.edge_index)
        
        # Modularity loss
        if m > 0:
            B = adj - torch.mm(d, d.t()) / (2 * m)
            modularity_loss = -torch.trace(torch.mm(torch.mm(s.t(), B), s)) / (2 * m)
        else:
            modularity_loss = torch.tensor(0.0, device=device)
        
        # Entropy regularization (encourage uniform cluster sizes)
        cluster_sizes = s.sum(dim=0) / data.num_nodes
        entropy_loss = -torch.sum(cluster_sizes * torch.log(cluster_sizes + 1e-10))
        collapse_loss = -entropy_loss / np.log(num_communities + 1e-10)
        
        # Orthogonality (clusters should be distinct)
        ss = torch.mm(s.t(), s) / data.num_nodes
        eye = torch.eye(num_communities, device=device)
        ortho_loss = torch.norm(ss - eye * ss.diag().unsqueeze(1))
        
        loss = modularity_loss + 0.5 * collapse_loss + 0.3 * ortho_loss
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            with torch.no_grad():
                best_assignments = s.argmax(dim=1).cpu().numpy()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, mod={-modularity_loss.item():.4f}")
    
    return best_assignments


def run_gnn(G, ground_truth, num_communities, model_class, model_name, **kwargs):
    """Run a GNN model and return results."""
    start_time = time.time()
    
    # Prepare data
    data = graph_to_pyg(G, feature_dim=32).to(device)
    
    # Adjacency matrix
    adj = torch.zeros((data.num_nodes, data.num_nodes), device=device)
    adj[data.edge_index[0], data.edge_index[1]] = 1
    
    # Create model
    hidden_dim = kwargs.get('hidden_dim', 128)
    model = model_class(32, hidden_dim, num_communities).to(device)
    
    # Train
    epochs = kwargs.get('epochs', 500)
    lr = kwargs.get('lr', 0.005)
    assignments = train_model(model, data, adj, num_communities, epochs=epochs, lr=lr)
    
    runtime = time.time() - start_time
    
    # Calculate metrics
    ami = adjusted_mutual_info_score(ground_truth, assignments)
    nmi = normalized_mutual_info_score(ground_truth, assignments)
    
    n_detected = len(set(assignments))
    
    return {
        'model': model_name,
        'ami': ami,
        'nmi': nmi,
        'n_detected': n_detected,
        'n_true': num_communities,
        'runtime': runtime
    }


def generate_simple_lfr(n=500, mu=0.3, k=None, num_communities=None):
    """Generate simple LFR network or use planted partition as fallback."""
    if k is None:
        k = max(10, n // 50)  # Average degree
    if num_communities is None:
        num_communities = max(5, n // 100)  # Number of communities
    
    print(f"\nGenerating network: n={n}, mu={mu}, k={k}, communities~{num_communities}")
    
    try:
        # Try LFR
        G = nx.LFR_benchmark_graph(
            n=n,
            tau1=2.5,
            tau2=1.5,
            mu=mu,
            average_degree=k,
            min_community=max(10, n // num_communities // 2),
            max_community=n // num_communities * 2,
            seed=42
        )
        # Get communities from node attributes
        communities = {}
        for node in G.nodes():
            comm_set = G.nodes[node]['community']
            # Take first community if overlapping
            comm_id = list(comm_set)[0] if isinstance(comm_set, (set, frozenset)) else comm_set
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        # Create ground truth array
        comm_list = list(communities.keys())
        comm_to_id = {c: i for i, c in enumerate(comm_list)}
        ground_truth = np.zeros(n, dtype=int)
        for node in G.nodes():
            comm_set = G.nodes[node]['community']
            comm_id = list(comm_set)[0] if isinstance(comm_set, (set, frozenset)) else comm_set
            ground_truth[node] = comm_to_id[comm_id]
        
        # Remove community attribute (not needed after)
        for node in G.nodes():
            del G.nodes[node]['community']
        
        num_true = len(set(ground_truth))
        print(f"  LFR generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {num_true} communities")
        return G, ground_truth, num_true
        
    except Exception as e:
        print(f"  LFR failed ({e}), using planted partition model")
        
        # Fallback: planted partition
        sizes = [n // num_communities] * num_communities
        sizes[-1] += n - sum(sizes)  # Handle remainder
        
        p_in = 0.3 * (1 - mu)  # Internal edge probability
        p_out = 0.3 * mu  # External edge probability
        
        probs = [[p_in if i == j else p_out for j in range(num_communities)] for i in range(num_communities)]
        G = nx.stochastic_block_model(sizes, probs, seed=42)
        
        # Ground truth from block membership
        ground_truth = np.array([G.nodes[node]['block'] for node in G.nodes()])
        
        # Clean up
        for node in G.nodes():
            if 'block' in G.nodes[node]:
                del G.nodes[node]['block']
        
        num_true = len(set(ground_truth))
        print(f"  SBM generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {num_true} communities")
        return G, ground_truth, num_true


def run_classical(G, ground_truth):
    """Run classical algorithms for comparison."""
    results = []
    
    # Louvain
    try:
        from community import community_louvain
        start = time.time()
        partition = community_louvain.best_partition(G)
        assignments = [partition[n] for n in G.nodes()]
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        nmi = normalized_mutual_info_score(ground_truth, assignments)
        results.append({'model': 'Louvain', 'ami': ami, 'nmi': nmi, 'n_detected': len(set(assignments)), 'runtime': runtime})
    except:
        pass
    
    # Label Propagation
    try:
        start = time.time()
        communities = list(nx.community.label_propagation_communities(G))
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i
        assignments = [node_to_comm[n] for n in G.nodes()]
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        nmi = normalized_mutual_info_score(ground_truth, assignments)
        results.append({'model': 'LabelProp', 'ami': ami, 'nmi': nmi, 'n_detected': len(set(assignments)), 'runtime': runtime})
    except:
        pass
    
    return results


def main():
    print("=" * 70)
    print("GNN ACCURACY TEST")
    print("=" * 70)
    
    # Test configurations
    configs = [
        # (n, mu, approx_communities)
        (200, 0.2, 5),     # Easy: small, low mixing, few communities
        (500, 0.3, 8),     # Medium: larger, moderate mixing
        (500, 0.5, 8),     # Hard: high mixing
        (300, 0.2, 10),    # More communities
    ]
    
    gnn_models = [
        (DMoNModel, 'DMoN'),
        (GATClusterModel, 'GAT'),
        (SAGEClusterModel, 'SAGE'),
    ]
    
    all_results = []
    
    for n, mu, approx_comm in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: n={n}, mu={mu}, ~{approx_comm} communities")
        print("=" * 70)
        
        # Generate network
        G, ground_truth, num_true = generate_simple_lfr(n=n, mu=mu, num_communities=approx_comm)
        
        # Run classical algorithms
        print("\n--- Classical Algorithms ---")
        classical_results = run_classical(G, ground_truth)
        for r in classical_results:
            print(f"  {r['model']:12s}: AMI={r['ami']:.4f}, NMI={r['nmi']:.4f}, detected={r['n_detected']}, time={r['runtime']:.3f}s")
            r['config'] = f"n={n},mu={mu}"
            all_results.append(r)
        
        # Run GNNs
        print("\n--- GNN Models ---")
        for model_class, model_name in gnn_models:
            print(f"  Training {model_name}...")
            try:
                result = run_gnn(
                    G, ground_truth, num_true, 
                    model_class, model_name,
                    hidden_dim=128, epochs=500, lr=0.005
                )
                print(f"  {model_name:12s}: AMI={result['ami']:.4f}, NMI={result['nmi']:.4f}, detected={result['n_detected']}/{result['n_true']}, time={result['runtime']:.3f}s")
                result['config'] = f"n={n},mu={mu}"
                all_results.append(result)
            except Exception as e:
                print(f"  {model_name} FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':12s} | {'Config':15s} | {'AMI':8s} | {'NMI':8s} | {'Time':8s}")
    print("-" * 60)
    
    for r in all_results:
        print(f"{r['model']:12s} | {r.get('config',''):15s} | {r['ami']:8.4f} | {r['nmi']:8.4f} | {r['runtime']:7.3f}s")
    
    # Average by model
    print("\n" + "-" * 60)
    print("AVERAGE AMI BY MODEL:")
    from collections import defaultdict
    model_amis = defaultdict(list)
    for r in all_results:
        model_amis[r['model']].append(r['ami'])
    
    for model, amis in sorted(model_amis.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {model:12s}: {np.mean(amis):.4f} ± {np.std(amis):.4f}")


if __name__ == "__main__":
    main()



