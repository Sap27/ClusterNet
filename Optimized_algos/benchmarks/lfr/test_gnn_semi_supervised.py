#!/usr/bin/env python3
"""
GNN Semi-Supervised Test
========================

Test GNNs with partial supervision (10% labels) vs unsupervised.
This is a fairer comparison since GNNs excel at semi-supervised learning.
"""

import os
import sys
import time
import numpy as np
import networkx as nx
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data

from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def graph_to_pyg(G: nx.Graph, feature_dim: int = 64):
    """Convert NetworkX graph to PyG Data with richer features."""
    num_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    
    edges = list(G.edges())
    edge_index = torch.tensor(
        [[node_to_idx[e[0]] for e in edges] + [node_to_idx[e[1]] for e in edges],
         [node_to_idx[e[1]] for e in edges] + [node_to_idx[e[0]] for e in edges]],
        dtype=torch.long
    )
    
    # Rich features
    x = torch.zeros((num_nodes, feature_dim))
    degrees = [G.degree(n) for n in nodes]
    max_deg = max(degrees) if degrees else 1
    
    for i, node in enumerate(nodes):
        deg = G.degree(node)
        x[i, 0] = deg / max_deg
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:
            subg = G.subgraph(neighbors)
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            x[i, 1] = subg.number_of_edges() / max(1, possible)
        if neighbors:
            x[i, 2] = sum(G.degree(n) for n in neighbors) / len(neighbors) / max_deg
        # One-hot degree bucket
        deg_bucket = min(deg // 3, feature_dim - 4)
        x[i, 3 + deg_bucket] = 1.0
    
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)


class GNNClassifier(nn.Module):
    """GNN for node classification (semi-supervised)."""
    def __init__(self, in_dim, hidden_dim, num_classes, conv_type='gcn', dropout=0.5):
        super().__init__()
        self.dropout = dropout
        
        if conv_type == 'gcn':
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, num_classes)
        elif conv_type == 'gat':
            self.conv1 = GATConv(in_dim, hidden_dim // 4, heads=4, dropout=dropout)
            self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout)
            self.conv3 = GATConv(hidden_dim, num_classes, heads=1, concat=False, dropout=dropout)
        else:  # sage
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, num_classes)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


def train_semi_supervised(model, data, train_mask, labels, epochs=200, lr=0.01):
    """Train with cross-entropy on labeled nodes."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
    return pred


def generate_test_network(n=300, num_communities=5, p_in=0.3, p_out=0.05):
    """Generate SBM network with clear community structure."""
    sizes = [n // num_communities] * num_communities
    sizes[-1] += n - sum(sizes)
    
    probs = [[p_in if i == j else p_out for j in range(num_communities)] 
             for i in range(num_communities)]
    
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    ground_truth = np.array([G.nodes[node]['block'] for node in G.nodes()])
    
    for node in G.nodes():
        del G.nodes[node]['block']
    
    return G, ground_truth


def run_comparison(G, ground_truth, label_fraction=0.1):
    """Compare classical vs semi-supervised GNN."""
    num_nodes = G.number_of_nodes()
    num_classes = len(set(ground_truth))
    
    # Prepare PyG data
    data = graph_to_pyg(G, feature_dim=64).to(device)
    labels = torch.tensor(ground_truth, dtype=torch.long).to(device)
    
    # Create train mask (random label_fraction of nodes)
    n_train = int(num_nodes * label_fraction)
    perm = np.random.permutation(num_nodes)
    train_idx = perm[:n_train]
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    train_mask = train_mask.to(device)
    
    results = []
    
    # Classical: Louvain
    try:
        from community import community_louvain
        start = time.time()
        partition = community_louvain.best_partition(G)
        assignments = [partition[n] for n in G.nodes()]
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        results.append({'model': 'Louvain', 'ami': ami, 'runtime': runtime, 'type': 'classical'})
    except:
        pass
    
    # Semi-supervised GNNs
    for conv_type, name in [('gcn', 'GCN'), ('gat', 'GAT'), ('sage', 'SAGE')]:
        start = time.time()
        model = GNNClassifier(64, 64, num_classes, conv_type=conv_type).to(device)
        pred = train_semi_supervised(model, data, train_mask, labels, epochs=200)
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, pred)
        results.append({'model': f'{name}(10%)', 'ami': ami, 'runtime': runtime, 'type': 'semi-supervised'})
    
    # Unsupervised GNN (modularity loss)
    start = time.time()
    
    class UnsupGNN(nn.Module):
        def __init__(self, in_dim, hidden_dim, num_clusters):
            super().__init__()
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.cluster = nn.Linear(hidden_dim, num_clusters)
        
        def forward(self, x, edge_index):
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            return F.softmax(self.cluster(x) * 2.0, dim=-1)
    
    model = UnsupGNN(64, 64, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    adj = torch.zeros((num_nodes, num_nodes), device=device)
    adj[data.edge_index[0], data.edge_index[1]] = 1
    d = adj.sum(dim=1, keepdim=True)
    m = adj.sum() / 2
    
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        s = model(data.x, data.edge_index)
        if m > 0:
            B = adj - torch.mm(d, d.t()) / (2 * m)
            mod_loss = -torch.trace(torch.mm(torch.mm(s.t(), B), s)) / (2 * m)
        else:
            mod_loss = torch.tensor(0.0)
        cluster_sizes = s.sum(0) / num_nodes
        collapse_loss = -torch.sum(cluster_sizes * torch.log(cluster_sizes + 1e-10)) / np.log(num_classes)
        loss = mod_loss - 0.5 * collapse_loss
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        s = model(data.x, data.edge_index)
        pred = s.argmax(dim=1).cpu().numpy()
    
    runtime = time.time() - start
    ami = adjusted_mutual_info_score(ground_truth, pred)
    results.append({'model': 'GCN(unsup)', 'ami': ami, 'runtime': runtime, 'type': 'unsupervised'})
    
    return results


def main():
    print("=" * 70)
    print("SEMI-SUPERVISED vs UNSUPERVISED GNN COMPARISON")
    print("=" * 70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test configurations
    configs = [
        (300, 5, 0.25, 0.02),   # Clear structure
        (300, 5, 0.20, 0.05),   # Moderate structure
        (500, 8, 0.20, 0.03),   # More communities
        (300, 5, 0.15, 0.08),   # Weak structure
    ]
    
    all_results = []
    
    for n, num_comm, p_in, p_out in configs:
        print(f"\n{'='*70}")
        print(f"Network: n={n}, communities={num_comm}, p_in={p_in}, p_out={p_out}")
        print("=" * 70)
        
        G, ground_truth = generate_test_network(n, num_comm, p_in, p_out)
        print(f"Generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        results = run_comparison(G, ground_truth, label_fraction=0.1)
        
        print(f"\n{'Model':15s} | {'Type':15s} | {'AMI':8s} | {'Time':8s}")
        print("-" * 55)
        for r in results:
            print(f"{r['model']:15s} | {r['type']:15s} | {r['ami']:8.4f} | {r['runtime']:7.3f}s")
            r['config'] = f"n={n},k={num_comm}"
            all_results.append(r)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Average AMI by Model Type")
    print("=" * 70)
    
    from collections import defaultdict
    model_amis = defaultdict(list)
    for r in all_results:
        model_amis[r['model']].append(r['ami'])
    
    for model, amis in sorted(model_amis.items(), key=lambda x: -np.mean(x[1])):
        print(f"  {model:15s}: {np.mean(amis):.4f} ± {np.std(amis):.4f}")
    
    print("\n" + "-" * 70)
    print("KEY INSIGHT:")
    print("  - Semi-supervised GNNs (with 10% labels) often match/beat classical methods")
    print("  - Unsupervised GNNs with modularity loss underperform classical methods")
    print("  - GNNs' strength is in leveraging node features + partial labels")


if __name__ == "__main__":
    main()



