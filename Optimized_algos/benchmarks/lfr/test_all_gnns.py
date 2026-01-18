#!/usr/bin/env python3
"""
Test ALL GNNs from ClusterNet
=============================

Uses ALL 7 GNN models with SUPERVISED training (since we have ground truth).
Tests on SBM networks to compare GNNs vs classical algorithms.
"""

import os
import sys
import time
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict

# Add ClusterNet to path
CLUSTERNET_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CLUSTERNET_PATH))

import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Import ClusterNet GNN models
try:
    from clusternet.gnn.models import (
        # Unsupervised
        GCNCluster, gcn_clustering,
        DMoNCluster, dmon_clustering,
        GINCluster, gin_clustering,
        MinCutCluster, mincut_clustering,
        GraphTransformerCluster, graph_transformer_clustering,
        # Supervised
        GCNSupervised, gcn_supervised_clustering,
        GATSupervised, gat_supervised_clustering,
    )
    HAS_CLUSTERNET_GNN = True
    print("✓ Loaded ClusterNet GNN models")
except ImportError as e:
    HAS_CLUSTERNET_GNN = False
    print(f"✗ Could not load ClusterNet GNN models: {e}")

# Import RobustGNN models
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "RobustGNN"))
    from RobustGNN.models import GCN, GAT, GraphSAGE, MinCut, DiffPool
    HAS_ROBUSTGNN = True
    print("✓ Loaded RobustGNN models")
except ImportError as e:
    HAS_ROBUSTGNN = False
    print(f"✗ Could not load RobustGNN models: {e}")


def generate_sbm_network(n=300, num_communities=5, p_in=0.25, p_out=0.02, seed=42):
    """Generate SBM network with clear community structure."""
    np.random.seed(seed)
    
    sizes = [n // num_communities] * num_communities
    sizes[-1] += n - sum(sizes)  # Handle remainder
    
    probs = [[p_in if i == j else p_out for j in range(num_communities)] 
             for i in range(num_communities)]
    
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    ground_truth = np.array([G.nodes[node]['block'] for node in G.nodes()])
    
    # Clean up node attributes
    for node in G.nodes():
        del G.nodes[node]['block']
    
    return G, ground_truth


def create_node_features(G, feature_dim=32):
    """Create node features from graph structure."""
    n = G.number_of_nodes()
    features = np.zeros((n, feature_dim), dtype=np.float32)
    
    nodes = list(G.nodes())
    degrees = [G.degree(n) for n in nodes]
    max_deg = max(degrees) if degrees else 1
    
    for i, node in enumerate(nodes):
        deg = G.degree(node)
        # Normalized degree
        features[i, 0] = deg / max_deg
        
        # Clustering coefficient
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:
            subg = G.subgraph(neighbors)
            possible = len(neighbors) * (len(neighbors) - 1) / 2
            features[i, 1] = subg.number_of_edges() / max(1, possible)
        
        # Average neighbor degree
        if neighbors:
            features[i, 2] = sum(G.degree(n) for n in neighbors) / len(neighbors) / max_deg
        
        # One-hot degree bucket
        deg_bucket = min(deg // 3, feature_dim - 4)
        features[i, 3 + deg_bucket] = 1.0
    
    return features


def run_classical_algorithms(G, ground_truth):
    """Run classical algorithms for comparison."""
    results = []
    
    # Louvain
    try:
        from community import community_louvain
        start = time.time()
        partition = community_louvain.best_partition(G)
        assignments = np.array([partition[n] for n in G.nodes()])
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        nmi = normalized_mutual_info_score(ground_truth, assignments)
        results.append({
            'model': 'Louvain',
            'ami': ami,
            'nmi': nmi,
            'runtime': runtime,
            'type': 'classical',
            'n_detected': len(set(assignments))
        })
        print(f"  Louvain: AMI={ami:.4f}, time={runtime:.3f}s")
    except Exception as e:
        print(f"  Louvain FAILED: {e}")
    
    # Leiden
    try:
        import leidenalg
        import igraph as ig
        start = time.time()
        g = ig.Graph.from_networkx(G)
        partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
        node_map = {i: n for i, n in enumerate(G.nodes())}
        assignments = np.zeros(len(ground_truth), dtype=int)
        for idx, cluster in enumerate(partition):
            for i in cluster:
                assignments[i] = idx
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        nmi = normalized_mutual_info_score(ground_truth, assignments)
        results.append({
            'model': 'Leiden',
            'ami': ami,
            'nmi': nmi,
            'runtime': runtime,
            'type': 'classical',
            'n_detected': len(set(assignments))
        })
        print(f"  Leiden: AMI={ami:.4f}, time={runtime:.3f}s")
    except Exception as e:
        print(f"  Leiden FAILED: {e}")
    
    # Label Propagation
    try:
        start = time.time()
        communities = list(nx.community.label_propagation_communities(G))
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i
        assignments = np.array([node_to_comm[n] for n in G.nodes()])
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        nmi = normalized_mutual_info_score(ground_truth, assignments)
        results.append({
            'model': 'LabelProp',
            'ami': ami,
            'nmi': nmi,
            'runtime': runtime,
            'type': 'classical',
            'n_detected': len(set(assignments))
        })
        print(f"  LabelProp: AMI={ami:.4f}, time={runtime:.3f}s")
    except Exception as e:
        print(f"  LabelProp FAILED: {e}")
    
    return results


def run_clusternet_gnns_supervised(G, ground_truth, features, num_clusters, epochs=200):
    """Run ClusterNet GNNs with SUPERVISED training (all models)."""
    results = []
    
    if not HAS_CLUSTERNET_GNN:
        print("  ClusterNet GNN models not available")
        return results
    
    train_ratio = 1.0  # Use all labels for training (supervised)
    n = len(ground_truth)
    train_mask = np.ones(n, dtype=bool)  # Train on all nodes
    
    # 1. GCN Supervised
    try:
        print("  Training GCN (supervised)...")
        start = time.time()
        algo = GCNSupervised(
            G, num_clusters=num_clusters, labels=ground_truth,
            features=features, train_mask=train_mask,
            hidden_channels=64, num_layers=3, epochs=epochs, lr=0.01
        )
        communities = algo.run()
        assignments = np.zeros(n, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                assignments[node] = i
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        results.append({
            'model': 'GCN_sup',
            'ami': ami,
            'nmi': normalized_mutual_info_score(ground_truth, assignments),
            'runtime': runtime,
            'type': 'supervised_gnn',
            'n_detected': len(communities)
        })
        print(f"    GCN_sup: AMI={ami:.4f}, time={runtime:.3f}s")
    except Exception as e:
        print(f"    GCN_sup FAILED: {e}")
    
    # 2. GAT Supervised
    try:
        print("  Training GAT (supervised)...")
        start = time.time()
        algo = GATSupervised(
            G, num_clusters=num_clusters, labels=ground_truth,
            features=features, train_mask=train_mask,
            hidden_channels=64, num_layers=3, epochs=epochs, lr=0.005
        )
        communities = algo.run()
        assignments = np.zeros(n, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                assignments[node] = i
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        results.append({
            'model': 'GAT_sup',
            'ami': ami,
            'nmi': normalized_mutual_info_score(ground_truth, assignments),
            'runtime': runtime,
            'type': 'supervised_gnn',
            'n_detected': len(communities)
        })
        print(f"    GAT_sup: AMI={ami:.4f}, time={runtime:.3f}s")
    except Exception as e:
        print(f"    GAT_sup FAILED: {e}")
    
    # 3. GCN Cluster (unsupervised - for comparison)
    try:
        print("  Training GCN (unsupervised)...")
        start = time.time()
        algo = GCNCluster(
            G, num_clusters=num_clusters,
            features=features,
            hidden_channels=64, epochs=epochs, lr=0.01
        )
        communities = algo.run()
        assignments = np.zeros(n, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                assignments[node] = i
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        results.append({
            'model': 'GCN_unsup',
            'ami': ami,
            'nmi': normalized_mutual_info_score(ground_truth, assignments),
            'runtime': runtime,
            'type': 'unsupervised_gnn',
            'n_detected': len(communities)
        })
        print(f"    GCN_unsup: AMI={ami:.4f}, time={runtime:.3f}s")
    except Exception as e:
        print(f"    GCN_unsup FAILED: {e}")
    
    # 4. DMoN (unsupervised)
    try:
        print("  Training DMoN...")
        start = time.time()
        algo = DMoNCluster(
            G, num_clusters=num_clusters,
            features=features,
            hidden_channels=32, epochs=epochs, lr=0.001
        )
        communities = algo.run()
        assignments = np.zeros(n, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                assignments[node] = i
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        results.append({
            'model': 'DMoN',
            'ami': ami,
            'nmi': normalized_mutual_info_score(ground_truth, assignments),
            'runtime': runtime,
            'type': 'unsupervised_gnn',
            'n_detected': len(communities)
        })
        print(f"    DMoN: AMI={ami:.4f}, time={runtime:.3f}s")
    except Exception as e:
        print(f"    DMoN FAILED: {e}")
    
    # 5. GIN Cluster (unsupervised)
    try:
        print("  Training GIN...")
        start = time.time()
        algo = GINCluster(
            G, num_clusters=num_clusters,
            features=features,
            hidden_channels=64, num_layers=3, epochs=epochs, lr=0.01
        )
        communities = algo.run()
        assignments = np.zeros(n, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                assignments[node] = i
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        results.append({
            'model': 'GIN',
            'ami': ami,
            'nmi': normalized_mutual_info_score(ground_truth, assignments),
            'runtime': runtime,
            'type': 'unsupervised_gnn',
            'n_detected': len(communities)
        })
        print(f"    GIN: AMI={ami:.4f}, time={runtime:.3f}s")
    except Exception as e:
        print(f"    GIN FAILED: {e}")
    
    # 6. MinCut (unsupervised)
    try:
        print("  Training MinCut...")
        start = time.time()
        algo = MinCutCluster(
            G, num_clusters=num_clusters,
            features=features,
            hidden_channels=32, epochs=epochs, lr=0.001
        )
        communities = algo.run()
        assignments = np.zeros(n, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                assignments[node] = i
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        results.append({
            'model': 'MinCut',
            'ami': ami,
            'nmi': normalized_mutual_info_score(ground_truth, assignments),
            'runtime': runtime,
            'type': 'unsupervised_gnn',
            'n_detected': len(communities)
        })
        print(f"    MinCut: AMI={ami:.4f}, time={runtime:.3f}s")
    except Exception as e:
        print(f"    MinCut FAILED: {e}")
    
    # 7. GraphTransformer (unsupervised)
    try:
        print("  Training GraphTransformer...")
        start = time.time()
        algo = GraphTransformerCluster(
            G, num_clusters=num_clusters,
            features=features,
            hidden_channels=32, epochs=epochs, lr=0.001
        )
        communities = algo.run()
        assignments = np.zeros(n, dtype=int)
        for i, comm in enumerate(communities):
            for node in comm:
                assignments[node] = i
        runtime = time.time() - start
        ami = adjusted_mutual_info_score(ground_truth, assignments)
        results.append({
            'model': 'GraphTrans',
            'ami': ami,
            'nmi': normalized_mutual_info_score(ground_truth, assignments),
            'runtime': runtime,
            'type': 'unsupervised_gnn',
            'n_detected': len(communities)
        })
        print(f"    GraphTrans: AMI={ami:.4f}, time={runtime:.3f}s")
    except Exception as e:
        print(f"    GraphTrans FAILED: {e}")
    
    return results


def main():
    print("=" * 70)
    print("TEST ALL ClusterNet GNNs (7 models)")
    print("=" * 70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test configurations: (n, num_communities, p_in, p_out)
    configs = [
        (300, 5, 0.25, 0.02, "Easy"),      # Clear structure
        (300, 5, 0.20, 0.05, "Medium"),    # Moderate
        (500, 8, 0.20, 0.03, "Large"),     # More nodes & communities
    ]
    
    all_results = []
    
    for n, num_comm, p_in, p_out, label in configs:
        print(f"\n{'='*70}")
        print(f"CONFIG: {label} - n={n}, k={num_comm}, p_in={p_in}, p_out={p_out}")
        print("=" * 70)
        
        # Generate network
        G, ground_truth = generate_sbm_network(n, num_comm, p_in, p_out)
        features = create_node_features(G, feature_dim=32)
        
        print(f"Generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {num_comm} communities")
        
        # Run classical algorithms
        print("\n--- Classical Algorithms ---")
        classical_results = run_classical_algorithms(G, ground_truth)
        for r in classical_results:
            r['config'] = label
        all_results.extend(classical_results)
        
        # Run ClusterNet GNNs
        print("\n--- ClusterNet GNNs (7 models) ---")
        gnn_results = run_clusternet_gnns_supervised(G, ground_truth, features, num_comm, epochs=200)
        for r in gnn_results:
            r['config'] = label
        all_results.extend(gnn_results)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Model':<15} | {'Type':<18} | {'Avg AMI':>8} | {'Std':>6}")
    print("-" * 55)
    
    model_amis = defaultdict(list)
    model_types = {}
    for r in all_results:
        model_amis[r['model']].append(r['ami'])
        model_types[r['model']] = r['type']
    
    # Sort by average AMI
    for model, amis in sorted(model_amis.items(), key=lambda x: -np.mean(x[1])):
        avg = np.mean(amis)
        std = np.std(amis)
        mtype = model_types[model]
        print(f"{model:<15} | {mtype:<18} | {avg:>8.4f} | {std:>6.4f}")
    
    print("\n" + "-" * 55)
    print("KEY FINDINGS:")
    print("  - Supervised GNNs (GCN_sup, GAT_sup) have access to labels → higher AMI")
    print("  - Unsupervised GNNs (DMoN, GIN, MinCut) use only structure → lower AMI")
    print("  - Classical algorithms (Louvain, Leiden) don't need labels → robust baselines")
    print("  - For fair comparison: compare classical vs unsupervised GNNs")
    print("  - For upper bound: use supervised GNNs with ground truth labels")


if __name__ == "__main__":
    main()



