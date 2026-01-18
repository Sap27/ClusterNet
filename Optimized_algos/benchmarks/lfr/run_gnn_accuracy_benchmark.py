#!/usr/bin/env python3
"""
GNN Accuracy Benchmark for LFR Networks

Tests GNN-based community detection algorithms on LFR benchmark networks
with varying mixing parameter (μ).

Key findings:
- Semi-supervised GNNs with learnable embeddings achieve near-perfect accuracy
- Unsupervised GNNs with modularity loss achieve ~0.6-0.7 AMI
- Spectral features alone fail on LFR (power-law degree distribution confounds)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algorithm_runner import AlgorithmRunner, GNNRunner


def load_lfr_network(network_dir: Path, mu: float, realization: int):
    """Load LFR network and ground truth."""
    mu_str = str(mu).replace('.', '')
    edge_file = network_dir / f"mu{mu}_r{realization}_edges.txt"
    comm_file = network_dir / f"mu{mu}_r{realization}_communities.txt"
    
    if not edge_file.exists():
        return None, None, None
    
    G = nx.read_edgelist(str(edge_file), nodetype=int)
    
    gt_labels = {}
    gt_communities = []
    with open(comm_file) as f:
        for i, line in enumerate(f):
            parts = line.strip().split(':')
            if len(parts) == 2:
                nodes = list(map(int, parts[1].strip().split()))
                gt_communities.append(nodes)
                for node in nodes:
                    gt_labels[node] = i
    
    return G, gt_labels, gt_communities


def calculate_ami(result, nodes, gt_labels):
    """Calculate AMI from algorithm result."""
    if not result.success or not result.communities:
        return None
    
    labels = {node: -1 for node in nodes}
    for i, comm in enumerate(result.communities):
        for node in comm:
            if node in labels:
                labels[node] = i
    
    if any(labels[n] < 0 for n in nodes):
        return None
    
    gt = [gt_labels[n] for n in nodes]
    pred = [labels[n] for n in nodes]
    return adjusted_mutual_info_score(gt, pred)


def run_gnn_benchmark(
    network_dir: Path,
    output_dir: Path,
    mu_values: list = [0.1, 0.2, 0.3, 0.4, 0.5],
    realizations: int = 2,
    labeled_ratios: list = [0.10, 0.05, 0.01]
):
    """Run GNN accuracy benchmark."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gnn_runner = GNNRunner(device='cpu', verbose=True)
    classical_runner = AlgorithmRunner(verbose=False)
    
    results = []
    
    for mu in mu_values:
        for r in range(realizations):
            G, gt_labels, gt_communities = load_lfr_network(network_dir, mu, r)
            
            if G is None:
                print(f"Network mu={mu}, r={r} not found, skipping")
                continue
            
            nodes = sorted(G.nodes())
            n_communities = len(gt_communities)
            
            print(f"\n{'='*60}")
            print(f"μ = {mu}, realization {r}: {G.number_of_nodes()} nodes, {n_communities} communities")
            print(f"{'='*60}")
            
            # Classical baseline (Leiden)
            result = classical_runner.run_algorithm(G, 'leiden')
            ami = calculate_ami(result, nodes, gt_labels)
            results.append({
                'mu': mu, 'realization': r, 'algorithm': 'leiden',
                'ami': ami, 'n_communities': len(result.communities) if result.success else 0,
                'runtime': result.runtime, 'type': 'classical'
            })
            print(f"  Leiden:                         AMI={ami:.4f}" if ami else "  Leiden: FAILED")
            
            # GNN Unsupervised (learnable embeddings)
            print("\n  Training GNN Unsupervised...")
            result = gnn_runner.run_gnn_unsupervised_learnable(
                G, num_communities=n_communities, epochs=300
            )
            ami = calculate_ami(result, nodes, gt_labels)
            results.append({
                'mu': mu, 'realization': r, 'algorithm': 'gnn_unsupervised',
                'ami': ami, 'n_communities': len(result.communities) if result.success else 0,
                'runtime': result.runtime, 'type': 'gnn_unsupervised'
            })
            print(f"  GNN Unsupervised:               AMI={ami:.4f}, {len(result.communities)} communities" if ami else "  GNN Unsupervised: FAILED")
            
            # GNN Semi-supervised with different labeled ratios
            for ratio in labeled_ratios:
                print(f"\n  Training GNN Semi-supervised ({int(ratio*100)}% labels)...")
                result = gnn_runner.run_gnn_semisupervised(
                    G, num_communities=n_communities,
                    ground_truth=gt_communities,
                    labeled_ratio=ratio,
                    epochs=200
                )
                ami = calculate_ami(result, nodes, gt_labels)
                results.append({
                    'mu': mu, 'realization': r, 
                    'algorithm': f'gnn_semisup_{int(ratio*100)}pct',
                    'ami': ami, 
                    'n_communities': len(result.communities) if result.success else 0,
                    'runtime': result.runtime, 
                    'type': f'gnn_semisup_{int(ratio*100)}pct',
                    'labeled_ratio': ratio
                })
                print(f"  GNN Semi-sup ({int(ratio*100)}% lbl):           AMI={ami:.4f}" if ami else f"  GNN Semi-sup {int(ratio*100)}%: FAILED")
            
            # DMoN with spectral features (for comparison)
            print("\n  Training DMoN (spectral)...")
            result = gnn_runner.run_dmon(G, num_communities=n_communities, epochs=300)
            ami = calculate_ami(result, nodes, gt_labels)
            results.append({
                'mu': mu, 'realization': r, 'algorithm': 'dmon_spectral',
                'ami': ami, 'n_communities': len(result.communities) if result.success else 0,
                'runtime': result.runtime, 'type': 'gnn_spectral'
            })
            print(f"  DMoN (spectral):                AMI={ami:.4f}, {len(result.communities)} communities" if ami else "  DMoN: FAILED")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "gnn_accuracy_results.csv", index=False)
    
    # Create summary
    summary = df.groupby(['mu', 'algorithm']).agg({
        'ami': ['mean', 'std'],
        'n_communities': 'mean',
        'runtime': 'mean'
    }).round(4)
    summary.columns = ['ami_mean', 'ami_std', 'n_communities', 'runtime']
    summary = summary.reset_index()
    summary.to_csv(output_dir / "gnn_accuracy_summary.csv", index=False)
    
    print("\n" + "="*60)
    print("GNN ACCURACY BENCHMARK SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    print(f"\nResults saved to {output_dir}")
    
    return df, summary


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    network_dir = script_dir / "accuracy" / "networks"
    output_dir = script_dir / "accuracy" / "gnn_results"
    
    # Run benchmark
    df, summary = run_gnn_benchmark(
        network_dir=network_dir,
        output_dir=output_dir,
        mu_values=[0.1, 0.3, 0.5],  # Focus on key mu values
        realizations=2,
        labeled_ratios=[0.10, 0.05, 0.01]
    )
