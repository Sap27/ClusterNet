"""
ClusterNet GNN Benchmark Demo

Demonstrates the integrated GNN + benchmarking framework:
1. Synthetic benchmarks (LFR, SBM, Hierarchical)
2. Rewired real networks
3. Biological motif analysis
4. Robustness evaluation

Usage:
    python gnn_benchmark_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import networkx as nx
from typing import Dict, List

# Check for required dependencies
try:
    import torch
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. GNN methods will be skipped.")

# ClusterNet imports
from clusternet.benchmarks.synthetic.lfr_generator import LFRBenchmark, generate_lfr_suite
from clusternet.benchmarks.synthetic.sbm_generator import SBMBenchmark, HierarchicalSBM
from clusternet.benchmarks.rewired.rewiring import RewiredBenchmark
from clusternet.benchmarks.biological.motif_analysis import compute_motif_enrichment, biological_benchmark_score

# Traditional algorithms
from clusternet.algorithms import (
    louvain_algorithm,
    leiden_algorithm,
    walktrap_algorithm,
)


def evaluate_partition(
    communities: List[List[int]], 
    true_labels: np.ndarray,
    n_nodes: int
) -> Dict[str, float]:
    """Compute partition quality metrics."""
    pred_labels = np.zeros(n_nodes, dtype=int)
    for i, comm in enumerate(communities):
        for node in comm:
            if node < n_nodes:
                pred_labels[node] = i
    
    return {
        'nmi': normalized_mutual_info_score(true_labels, pred_labels),
        'ari': adjusted_rand_score(true_labels, pred_labels),
        'n_communities': len(communities)
    }


def demo_lfr_benchmark():
    """Demonstrate LFR benchmark generation and evaluation."""
    print("\n" + "="*60)
    print("DEMO 1: LFR Synthetic Benchmarks")
    print("="*60)
    
    # Generate LFR graphs with varying μ
    mu_values = [0.1, 0.3, 0.5]
    
    results = {}
    
    for mu in mu_values:
        print(f"\n--- LFR with μ={mu} ---")
        
        benchmark = LFRBenchmark(n=500, mu=mu, seed=42)
        G, communities, labels = benchmark.generate(feat_dim=32)
        
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"  Communities: {len(communities)}")
        
        # Run traditional algorithms
        for algo_name, algo_func in [
            ('Louvain', lambda g: louvain_algorithm(g)[0]),
            ('Leiden', lambda g: leiden_algorithm(g)[0]),
        ]:
            try:
                detected = algo_func(G)
                metrics = evaluate_partition(detected, labels, G.number_of_nodes())
                print(f"  {algo_name}: NMI={metrics['nmi']:.3f}, ARI={metrics['ari']:.3f}, k={metrics['n_communities']}")
                
                if mu not in results:
                    results[mu] = {}
                results[mu][algo_name] = metrics['nmi']
            except Exception as e:
                print(f"  {algo_name}: Error - {e}")
    
    print("\n--- Summary ---")
    print("NMI scores by μ and algorithm:")
    for mu in mu_values:
        scores = results.get(mu, {})
        print(f"  μ={mu}: " + ", ".join([f"{k}={v:.3f}" for k, v in scores.items()]))
    
    return results


def demo_sbm_benchmark():
    """Demonstrate SBM benchmark with varying SNR."""
    print("\n" + "="*60)
    print("DEMO 2: SBM Synthetic Benchmarks")
    print("="*60)
    
    snr_values = [1.0, 2.0, 4.0]
    
    for snr in snr_values:
        print(f"\n--- SBM with SNR={snr} ---")
        
        p_out = 0.05
        p_in = snr * np.sqrt(p_out) + p_out
        
        benchmark = SBMBenchmark(n=500, k=5, p_in=min(p_in, 1.0), p_out=p_out, seed=42)
        G, communities, labels = benchmark.generate()
        
        print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        print(f"  p_in={p_in:.3f}, p_out={p_out:.3f}")
        
        # Test with Louvain
        try:
            detected, _ = louvain_algorithm(G)
            metrics = evaluate_partition(detected, labels, G.number_of_nodes())
            print(f"  Louvain: NMI={metrics['nmi']:.3f}, ARI={metrics['ari']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")


def demo_hierarchical_sbm():
    """Demonstrate hierarchical community structure."""
    print("\n" + "="*60)
    print("DEMO 3: Hierarchical SBM")
    print("="*60)
    
    benchmark = HierarchicalSBM(
        n=500,
        levels=2,
        branching=3,
        p_in_base=0.5,
        p_decay=0.3,
        seed=42
    )
    G, communities, labels = benchmark.generate()
    
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Fine-grained communities: {len(communities)}")
    
    # Get communities at different levels
    for level in range(benchmark.levels + 1):
        comms = benchmark.get_communities_at_level(level)
        print(f"  Level {level}: {len(comms)} communities")


def demo_rewired_benchmark():
    """Demonstrate rewired real network benchmarks."""
    print("\n" + "="*60)
    print("DEMO 4: Rewired Real Networks")
    print("="*60)
    
    # Use Karate club as example
    G = nx.karate_club_graph()
    print(f"Original network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    benchmark = RewiredBenchmark(G, seed=42)
    results = benchmark.rewire_sweep(fractions=[0.0, 0.25, 0.5, 0.75])
    
    for frac, (G_rewired, communities) in results.items():
        topo_metrics = benchmark.compute_topology_preservation(G_rewired)
        
        # Run Louvain
        try:
            detected, _ = louvain_algorithm(G_rewired)
            true_labels = np.array([
                next(i for i, c in enumerate(communities) if node in c)
                for node in range(G_rewired.number_of_nodes())
            ])
            metrics = evaluate_partition(detected, true_labels, G_rewired.number_of_nodes())
            
            print(f"\n  Rewiring={int(frac*100)}%:")
            print(f"    Edge preservation: {topo_metrics['edge_preservation']:.3f}")
            print(f"    Degree correlation: {topo_metrics['degree_correlation']:.3f}")
            print(f"    Louvain NMI: {metrics['nmi']:.3f}")
        except Exception as e:
            print(f"  Error at {frac}: {e}")


def demo_motif_analysis():
    """Demonstrate biological motif enrichment analysis."""
    print("\n" + "="*60)
    print("DEMO 5: Biological Motif Enrichment")
    print("="*60)
    
    # Create a sample network with community structure
    G = nx.karate_club_graph()
    
    # Get communities
    detected, _ = louvain_algorithm(G)
    
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Detected {len(detected)} communities")
    
    # Compute motif enrichment
    motif_scores = compute_motif_enrichment(G, detected, n_random=50, seed=42)
    
    print("\nMotif Enrichment Results:")
    for key, value in motif_scores.items():
        print(f"  {key}: {value:.3f}")
    
    # Full biological score
    scores = biological_benchmark_score(G, detected)
    print(f"\nCombined biological score: {scores.get('combined', 0):.3f}")


def demo_gnn_clustering():
    """Demonstrate GNN-based community detection."""
    if not HAS_TORCH:
        print("\n" + "="*60)
        print("DEMO 6: GNN Clustering (SKIPPED - PyTorch not installed)")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("DEMO 6: GNN-Based Community Detection")
    print("="*60)
    
    try:
        from clusternet.gnn.models import (
            dmon_clustering,
            gin_clustering,
            mincut_clustering,
            graph_transformer_clustering
        )
        from clusternet.gnn.utils import generate_features
    except ImportError as e:
        print(f"GNN modules not available: {e}")
        return
    
    # Generate test graph
    benchmark = LFRBenchmark(n=200, mu=0.2, seed=42)
    G, communities, labels = benchmark.generate(feat_dim=32)
    features = benchmark.features
    
    print(f"Test graph: {G.number_of_nodes()} nodes, {len(communities)} communities")
    
    # Test each GNN method
    gnn_methods = [
        ('DMoN', dmon_clustering),
        ('GIN', gin_clustering),
        ('MinCut', mincut_clustering),
        ('GraphTransformer', graph_transformer_clustering),
    ]
    
    for name, method in gnn_methods:
        try:
            print(f"\n  Testing {name}...")
            detected, runtime = method(
                G,
                num_clusters=len(communities),
                features=features,
                epochs=50,  # Reduced for demo
                verbose=False
            )
            metrics = evaluate_partition(detected, labels, G.number_of_nodes())
            print(f"    NMI: {metrics['nmi']:.3f}, Runtime: {runtime:.2f}s")
        except Exception as e:
            print(f"    Error: {e}")


def demo_robustness():
    """Demonstrate robustness evaluation."""
    print("\n" + "="*60)
    print("DEMO 7: Robustness Evaluation")
    print("="*60)
    
    from clusternet.robustness import (
        perturb_edges_random,
        perturb_edges_targeted,
        perturb_features_mean
    )
    
    # Generate test graph
    benchmark = LFRBenchmark(n=300, mu=0.2, seed=42)
    G, communities, labels = benchmark.generate()
    
    print(f"Original graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Baseline
    detected, _ = louvain_algorithm(G)
    baseline_nmi = evaluate_partition(detected, labels, G.number_of_nodes())['nmi']
    print(f"Baseline NMI: {baseline_nmi:.3f}")
    
    # Test perturbations
    perturbation_levels = [0.1, 0.2, 0.3]
    
    print("\nRandom edge removal:")
    for level in perturbation_levels:
        G_pert = perturb_edges_random(G, remove_ratio=level, seed=42)
        detected, _ = louvain_algorithm(G_pert)
        nmi = evaluate_partition(detected, labels, G.number_of_nodes())['nmi']
        recovery = nmi / baseline_nmi if baseline_nmi > 0 else 0
        print(f"  {int(level*100)}% removed: NMI={nmi:.3f}, Recovery={recovery:.1%}")
    
    print("\nTargeted edge removal (betweenness):")
    for level in perturbation_levels:
        G_pert = perturb_edges_targeted(G, remove_ratio=level, seed=42)
        detected, _ = louvain_algorithm(G_pert)
        nmi = evaluate_partition(detected, labels, G.number_of_nodes())['nmi']
        recovery = nmi / baseline_nmi if baseline_nmi > 0 else 0
        print(f"  {int(level*100)}% removed: NMI={nmi:.3f}, Recovery={recovery:.1%}")


def main():
    """Run all demos."""
    print("="*60)
    print("ClusterNet GNN + Benchmark Integration Demo")
    print("="*60)
    
    # Run demos
    demo_lfr_benchmark()
    demo_sbm_benchmark()
    demo_hierarchical_sbm()
    demo_rewired_benchmark()
    demo_motif_analysis()
    demo_gnn_clustering()
    demo_robustness()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    main()

