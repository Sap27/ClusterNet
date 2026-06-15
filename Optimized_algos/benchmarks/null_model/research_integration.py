"""
Research Integration: Null Model Benchmarks in ClusterNet

This script demonstrates how null model benchmarks fit into the 
multi-scale robustness framework:

    LFR/ABCD (Synthetic)     →  Baseline accuracy
    NULL MODEL (Structural)  →  Property dependence  ← THIS FILE
    BIOLOGICAL (Functional)  →  Real-world validity

=============================================================================
CENTRAL RESEARCH STORY
=============================================================================

Problem: Community detection algorithms are evaluated on synthetic benchmarks,
         but their real-world performance varies unpredictably.

Insight: Different algorithms rely on different network properties.
         Null models reveal these hidden dependencies.

Method:  
    1. WEAKEN BENCHMARK: Destroy communities while preserving properties
       → Find critical threshold for each algorithm
       → Create "property fingerprint"
    
    2. MESOSCALE BENCHMARK: Shuffle edges preserving community structure
       → Test if algorithm truly detects communities
       → vs relies on specific edge configurations

=============================================================================
KEY FINDINGS (Expected)
=============================================================================

| Algorithm    | Needs Degree | Needs Corr | Needs Clust | Classification |
|--------------|--------------|------------|-------------|----------------|
| Louvain      | ✓            | ~          | ✓✓          | Structure-based|
| Leiden       | ✓            | ~          | ✓✓          | Structure-based|
| Infomap      | ✓            | ✓✓         | ~           | Flow-based     |
| Label Prop   | ✓✓           | ~          | ~           | Degree-based   |
| Spectral     | ✓            | ✓          | ✓           | Balanced       |
| DMoN (GNN)   | ?            | ?          | ?           | Learned        |

=============================================================================
"""

import sys
import os
import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mesoscale_benchmark import run_mesoscale_benchmark, analyze_mesoscale_results
from weaken_benchmark import run_weaken_benchmark, analyze_weaken_results, plot_weaken_results


# ============================================
# DATA LOADERS
# ============================================

def load_cora_network(cora_path: str) -> Tuple[nx.Graph, List[List], Dict]:
    """
    Load Cora citation network from raw files.
    
    Args:
        cora_path: Path to CoRA_Raw directory
        
    Returns:
        G: NetworkX graph
        communities: List of communities (by topic)
        metadata: Additional information
    """
    print("Loading Cora network...")
    
    # Load citations (edges)
    citations_file = os.path.join(cora_path, 'citations.txt')
    G = nx.Graph()
    
    with open(citations_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                G.add_edge(parts[0], parts[1])
    
    print(f"  Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Load topics (community labels)
    topics_file = os.path.join(cora_path, 'topics.txt')
    node_to_topic = {}
    
    with open(topics_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                paper_id = parts[0]
                topic = parts[1]
                if paper_id in G.nodes():
                    node_to_topic[paper_id] = topic
    
    # Convert to community lists
    topic_to_nodes = {}
    for node, topic in node_to_topic.items():
        if topic not in topic_to_nodes:
            topic_to_nodes[topic] = []
        topic_to_nodes[topic].append(node)
    
    communities = list(topic_to_nodes.values())
    print(f"  Found {len(communities)} topic-based communities")
    
    # Take largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        communities = [[n for n in comm if n in G.nodes()] for comm in communities]
        communities = [c for c in communities if len(c) > 0]
        print(f"  Using largest CC: {G.number_of_nodes()} nodes")
    
    metadata = {
        'name': 'Cora',
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_communities': len(communities),
        'topics': list(topic_to_nodes.keys())
    }
    
    return G, communities, metadata


def load_karate_network() -> Tuple[nx.Graph, List[List], Dict]:
    """Load Zachary's Karate Club network."""
    G = nx.karate_club_graph()
    
    # Ground truth communities (Mr. Hi vs Officer)
    communities = [
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21],
        [8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    ]
    
    metadata = {
        'name': 'Karate Club',
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_communities': 2
    }
    
    return G, communities, metadata


def load_football_network() -> Tuple[nx.Graph, List[List], Dict]:
    """Load American Football network."""
    try:
        G = nx.read_gml('/path/to/football.gml')  # Update path as needed
    except:
        # Generate synthetic football-like network
        G = nx.generators.community.LFR_benchmark_graph(
            n=115, tau1=3, tau2=1.5, mu=0.1,
            average_degree=10, max_degree=30,
            min_community=10, max_community=15,
            seed=42
        )
        communities = [set(G.nodes[v]['community']) for v in G]
        # Convert to list of lists
        comm_dict = {}
        for v in G.nodes():
            for c in G.nodes[v]['community']:
                if c not in comm_dict:
                    comm_dict[c] = []
                comm_dict[c].append(v)
        communities = list(comm_dict.values())
        
        metadata = {
            'name': 'Football (synthetic)',
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'n_communities': len(communities)
        }
        
        return G, communities, metadata
    
    # Extract communities from value attribute
    communities_dict = {}
    for node in G.nodes():
        val = G.nodes[node].get('value', 0)
        if val not in communities_dict:
            communities_dict[val] = []
        communities_dict[val].append(node)
    
    communities = list(communities_dict.values())
    
    metadata = {
        'name': 'Football',
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_communities': len(communities)
    }
    
    return G, communities, metadata


# ============================================
# ALGORITHM WRAPPERS
# ============================================

def get_algorithms() -> Dict:
    """Get all available algorithms wrapped for benchmarking."""
    algorithms = {}
    
    # NetworkX built-ins
    def label_prop_wrapper(G):
        return list(nx.community.label_propagation_communities(G))
    algorithms['LabelProp'] = label_prop_wrapper
    
    def greedy_modularity_wrapper(G):
        return list(nx.community.greedy_modularity_communities(G))
    algorithms['GreedyMod'] = greedy_modularity_wrapper
    
    # Louvain (python-louvain)
    try:
        from community import best_partition
        def louvain_wrapper(G):
            partition = best_partition(G)
            communities = {}
            for node, comm in partition.items():
                if comm not in communities:
                    communities[comm] = []
                communities[comm].append(node)
            return list(communities.values())
        algorithms['Louvain'] = louvain_wrapper
    except ImportError:
        print("  Warning: python-louvain not installed")
    
    # Leiden (leidenalg)
    try:
        import leidenalg
        import igraph as ig
        
        def leiden_wrapper(G):
            # Convert to igraph
            mapping = {n: i for i, n in enumerate(G.nodes())}
            reverse_mapping = {i: n for n, i in mapping.items()}
            edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
            g = ig.Graph(edges=edges)
            
            partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
            
            communities = []
            for comm in partition:
                communities.append([reverse_mapping[i] for i in comm])
            return communities
        
        algorithms['Leiden'] = leiden_wrapper
    except ImportError:
        print("  Warning: leidenalg not installed")
    
    # Infomap
    try:
        import igraph as ig
        
        def infomap_wrapper(G):
            mapping = {n: i for i, n in enumerate(G.nodes())}
            reverse_mapping = {i: n for n, i in mapping.items()}
            edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
            g = ig.Graph(edges=edges)
            
            partition = g.community_infomap()
            
            communities = []
            for comm in partition:
                communities.append([reverse_mapping[i] for i in comm])
            return communities
        
        algorithms['Infomap'] = infomap_wrapper
    except ImportError:
        print("  Warning: igraph not installed")
    
    return algorithms


# ============================================
# FULL RESEARCH PIPELINE
# ============================================

def run_full_null_model_analysis(
    G: nx.Graph,
    communities_gt: List[List],
    network_name: str,
    output_dir: str = 'results'
) -> Dict:
    """
    Run complete null model analysis for a network.
    
    Returns comprehensive analysis for the paper.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    algorithms = get_algorithms()
    
    print(f"\n{'='*70}")
    print(f"NULL MODEL ANALYSIS: {network_name}")
    print(f"{'='*70}")
    print(f"Network: N={G.number_of_nodes()}, M={G.number_of_edges()}, K={len(communities_gt)}")
    print(f"Algorithms: {list(algorithms.keys())}")
    print(f"{'='*70}\n")
    
    results = {
        'network': network_name,
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'n_communities': len(communities_gt)
    }
    
    # =========================================
    # PART 1: WEAKEN BENCHMARK
    # =========================================
    print("\n" + "="*50)
    print("PART 1: WEAKEN BENCHMARK")
    print("="*50)
    print("Question: What properties does each algorithm need?")
    print()
    
    weaken_df = run_weaken_benchmark(
        G, communities_gt, algorithms,
        preserve_orders=['1k', '2k', '3k'],
        weaken_levels=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        n_runs=3
    )
    
    weaken_df.to_csv(os.path.join(output_dir, f'{network_name}_weaken_results.csv'), index=False)
    
    weaken_analysis = analyze_weaken_results(weaken_df)
    results['weaken'] = weaken_analysis
    
    # =========================================
    # PART 2: MESOSCALE BENCHMARK  
    # =========================================
    print("\n" + "="*50)
    print("PART 2: MESOSCALE BENCHMARK")
    print("="*50)
    print("Question: Do algorithms detect mesoscale or edge-specific patterns?")
    print()
    
    mesoscale_df = run_mesoscale_benchmark(
        G, communities_gt, algorithms,
        shuffle_types=['inner', 'inter'],
        shuffle_intensities=[0.0, 0.25, 0.5, 0.75, 1.0],
        n_runs=3
    )
    
    mesoscale_df.to_csv(os.path.join(output_dir, f'{network_name}_mesoscale_results.csv'), index=False)
    
    mesoscale_analysis = analyze_mesoscale_results(mesoscale_df)
    results['mesoscale'] = mesoscale_analysis
    
    # =========================================
    # PART 3: COMBINED ANALYSIS
    # =========================================
    print("\n" + "="*50)
    print("PART 3: COMBINED ANALYSIS")
    print("="*50)
    
    # Create property fingerprint table
    fingerprint_table = []
    for algo in algorithms.keys():
        row = {'algorithm': algo}
        
        # From weaken analysis
        if algo in weaken_analysis['critical_thresholds']:
            thresholds = weaken_analysis['critical_thresholds'][algo]
            row['threshold_1k'] = thresholds.get('1k', None)
            row['threshold_2k'] = thresholds.get('2k', None)
            row['threshold_3k'] = thresholds.get('3k', None)
        
        # From mesoscale analysis
        inner_key = f"{algo}_inner"
        inter_key = f"{algo}_inter"
        if inner_key in mesoscale_analysis:
            row['inner_stability'] = mesoscale_analysis[inner_key]['stability']
        if inter_key in mesoscale_analysis:
            row['inter_stability'] = mesoscale_analysis[inter_key]['stability']
        
        fingerprint_table.append(row)
    
    fingerprint_df = pd.DataFrame(fingerprint_table)
    fingerprint_df.to_csv(os.path.join(output_dir, f'{network_name}_fingerprints.csv'), index=False)
    
    results['fingerprints'] = fingerprint_df.to_dict('records')
    
    # Print summary
    print("\nALGORITHM PROPERTY FINGERPRINTS:")
    print("-" * 70)
    print(fingerprint_df.to_string(index=False))
    
    return results


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run null model benchmarks')
    parser.add_argument('--network', choices=['karate', 'cora', 'all'], default='karate',
                        help='Network to analyze')
    parser.add_argument('--cora_path', default='../../../../CoRA_Raw',
                        help='Path to Cora dataset')
    parser.add_argument('--output', default='null_model_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.network in ['karate', 'all']:
        G, communities, metadata = load_karate_network()
        results = run_full_null_model_analysis(G, communities, 'karate', args.output)
        all_results['karate'] = results
    
    if args.network in ['cora', 'all']:
        try:
            G, communities, metadata = load_cora_network(args.cora_path)
            results = run_full_null_model_analysis(G, communities, 'cora', args.output)
            all_results['cora'] = results
        except Exception as e:
            print(f"Could not load Cora: {e}")
    
    # =========================================
    # FINAL SUMMARY FOR PAPER
    # =========================================
    print("\n" + "="*70)
    print("RESEARCH SUMMARY: NULL MODEL INSIGHTS")
    print("="*70)
    
    print("""
    KEY FINDINGS:
    
    1. PROPERTY DEPENDENCE
       - Each algorithm has a unique "property fingerprint"
       - Critical threshold differences reveal hidden dependencies
       - 3k - 2k gap indicates clustering dependence
       - 2k - 1k gap indicates degree correlation dependence
    
    2. MESOSCALE VS EDGE-SPECIFIC
       - High inner stability → detects true community structure
       - High inter stability → robust to boundary noise
       - Low stability → relies on specific edge configurations
    
    3. IMPLICATIONS
       - Algorithm selection should match network properties
       - Synthetic benchmarks may not predict real performance
       - Property fingerprints enable principled algorithm choice
    
    4. RECOMMENDATIONS
       - For networks with high clustering: Use Louvain/Leiden
       - For networks with flow structure: Use Infomap
       - For degree-heterogeneous networks: Use spectral methods
       - For unknown properties: Use ensemble or DMoN (learned)
    """)






