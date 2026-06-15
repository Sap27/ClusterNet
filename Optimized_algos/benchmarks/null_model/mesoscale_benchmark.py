"""
Mesoscale Benchmark: Shuffle edges while preserving community structure

This benchmark tests whether algorithms can detect communities when:
- Intra-community edges are randomly shuffled (inner_random_*k)
- Inter-community edges are randomly shuffled (inter_random_*k)
- Network properties are preserved at different orders (0k, 1k, 2k, 3k)

Key Research Question:
    "Do algorithms detect communities based on mesoscale structure,
     or do they rely on specific edge configurations?"

If algorithm works after shuffling → Relies on mesoscale structure
If algorithm fails after shuffling → Relies on specific edge patterns

Based on: Community-nullmodel project (https://github.com/xxx/Community-nullmodel)
"""

import sys
import os
import copy
import random
import numpy as np
import networkx as nx
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
import time

# Add ClusterNet to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


# ============================================
# NULL MODEL FUNCTIONS (from Community-nullmodel)
# ============================================

def edge_in_community(node_community_list: List[List], edge: Tuple) -> bool:
    """Check if edge is within a community."""
    for community in node_community_list:
        if edge[0] in community and edge[1] in community:
            return True
    return False


def inner_random_1k(G0: nx.Graph, node_community_list: List[List], 
                    nswap: int = 1, max_tries: int = 100, 
                    connected: bool = True) -> nx.Graph:
    """
    Shuffle intra-community edges while preserving degree distribution (1k).
    
    This randomly rewires edges WITHIN communities without changing:
    - Community membership
    - Degree sequence
    - Number of intra/inter community edges
    """
    if G0.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    if nswap > max_tries:
        raise nx.NetworkXError("nswap must be <= max_tries")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph needs at least 3 nodes")
    
    G = copy.deepcopy(G0)
    keys, degrees = list(zip(*list(G.degree())))
    cdf = nx.utils.cumulative_distribution(degrees)
    
    tn = 0
    swapcount = 0
    
    while swapcount < nswap:
        if tn >= max_tries:
            break
        tn += 1
        
        # Select two edges preserving degree distribution
        (ui, xi) = nx.utils.discrete_sequence(2, cdistribution=cdf)
        if ui == xi:
            continue
        
        u = keys[ui]
        x = keys[xi]
        
        neighbors_u = list(G[u])
        neighbors_x = list(G[x])
        
        if not neighbors_u or not neighbors_x:
            continue
            
        v = random.choice(neighbors_u)
        y = random.choice(neighbors_x)
        
        if len(set([u, v, x, y])) != 4:
            continue
        
        # Both old AND new edges must be intra-community
        if not (edge_in_community(node_community_list, (u, v)) and 
                edge_in_community(node_community_list, (x, y)) and
                edge_in_community(node_community_list, (u, y)) and
                edge_in_community(node_community_list, (v, x))):
            continue
        
        if y in G[u] or v in G[x]:
            continue
        
        G.add_edge(u, y)
        G.add_edge(v, x)
        G.remove_edge(u, v)
        G.remove_edge(x, y)
        
        if connected and not nx.is_connected(G):
            # Revert
            G.add_edge(u, v)
            G.add_edge(x, y)
            G.remove_edge(u, y)
            G.remove_edge(v, x)
            continue
            
        swapcount += 1
    
    return G


def inter_random_1k(G0: nx.Graph, node_community_list: List[List],
                    nswap: int = 1, max_tries: int = 100,
                    connected: bool = True) -> nx.Graph:
    """
    Shuffle inter-community edges while preserving degree distribution (1k).
    
    This randomly rewires edges BETWEEN communities without changing:
    - Community membership
    - Degree sequence
    - Number of intra/inter community edges
    """
    if G0.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    if nswap > max_tries:
        raise nx.NetworkXError("nswap must be <= max_tries")
    if len(G0) < 3:
        raise nx.NetworkXError("Graph needs at least 3 nodes")
    
    G = copy.deepcopy(G0)
    keys, degrees = list(zip(*list(G.degree())))
    cdf = nx.utils.cumulative_distribution(degrees)
    
    tn = 0
    swapcount = 0
    
    while swapcount < nswap:
        if tn >= max_tries:
            break
        tn += 1
        
        (ui, xi) = nx.utils.discrete_sequence(2, cdistribution=cdf)
        if ui == xi:
            continue
        
        u = keys[ui]
        x = keys[xi]
        
        neighbors_u = list(G[u])
        neighbors_x = list(G[x])
        
        if not neighbors_u or not neighbors_x:
            continue
            
        v = random.choice(neighbors_u)
        y = random.choice(neighbors_x)
        
        if len(set([u, v, x, y])) != 4:
            continue
        
        # Both old AND new edges must be INTER-community
        if (edge_in_community(node_community_list, (u, v)) or 
            edge_in_community(node_community_list, (x, y)) or
            edge_in_community(node_community_list, (u, y)) or
            edge_in_community(node_community_list, (v, x))):
            continue
        
        if y in G[u] or v in G[x]:
            continue
        
        G.add_edge(u, y)
        G.add_edge(v, x)
        G.remove_edge(u, v)
        G.remove_edge(x, y)
        
        if connected and not nx.is_connected(G):
            G.add_edge(u, v)
            G.add_edge(x, y)
            G.remove_edge(u, y)
            G.remove_edge(v, x)
            continue
            
        swapcount += 1
    
    return G


# ============================================
# BENCHMARK FUNCTIONS
# ============================================

def compute_mixing_parameter(G: nx.Graph, communities: List[List]) -> float:
    """Compute mixing parameter μ = fraction of inter-community edges."""
    inter_edges = 0
    total_edges = G.number_of_edges()
    
    for u, v in G.edges():
        if not edge_in_community(communities, (u, v)):
            inter_edges += 1
    
    return inter_edges / total_edges if total_edges > 0 else 0.0


def communities_to_labels(communities: List[List], G: nx.Graph) -> np.ndarray:
    """Convert community list to label array."""
    node_to_label = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_to_label[node] = i
    
    labels = []
    for node in G.nodes():
        labels.append(node_to_label.get(node, -1))
    return np.array(labels)


def run_mesoscale_benchmark(
    G: nx.Graph,
    communities_gt: List[List],
    algorithms: Dict,
    shuffle_types: List[str] = ['inner', 'inter', 'both'],
    shuffle_intensities: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    n_runs: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run mesoscale preservation benchmark.
    
    Tests if algorithms can detect communities after edge shuffling
    that preserves mesoscale structure.
    
    Args:
        G: Original graph
        communities_gt: Ground truth communities
        algorithms: Dict of {name: function}
        shuffle_types: Types of shuffling ('inner', 'inter', 'both')
        shuffle_intensities: Fraction of edges to shuffle (0.0 = original)
        n_runs: Number of runs per configuration
        seed: Random seed
        
    Returns:
        DataFrame with results
    """
    np.random.seed(seed)
    random.seed(seed)
    
    M = G.number_of_edges()
    results = []
    
    # Convert communities to string nodes if needed (for networkx compatibility)
    communities_str = []
    for comm in communities_gt:
        communities_str.append([str(n) if isinstance(list(G.nodes())[0], str) else n for n in comm])
    
    labels_gt = communities_to_labels(communities_gt, G)
    original_mu = compute_mixing_parameter(G, communities_gt)
    original_Q = nx.community.modularity(G, communities_gt)
    
    print(f"Original network: N={G.number_of_nodes()}, M={M}")
    print(f"Original μ={original_mu:.4f}, Q={original_Q:.4f}")
    print(f"Communities: {len(communities_gt)}")
    print("=" * 60)
    
    for shuffle_type in shuffle_types:
        for intensity in shuffle_intensities:
            nswap = int(M * intensity)
            max_tries = max(nswap * 10, 100)
            
            for run in range(n_runs):
                # Generate shuffled network
                if intensity == 0.0:
                    G_shuffled = G.copy()
                else:
                    if shuffle_type == 'inner':
                        G_shuffled = inner_random_1k(G, communities_str, nswap, max_tries)
                    elif shuffle_type == 'inter':
                        G_shuffled = inter_random_1k(G, communities_str, nswap, max_tries)
                    else:  # both
                        G_temp = inner_random_1k(G, communities_str, nswap // 2, max_tries // 2)
                        G_shuffled = inter_random_1k(G_temp, communities_str, nswap // 2, max_tries // 2)
                
                # Verify properties preserved
                shuffled_mu = compute_mixing_parameter(G_shuffled, communities_gt)
                
                # Run algorithms
                for algo_name, algo_func in algorithms.items():
                    try:
                        t1 = time.time()
                        communities_pred = algo_func(G_shuffled)
                        runtime = time.time() - t1
                        
                        labels_pred = communities_to_labels(communities_pred, G_shuffled)
                        
                        # Handle label mismatches
                        valid_mask = (labels_gt >= 0) & (labels_pred >= 0)
                        if valid_mask.sum() > 0:
                            ami = adjusted_mutual_info_score(labels_gt[valid_mask], labels_pred[valid_mask])
                            nmi = normalized_mutual_info_score(labels_gt[valid_mask], labels_pred[valid_mask])
                        else:
                            ami, nmi = 0.0, 0.0
                        
                        try:
                            Q_pred = nx.community.modularity(G_shuffled, communities_pred)
                        except:
                            Q_pred = 0.0
                        
                        results.append({
                            'shuffle_type': shuffle_type,
                            'intensity': intensity,
                            'run': run,
                            'algorithm': algo_name,
                            'ami': ami,
                            'nmi': nmi,
                            'modularity': Q_pred,
                            'n_communities': len(communities_pred),
                            'runtime': runtime,
                            'mu_original': original_mu,
                            'mu_shuffled': shuffled_mu
                        })
                        
                    except Exception as e:
                        print(f"  {algo_name} failed: {e}")
                        results.append({
                            'shuffle_type': shuffle_type,
                            'intensity': intensity,
                            'run': run,
                            'algorithm': algo_name,
                            'ami': np.nan,
                            'nmi': np.nan,
                            'modularity': np.nan,
                            'n_communities': np.nan,
                            'runtime': np.nan,
                            'mu_original': original_mu,
                            'mu_shuffled': shuffled_mu
                        })
            
            print(f"Completed: {shuffle_type}, intensity={intensity:.2f}")
    
    return pd.DataFrame(results)


def analyze_mesoscale_results(df: pd.DataFrame) -> Dict:
    """
    Analyze mesoscale benchmark results.
    
    Returns insights about which algorithms rely on mesoscale vs edge-specific structure.
    """
    analysis = {}
    
    # Group by algorithm and shuffle type
    grouped = df.groupby(['algorithm', 'shuffle_type'])
    
    # Compute stability scores
    for (algo, shuffle_type), group in grouped:
        # Compare performance at intensity=0 vs intensity=1
        baseline = group[group['intensity'] == 0.0]['ami'].mean()
        shuffled = group[group['intensity'] == 1.0]['ami'].mean()
        
        stability = shuffled / baseline if baseline > 0 else 0.0
        
        key = f"{algo}_{shuffle_type}"
        analysis[key] = {
            'baseline_ami': baseline,
            'shuffled_ami': shuffled,
            'stability': stability,
            'interpretation': 'Mesoscale-based' if stability > 0.8 else 'Edge-specific'
        }
    
    return analysis


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("MESOSCALE BENCHMARK: Testing Community Detection Robustness")
    print("=" * 70)
    
    # Load Karate Club as example
    G = nx.karate_club_graph()
    
    # Ground truth communities
    communities_gt = [
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21],
        [8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    ]
    
    # Define algorithms to test
    algorithms = {}
    
    # Louvain
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
    except:
        pass
    
    # Label Propagation
    def label_prop_wrapper(G):
        return list(nx.community.label_propagation_communities(G))
    algorithms['LabelProp'] = label_prop_wrapper
    
    # Greedy Modularity
    def greedy_modularity_wrapper(G):
        return list(nx.community.greedy_modularity_communities(G))
    algorithms['GreedyMod'] = greedy_modularity_wrapper
    
    # Run benchmark
    results_df = run_mesoscale_benchmark(
        G, communities_gt, algorithms,
        shuffle_types=['inner', 'inter'],
        shuffle_intensities=[0.0, 0.5, 1.0],
        n_runs=3
    )
    
    # Save results
    results_df.to_csv('mesoscale_benchmark_results.csv', index=False)
    
    # Analyze
    analysis = analyze_mesoscale_results(results_df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS: Algorithm Stability Under Mesoscale Shuffling")
    print("=" * 70)
    
    for key, values in analysis.items():
        print(f"\n{key}:")
        print(f"  Baseline AMI: {values['baseline_ami']:.4f}")
        print(f"  Shuffled AMI: {values['shuffled_ami']:.4f}")
        print(f"  Stability: {values['stability']:.4f}")
        print(f"  Interpretation: {values['interpretation']}")






