"""
Weaken Benchmark: Progressive destruction of community structure

This benchmark tests algorithm robustness by progressively weakening
community structure through controlled edge rewiring:
- Q_decrease_1k: Weaken while preserving degree distribution
- Q_decrease_2k: Weaken while preserving degree correlation
- Q_decrease_3k: Weaken while preserving clustering coefficient

Key Research Questions:
    1. "At what point does each algorithm fail to detect communities?"
    2. "Which network properties does each algorithm rely on?"
    3. "Can we create a robustness fingerprint for each algorithm?"

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
import matplotlib.pyplot as plt

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


def Q_decrease_1k(G0: nx.Graph, node_community_list: List[List],
                  nswap: int = 1, max_tries: int = 100) -> nx.Graph:
    """
    Weaken community structure while preserving degree distribution (1k).
    
    Converts intra-community edges to inter-community edges,
    effectively increasing the mixing parameter μ.
    
    Properties preserved:
    - Degree sequence (each node keeps its degree)
    - Edge count
    
    Properties destroyed:
    - Community structure (modularity decreases)
    """
    if G0.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    if nswap > max_tries:
        raise nx.NetworkXError("nswap must be <= max_tries")
    if len(G0) < 4:
        raise nx.NetworkXError("Graph needs at least 4 nodes")
    
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
        
        # Old edges must be INTRA-community
        if not (edge_in_community(node_community_list, (u, v)) and 
                edge_in_community(node_community_list, (x, y))):
            continue
        
        # New edges must be INTER-community
        if (edge_in_community(node_community_list, (u, y)) or 
            edge_in_community(node_community_list, (v, x))):
            continue
        
        if y in G[u] or v in G[x]:
            continue
        
        G.remove_edge(u, v)
        G.remove_edge(x, y)
        G.add_edge(u, y)
        G.add_edge(v, x)
        
        swapcount += 1
    
    return G


def Q_decrease_2k(G0: nx.Graph, node_community_list: List[List],
                  nswap: int = 1, max_tries: int = 100) -> nx.Graph:
    """
    Weaken community structure while preserving degree correlation (2k).
    
    Like Q_decrease_1k but also preserves degree-degree correlation
    (assortativity pattern).
    
    Properties preserved:
    - Degree sequence
    - Degree correlation (assortativity)
    
    Properties destroyed:
    - Community structure
    """
    if G0.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    
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
        
        # Check intra/inter community constraints
        if not (edge_in_community(node_community_list, (u, v)) and 
                edge_in_community(node_community_list, (x, y))):
            continue
        
        if (edge_in_community(node_community_list, (u, y)) or 
            edge_in_community(node_community_list, (v, x))):
            continue
        
        # PRESERVE DEGREE CORRELATION: degrees of swapped endpoints must match
        if G.degree(v) != G.degree(y):
            continue
        
        if y in G[u] or v in G[x]:
            continue
        
        G.remove_edge(u, v)
        G.remove_edge(x, y)
        G.add_edge(u, y)
        G.add_edge(v, x)
        
        swapcount += 1
    
    return G


def Q_decrease_3k(G0: nx.Graph, node_community_list: List[List],
                  nswap: int = 1, max_tries: int = 100) -> nx.Graph:
    """
    Weaken community structure while preserving clustering coefficient (3k).
    
    The most restrictive variant - preserves local clustering.
    
    Properties preserved:
    - Degree sequence
    - Degree correlation
    - Local clustering coefficient
    
    Properties destroyed:
    - Community structure (but more slowly)
    """
    if G0.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    
    G0_original = G0  # Keep for CC comparison
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
        
        if not (edge_in_community(node_community_list, (u, v)) and 
                edge_in_community(node_community_list, (x, y))):
            continue
        
        if (edge_in_community(node_community_list, (u, y)) or 
            edge_in_community(node_community_list, (v, x))):
            continue
        
        if G.degree(v) != G.degree(y):
            continue
        
        if y in G[u] or v in G[x]:
            continue
        
        # Perform swap
        G.remove_edge(u, v)
        G.remove_edge(x, y)
        G.add_edge(u, y)
        G.add_edge(v, x)
        
        # CHECK CLUSTERING COEFFICIENT PRESERVATION
        affected_nodes = set([u, v, x, y])
        affected_nodes.update(G[u])
        affected_nodes.update(G[v])
        affected_nodes.update(G[x])
        affected_nodes.update(G[y])
        affected_nodes = list(affected_nodes)
        
        cc_old = nx.clustering(G0_original, nodes=affected_nodes)
        cc_new = nx.clustering(G, nodes=affected_nodes)
        
        if cc_old != cc_new:
            # Revert
            G.remove_edge(u, y)
            G.remove_edge(v, x)
            G.add_edge(u, v)
            G.add_edge(x, y)
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


def run_weaken_benchmark(
    G: nx.Graph,
    communities_gt: List[List],
    algorithms: Dict,
    preserve_orders: List[str] = ['1k', '2k', '3k'],
    weaken_levels: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    n_runs: int = 3,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run weakening benchmark.
    
    Progressively destroys community structure at different preservation levels.
    
    Args:
        G: Original graph
        communities_gt: Ground truth communities
        algorithms: Dict of {name: function}
        preserve_orders: Property preservation levels ('1k', '2k', '3k')
        weaken_levels: Fraction of edges to rewire (0.0 = original)
        n_runs: Number of runs per configuration
        seed: Random seed
        
    Returns:
        DataFrame with results
    """
    np.random.seed(seed)
    random.seed(seed)
    
    M = G.number_of_edges()
    results = []
    
    # Convert communities for compatibility
    communities_list = [[n for n in comm] for comm in communities_gt]
    labels_gt = communities_to_labels(communities_gt, G)
    
    original_mu = compute_mixing_parameter(G, communities_gt)
    original_Q = nx.community.modularity(G, communities_gt)
    original_assort = nx.degree_assortativity_coefficient(G)
    original_cc = nx.average_clustering(G)
    
    print(f"Original network: N={G.number_of_nodes()}, M={M}")
    print(f"Original μ={original_mu:.4f}, Q={original_Q:.4f}")
    print(f"Assortativity={original_assort:.4f}, Clustering={original_cc:.4f}")
    print("=" * 70)
    
    # Select weakening function based on preserve_order
    weaken_funcs = {
        '1k': Q_decrease_1k,
        '2k': Q_decrease_2k,
        '3k': Q_decrease_3k
    }
    
    for preserve_order in preserve_orders:
        weaken_func = weaken_funcs[preserve_order]
        
        for level in weaken_levels:
            nswap = int(M * level)
            max_tries = max(nswap * 20, 200)
            
            for run in range(n_runs):
                # Generate weakened network
                if level == 0.0:
                    G_weak = G.copy()
                else:
                    try:
                        G_weak = weaken_func(G, communities_list, nswap, max_tries)
                    except Exception as e:
                        print(f"  Weakening failed at {preserve_order}, level={level}: {e}")
                        continue
                
                # Compute network properties
                weak_mu = compute_mixing_parameter(G_weak, communities_gt)
                weak_assort = nx.degree_assortativity_coefficient(G_weak)
                weak_cc = nx.average_clustering(G_weak)
                
                try:
                    weak_Q = nx.community.modularity(G_weak, communities_gt)
                except:
                    weak_Q = 0.0
                
                # Run algorithms
                for algo_name, algo_func in algorithms.items():
                    try:
                        t1 = time.time()
                        communities_pred = algo_func(G_weak)
                        runtime = time.time() - t1
                        
                        labels_pred = communities_to_labels(communities_pred, G_weak)
                        
                        # Compute metrics
                        valid_mask = (labels_gt >= 0) & (labels_pred >= 0)
                        if valid_mask.sum() > 0:
                            ami = adjusted_mutual_info_score(labels_gt[valid_mask], labels_pred[valid_mask])
                            nmi = normalized_mutual_info_score(labels_gt[valid_mask], labels_pred[valid_mask])
                        else:
                            ami, nmi = 0.0, 0.0
                        
                        try:
                            Q_pred = nx.community.modularity(G_weak, communities_pred)
                        except:
                            Q_pred = 0.0
                        
                        results.append({
                            'preserve_order': preserve_order,
                            'weaken_level': level,
                            'run': run,
                            'algorithm': algo_name,
                            'ami': ami,
                            'nmi': nmi,
                            'modularity_gt': weak_Q,
                            'modularity_pred': Q_pred,
                            'n_communities': len(communities_pred),
                            'runtime': runtime,
                            'mu': weak_mu,
                            'assortativity': weak_assort,
                            'clustering': weak_cc,
                            'mu_original': original_mu,
                            'assort_original': original_assort,
                            'cc_original': original_cc
                        })
                        
                    except Exception as e:
                        print(f"  {algo_name} failed: {e}")
                        results.append({
                            'preserve_order': preserve_order,
                            'weaken_level': level,
                            'run': run,
                            'algorithm': algo_name,
                            'ami': np.nan,
                            'nmi': np.nan,
                            'modularity_gt': weak_Q,
                            'modularity_pred': np.nan,
                            'n_communities': np.nan,
                            'runtime': np.nan,
                            'mu': weak_mu,
                            'assortativity': weak_assort,
                            'clustering': weak_cc,
                            'mu_original': original_mu,
                            'assort_original': original_assort,
                            'cc_original': original_cc
                        })
            
            print(f"Completed: {preserve_order}, level={level:.2f}, μ={weak_mu:.4f}")
    
    return pd.DataFrame(results)


def analyze_weaken_results(df: pd.DataFrame) -> Dict:
    """
    Analyze weakening benchmark results.
    
    Computes:
    - Critical threshold (where AMI < 0.5) for each algorithm/preserve_order
    - Property dependence fingerprint
    - Robustness ranking
    """
    analysis = {
        'critical_thresholds': {},
        'property_fingerprints': {},
        'robustness_ranking': []
    }
    
    # Compute critical thresholds
    for algo in df['algorithm'].unique():
        analysis['critical_thresholds'][algo] = {}
        analysis['property_fingerprints'][algo] = {}
        
        for order in df['preserve_order'].unique():
            subset = df[(df['algorithm'] == algo) & (df['preserve_order'] == order)]
            grouped = subset.groupby('weaken_level')['ami'].mean()
            
            # Find critical threshold where AMI drops below 0.5
            critical = 1.0
            for level, ami in grouped.items():
                if ami < 0.5:
                    critical = level
                    break
            
            analysis['critical_thresholds'][algo][order] = critical
        
        # Property fingerprint: difference in critical thresholds
        thresholds = analysis['critical_thresholds'][algo]
        if '1k' in thresholds and '2k' in thresholds:
            analysis['property_fingerprints'][algo]['degree_corr_dependence'] = \
                thresholds.get('2k', 0) - thresholds.get('1k', 0)
        if '2k' in thresholds and '3k' in thresholds:
            analysis['property_fingerprints'][algo]['clustering_dependence'] = \
                thresholds.get('3k', 0) - thresholds.get('2k', 0)
    
    # Robustness ranking (average critical threshold)
    for algo, thresholds in analysis['critical_thresholds'].items():
        avg_threshold = np.mean(list(thresholds.values()))
        analysis['robustness_ranking'].append((algo, avg_threshold))
    
    analysis['robustness_ranking'].sort(key=lambda x: x[1], reverse=True)
    
    return analysis


def plot_weaken_results(df: pd.DataFrame, save_path: str = None):
    """Create visualization of weakening benchmark results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    algorithms = df['algorithm'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(algorithms)))
    
    for i, order in enumerate(['1k', '2k', '3k']):
        ax = axes[i]
        subset = df[df['preserve_order'] == order]
        
        for j, algo in enumerate(algorithms):
            algo_data = subset[subset['algorithm'] == algo]
            grouped = algo_data.groupby('weaken_level').agg({
                'ami': ['mean', 'std']
            }).reset_index()
            
            x = grouped['weaken_level']
            y = grouped[('ami', 'mean')]
            yerr = grouped[('ami', 'std')]
            
            ax.plot(x, y, '-o', label=algo, color=colors[j], markersize=4)
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=colors[j])
        
        ax.set_xlabel('Weakening Level')
        ax.set_ylabel('AMI')
        ax.set_title(f'{order} Preservation')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("WEAKEN BENCHMARK: Testing Algorithm Property Dependence")
    print("=" * 70)
    
    # Load Karate Club
    G = nx.karate_club_graph()
    
    # Ground truth communities
    communities_gt = [
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21],
        [8, 9, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    ]
    
    # Define algorithms
    algorithms = {}
    
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
    
    def label_prop_wrapper(G):
        return list(nx.community.label_propagation_communities(G))
    algorithms['LabelProp'] = label_prop_wrapper
    
    def greedy_modularity_wrapper(G):
        return list(nx.community.greedy_modularity_communities(G))
    algorithms['GreedyMod'] = greedy_modularity_wrapper
    
    # Run benchmark
    results_df = run_weaken_benchmark(
        G, communities_gt, algorithms,
        preserve_orders=['1k', '2k', '3k'],
        weaken_levels=[0.0, 0.2, 0.4, 0.6, 0.8],
        n_runs=3
    )
    
    # Save results
    results_df.to_csv('weaken_benchmark_results.csv', index=False)
    
    # Analyze
    analysis = analyze_weaken_results(results_df)
    
    print("\n" + "=" * 70)
    print("ANALYSIS: Critical Thresholds and Property Dependence")
    print("=" * 70)
    
    print("\nCritical Thresholds (where AMI < 0.5):")
    for algo, thresholds in analysis['critical_thresholds'].items():
        print(f"\n  {algo}:")
        for order, threshold in thresholds.items():
            print(f"    {order}: {threshold:.2f}")
    
    print("\nProperty Fingerprints:")
    for algo, fingerprint in analysis['property_fingerprints'].items():
        print(f"\n  {algo}:")
        for prop, value in fingerprint.items():
            interpretation = "relies on" if value > 0.1 else "doesn't need"
            print(f"    {prop}: {value:+.2f} ({interpretation})")
    
    print("\nRobustness Ranking:")
    for rank, (algo, score) in enumerate(analysis['robustness_ranking'], 1):
        print(f"  {rank}. {algo}: {score:.2f}")
    
    # Plot
    try:
        plot_weaken_results(results_df, 'weaken_benchmark_plot.png')
    except:
        print("Plotting skipped (matplotlib may not be available)")






