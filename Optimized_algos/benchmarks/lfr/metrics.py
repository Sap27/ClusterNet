"""
Evaluation Metrics for Community Detection
==========================================

Comprehensive metrics for both disjoint and overlapping community detection.
"""

import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter


def communities_to_labels(communities: List[List[int]], num_nodes: int) -> np.ndarray:
    """
    Convert community list to label array.
    For overlapping communities, assigns node to first community.
    
    Args:
        communities: List of communities (each is list of node ids)
        num_nodes: Total number of nodes
        
    Returns:
        Array of community labels (length = num_nodes)
    """
    labels = np.full(num_nodes, -1, dtype=int)
    for comm_id, comm in enumerate(communities):
        for node in comm:
            if 0 <= node < num_nodes and labels[node] == -1:
                labels[node] = comm_id
    
    # Assign uncovered nodes to singleton communities
    next_comm = len(communities)
    for node in range(num_nodes):
        if labels[node] == -1:
            labels[node] = next_comm
            next_comm += 1
    
    return labels


def compute_ami(
    true_communities: List[List[int]],
    pred_communities: List[List[int]],
    num_nodes: int
) -> float:
    """
    Compute Adjusted Mutual Information between two partitions.
    
    Args:
        true_communities: Ground truth communities
        pred_communities: Predicted communities
        num_nodes: Number of nodes in the graph
        
    Returns:
        AMI score in [-1, 1], 1 = perfect match
    """
    true_labels = communities_to_labels(true_communities, num_nodes)
    pred_labels = communities_to_labels(pred_communities, num_nodes)
    return adjusted_mutual_info_score(true_labels, pred_labels)


def compute_nmi(
    true_communities: List[List[int]],
    pred_communities: List[List[int]],
    num_nodes: int
) -> float:
    """Compute Normalized Mutual Information."""
    true_labels = communities_to_labels(true_communities, num_nodes)
    pred_labels = communities_to_labels(pred_communities, num_nodes)
    return normalized_mutual_info_score(true_labels, pred_labels)


def compute_ari(
    true_communities: List[List[int]],
    pred_communities: List[List[int]],
    num_nodes: int
) -> float:
    """Compute Adjusted Rand Index."""
    true_labels = communities_to_labels(true_communities, num_nodes)
    pred_labels = communities_to_labels(pred_communities, num_nodes)
    return adjusted_rand_score(true_labels, pred_labels)


def compute_modularity(
    G: nx.Graph,
    communities: List[List[int]]
) -> float:
    """
    Compute modularity of a partition.
    
    Handles partial coverage by adding singletons for uncovered nodes.
    """
    if not communities:
        return 0.0
    
    # Convert to sets and ensure integer node ids
    mod_communities = []
    covered_nodes = set()
    
    for comm in communities:
        comm_set = set()
        for node in comm:
            node_int = int(node)
            if node_int in G.nodes():
                comm_set.add(node_int)
                covered_nodes.add(node_int)
        if comm_set:
            mod_communities.append(comm_set)
    
    # Add uncovered nodes as singletons
    for node in G.nodes():
        if node not in covered_nodes:
            mod_communities.append({node})
    
    if not mod_communities:
        return 0.0
    
    try:
        return nx.community.modularity(G, mod_communities)
    except Exception as e:
        print(f"Modularity calculation failed: {e}")
        return 0.0


def compute_coverage(
    G: nx.Graph,
    communities: List[List[int]]
) -> float:
    """
    Compute coverage: fraction of intra-community edges.
    """
    if not communities or G.number_of_edges() == 0:
        return 0.0
    
    # Create node to community mapping
    node_to_comm = {}
    for i, comm in enumerate(communities):
        for node in comm:
            if node not in node_to_comm:
                node_to_comm[node] = i
    
    intra_edges = 0
    for u, v in G.edges():
        if u in node_to_comm and v in node_to_comm:
            if node_to_comm[u] == node_to_comm[v]:
                intra_edges += 1
    
    return intra_edges / G.number_of_edges()


def compute_conductance(
    G: nx.Graph,
    communities: List[List[int]]
) -> float:
    """
    Compute average conductance across communities.
    Lower is better (fewer edges leaving community relative to size).
    """
    if not communities:
        return 1.0
    
    conductances = []
    for comm in communities:
        comm_set = set(comm)
        if len(comm_set) == 0:
            continue
        
        # Count internal and external edges
        internal = 0
        external = 0
        total_degree = 0
        
        for node in comm_set:
            if node not in G:
                continue
            for neighbor in G.neighbors(node):
                total_degree += 1
                if neighbor in comm_set:
                    internal += 1
                else:
                    external += 1
        
        # Conductance = external / min(total_degree, 2*m - total_degree)
        if total_degree == 0:
            continue
        
        m = G.number_of_edges()
        denom = min(total_degree, 2 * m - total_degree)
        if denom > 0:
            conductances.append(external / denom)
    
    return np.mean(conductances) if conductances else 1.0


# ============= OVERLAPPING METRICS =============

def compute_onmi(
    true_communities: List[List[int]],
    pred_communities: List[List[int]],
    num_nodes: int
) -> float:
    """
    Compute Overlapping NMI (based on Lancichinetti et al. 2009).
    
    Simplified version using coverage-based similarity.
    """
    def community_entropy(comms, n):
        """Compute entropy of community membership."""
        membership_counts = Counter()
        for comm in comms:
            for node in comm:
                membership_counts[node] += 1
        
        # Probability of being in a community
        probs = []
        for node in range(n):
            p = membership_counts.get(node, 0) / max(1, len(comms))
            if 0 < p < 1:
                probs.append(-p * np.log2(p) - (1-p) * np.log2(1-p))
            else:
                probs.append(0)
        return np.mean(probs) if probs else 0
    
    def conditional_entropy(comms1, comms2, n):
        """Compute conditional entropy H(X|Y)."""
        # For each community in comms1, find best matching in comms2
        total_entropy = 0
        for c1 in comms1:
            c1_set = set(c1)
            best_match = 0
            for c2 in comms2:
                c2_set = set(c2)
                overlap = len(c1_set & c2_set)
                match = overlap / max(1, len(c1_set | c2_set))
                best_match = max(best_match, match)
            total_entropy += (1 - best_match)
        return total_entropy / max(1, len(comms1))
    
    h_true = community_entropy(true_communities, num_nodes)
    h_pred = community_entropy(pred_communities, num_nodes)
    
    if h_true == 0 and h_pred == 0:
        return 1.0
    
    h_true_given_pred = conditional_entropy(true_communities, pred_communities, num_nodes)
    h_pred_given_true = conditional_entropy(pred_communities, true_communities, num_nodes)
    
    # Normalized mutual information
    if h_true + h_pred > 0:
        nmi = 1 - (h_true_given_pred + h_pred_given_true) / (h_true + h_pred + 1e-10)
        return max(0, min(1, nmi))
    return 0.0


def compute_omega_index(
    true_communities: List[List[int]],
    pred_communities: List[List[int]],
    num_nodes: int
) -> float:
    """
    Compute Omega Index for overlapping communities.
    
    Measures agreement on whether pairs of nodes are in same number of communities.
    """
    def count_shared_communities(comms, n):
        """Count how many communities each pair shares."""
        pair_counts = {}
        for node in range(n):
            pair_counts[(node, node)] = 0
        
        for comm in comms:
            for i, u in enumerate(comm):
                for v in comm[i+1:]:
                    key = tuple(sorted([u, v]))
                    pair_counts[key] = pair_counts.get(key, 0) + 1
        
        return pair_counts
    
    true_pairs = count_shared_communities(true_communities, num_nodes)
    pred_pairs = count_shared_communities(pred_communities, num_nodes)
    
    # Count agreements
    agreements = 0
    total_pairs = 0
    
    all_pairs = set(true_pairs.keys()) | set(pred_pairs.keys())
    for pair in all_pairs:
        if len(pair) == 2:
            if true_pairs.get(pair, 0) == pred_pairs.get(pair, 0):
                agreements += 1
            total_pairs += 1
    
    return agreements / max(1, total_pairs)


def compute_f1_communities(
    true_communities: List[List[int]],
    pred_communities: List[List[int]]
) -> float:
    """
    Compute average F1 score between matched communities.
    """
    if not true_communities or not pred_communities:
        return 0.0
    
    true_sets = [set(c) for c in true_communities]
    pred_sets = [set(c) for c in pred_communities]
    
    f1_scores = []
    
    for true_set in true_sets:
        best_f1 = 0
        for pred_set in pred_sets:
            if len(true_set) == 0 and len(pred_set) == 0:
                f1 = 1.0
            elif len(true_set) == 0 or len(pred_set) == 0:
                f1 = 0.0
            else:
                precision = len(true_set & pred_set) / len(pred_set)
                recall = len(true_set & pred_set) / len(true_set)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                else:
                    f1 = 0.0
            best_f1 = max(best_f1, f1)
        f1_scores.append(best_f1)
    
    return np.mean(f1_scores)


def compute_all_metrics(
    G: nx.Graph,
    true_communities: List[List[int]],
    pred_communities: List[List[int]],
    overlapping: bool = False
) -> Dict[str, float]:
    """
    Compute all relevant metrics.
    
    Args:
        G: The graph
        true_communities: Ground truth
        pred_communities: Predicted communities
        overlapping: Whether to compute overlapping-specific metrics
        
    Returns:
        Dictionary of metric name -> value
    """
    num_nodes = G.number_of_nodes()
    
    metrics = {
        'ami': compute_ami(true_communities, pred_communities, num_nodes),
        'nmi': compute_nmi(true_communities, pred_communities, num_nodes),
        'ari': compute_ari(true_communities, pred_communities, num_nodes),
        'modularity': compute_modularity(G, pred_communities),
        'coverage': compute_coverage(G, pred_communities),
        'conductance': compute_conductance(G, pred_communities),
        'num_communities': len(pred_communities),
        'num_true_communities': len(true_communities),
    }
    
    if overlapping:
        metrics.update({
            'onmi': compute_onmi(true_communities, pred_communities, num_nodes),
            'omega_index': compute_omega_index(true_communities, pred_communities, num_nodes),
            'f1': compute_f1_communities(true_communities, pred_communities),
        })
    
    return metrics


if __name__ == '__main__':
    # Test metrics
    print("Testing metrics module...")
    
    # Create a simple test case
    G = nx.karate_club_graph()
    
    # Ground truth (approximate)
    true_communities = [
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21],
        [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 8, 9, 14, 15, 18, 20, 22]
    ]
    
    # Predicted (slightly different)
    pred_communities = [
        [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13],
        [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
        [8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    ]
    
    metrics = compute_all_metrics(G, true_communities, pred_communities)
    
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")




