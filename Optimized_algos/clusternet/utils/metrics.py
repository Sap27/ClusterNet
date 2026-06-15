"""
Metrics for evaluating community detection results.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_completeness_v_measure
)


def communities_to_labels(communities: List[List], num_nodes: Optional[int] = None) -> np.ndarray:
    """
    Convert community list-of-lists to label array.
    
    Args:
        communities: List of communities
        num_nodes: Total number of nodes (auto-detected if None)
        
    Returns:
        Array of community labels
    """
    # Determine number of nodes
    if num_nodes is None:
        all_nodes = set()
        for comm in communities:
            all_nodes.update(comm)
        num_nodes = len(all_nodes)
    
    labels = np.full(num_nodes, -1, dtype=int)
    
    for comm_id, comm in enumerate(communities):
        for node in comm:
            labels[node] = comm_id
    
    return labels


def compare_communities(communities1: List[List], 
                       communities2: List[List],
                       verbose: bool = True) -> Dict[str, float]:
    """
    Compare two community structures using multiple metrics.
    
    Args:
        communities1: First community structure
        communities2: Second community structure
        verbose: If True, print results
        
    Returns:
        Dictionary of metric scores
    """
    # Convert to labels
    all_nodes = set()
    for comm in communities1:
        all_nodes.update(comm)
    for comm in communities2:
        all_nodes.update(comm)
    
    # Create node mapping
    node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    num_nodes = len(all_nodes)
    
    # Convert communities to labels
    labels1 = np.full(num_nodes, -1, dtype=int)
    labels2 = np.full(num_nodes, -1, dtype=int)
    
    for comm_id, comm in enumerate(communities1):
        for node in comm:
            if node in node_to_idx:
                labels1[node_to_idx[node]] = comm_id
    
    for comm_id, comm in enumerate(communities2):
        for node in comm:
            if node in node_to_idx:
                labels2[node_to_idx[node]] = comm_id
    
    # Filter out unclustered nodes
    mask = (labels1 >= 0) & (labels2 >= 0)
    labels1 = labels1[mask]
    labels2 = labels2[mask]
    
    # Compute metrics
    metrics = {}
    
    try:
        metrics['nmi'] = normalized_mutual_info_score(labels1, labels2)
        metrics['ami'] = adjusted_mutual_info_score(labels1, labels2)
        metrics['ari'] = adjusted_rand_score(labels1, labels2)
        metrics['fmi'] = fowlkes_mallows_score(labels1, labels2)
        
        homo, comp, vmeas = homogeneity_completeness_v_measure(labels1, labels2)
        metrics['homogeneity'] = homo
        metrics['completeness'] = comp
        metrics['v_measure'] = vmeas
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {}
    
    if verbose:
        print("\nCommunity Comparison Metrics:")
        print("=" * 50)
        print(f"  NMI (Normalized Mutual Info):     {metrics['nmi']:.4f}")
        print(f"  AMI (Adjusted Mutual Info):       {metrics['ami']:.4f}")
        print(f"  ARI (Adjusted Rand Index):        {metrics['ari']:.4f}")
        print(f"  FMI (Fowlkes-Mallows Index):      {metrics['fmi']:.4f}")
        print(f"  V-Measure:                         {metrics['v_measure']:.4f}")
        print(f"  Homogeneity:                       {metrics['homogeneity']:.4f}")
        print(f"  Completeness:                      {metrics['completeness']:.4f}")
        print("=" * 50)
    
    return metrics


def compute_modularity(G: nx.Graph, communities: List[List]) -> float:
    """
    Compute modularity of a community structure.
    
    Args:
        G: NetworkX graph
        communities: List of communities
        
    Returns:
        Modularity score
    """
    # Convert to partition dict
    partition = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            partition[node] = comm_id
    
    try:
        modularity = nx.community.modularity(G, [set(comm) for comm in communities])
        return modularity
    except Exception as e:
        print(f"Error computing modularity: {e}")
        return 0.0


def compute_coverage(G: nx.Graph, communities: List[List]) -> float:
    """
    Compute coverage of a community structure.
    
    Coverage is the fraction of edges that fall within communities.
    
    Args:
        G: NetworkX graph
        communities: List of communities
        
    Returns:
        Coverage score (0 to 1)
    """
    try:
        coverage = nx.community.coverage(G, [set(comm) for comm in communities])
        return coverage
    except Exception as e:
        print(f"Error computing coverage: {e}")
        return 0.0


def compute_performance(G: nx.Graph, communities: List[List]) -> float:
    """
    Compute performance of a community structure.
    
    Performance measures the ratio of correctly identified edges.
    
    Args:
        G: NetworkX graph
        communities: List of communities
        
    Returns:
        Performance score (0 to 1)
    """
    try:
        performance = nx.community.performance(G, [set(comm) for comm in communities])
        return performance
    except Exception as e:
        print(f"Error computing performance: {e}")
        return 0.0


def community_statistics(communities: List[List], verbose: bool = True) -> Dict:
    """
    Compute statistics about community structure.
    
    Args:
        communities: List of communities
        verbose: If True, print results
        
    Returns:
        Dictionary of statistics
    """
    sizes = [len(comm) for comm in communities]
    
    stats = {
        'num_communities': len(communities),
        'total_nodes': sum(sizes),
        'min_size': min(sizes) if sizes else 0,
        'max_size': max(sizes) if sizes else 0,
        'mean_size': np.mean(sizes) if sizes else 0,
        'median_size': np.median(sizes) if sizes else 0,
        'std_size': np.std(sizes) if sizes else 0,
    }
    
    if verbose:
        print("\nCommunity Statistics:")
        print("=" * 50)
        print(f"  Number of communities:  {stats['num_communities']}")
        print(f"  Total nodes:            {stats['total_nodes']}")
        print(f"  Size range:             {stats['min_size']} - {stats['max_size']}")
        print(f"  Mean size:              {stats['mean_size']:.2f}")
        print(f"  Median size:            {stats['median_size']:.0f}")
        print(f"  Std deviation:          {stats['std_size']:.2f}")
        print("=" * 50)
    
    return stats


def evaluate_communities(G: nx.Graph, 
                        communities: List[List],
                        verbose: bool = True) -> Dict:
    """
    Comprehensive evaluation of a community structure.
    
    Args:
        G: NetworkX graph
        communities: List of communities
        verbose: If True, print results
        
    Returns:
        Dictionary of all metrics
    """
    results = {}
    
    # Community statistics
    results.update(community_statistics(communities, verbose=verbose))
    
    # Graph-based metrics
    results['modularity'] = compute_modularity(G, communities)
    results['coverage'] = compute_coverage(G, communities)
    results['performance'] = compute_performance(G, communities)
    
    if verbose:
        print("\nQuality Metrics:")
        print("=" * 50)
        print(f"  Modularity:   {results['modularity']:.4f}")
        print(f"  Coverage:     {results['coverage']:.4f}")
        print(f"  Performance:  {results['performance']:.4f}")
        print("=" * 50)
    
    return results

