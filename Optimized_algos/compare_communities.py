
from sklearn.metrics import (normalized_mutual_info_score, 
                              adjusted_mutual_info_score,
                              adjusted_rand_score,
                              fowlkes_mallows_score,
                              v_measure_score,
                              homogeneity_score,
                              completeness_score)
import numpy as np
def communities_to_labels(communities):
    """Convert list of lists (communities) to node labels format
    
    Parameters
    ----------
    communities : list of lists
        Each sublist contains node IDs belonging to that community
        
    Returns
    -------
    dict
        Dictionary mapping node_id to community_id
    numpy.ndarray
        Array of labels where index is node_id and value is community_id
    """
    node_to_community = {}
    for community_id, community_nodes in enumerate(communities):
        for node in community_nodes:
            node_to_community[node] = community_id
    
    # Get all unique nodes
    all_nodes = sorted(node_to_community.keys())
    labels = np.array([node_to_community[node] for node in all_nodes])
    
    return node_to_community, labels


def align_communities(communities1, communities2):
    """Align two community structures to have the same node set
    
    Parameters
    ----------
    communities1 : list of lists
        First community structure
    communities2 : list of lists
        Second community structure
        
    Returns
    -------
    tuple
        (labels1, labels2) - aligned label arrays for the same node set
    """
    # Convert to dictionaries
    node2com1, _ = communities_to_labels(communities1)
    node2com2, _ = communities_to_labels(communities2)
    
    # Find common nodes
    nodes1 = set(node2com1.keys())
    nodes2 = set(node2com2.keys())
    common_nodes = sorted(nodes1.intersection(nodes2))
    
    if len(common_nodes) == 0:
        raise ValueError("No common nodes found between the two community structures")
    
    # Create aligned label arrays
    labels1 = np.array([node2com1[node] for node in common_nodes])
    labels2 = np.array([node2com2[node] for node in common_nodes])
    
    return labels1, labels2, common_nodes

def compare_communities(communities1, communities2, verbose=True):
    """Compare two community structures using various clustering metrics
    
    Parameters
    ----------
    communities1 : list of lists
        First community structure (each sublist is a community)
    communities2 : list of lists
        Second community structure (each sublist is a community)
    verbose : bool, optional
        If True, prints detailed results (default: True)
        
    Returns
    -------
    dict
        Dictionary containing all comparison metrics:
        - nmi: Normalized Mutual Information
        - ami: Adjusted Mutual Information
        - ari: Adjusted Rand Index
        - fmi: Fowlkes-Mallows Index
        - v_measure: V-measure score
        - homogeneity: Homogeneity score
        - completeness: Completeness score
        - num_communities_1: Number of communities in first structure
        - num_communities_2: Number of communities in second structure
        - num_common_nodes: Number of nodes present in both structures
        - num_only_in_1: Number of nodes only in first structure
        - num_only_in_2: Number of nodes only in second structure
        
    Notes
    -----
    - NMI (Normalized Mutual Information): Measures mutual information normalized [0, 1]
      1.0 means perfect agreement
    - AMI (Adjusted Mutual Information): NMI adjusted for chance [~0, 1]
      Accounts for agreement by random chance
    - ARI (Adjusted Rand Index): Similarity measure adjusted for chance [-1, 1]
      1.0 means perfect agreement, 0 means random labeling
    - FMI (Fowlkes-Mallows Index): Geometric mean of precision and recall [0, 1]
    - V-measure: Harmonic mean of homogeneity and completeness [0, 1]
    - Homogeneity: Each cluster contains only members of a single class [0, 1]
    - Completeness: All members of a class are in the same cluster [0, 1]
    
    Examples
    --------
    >>> communities1 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    >>> communities2 = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
    >>> metrics = compare_communities(communities1, communities2)
    >>> print(f"NMI: {metrics['nmi']:.4f}")
    >>> print(f"ARI: {metrics['ari']:.4f}")
    """
    # Get node sets
    nodes1 = set(node for community in communities1 for node in community)
    nodes2 = set(node for community in communities2 for node in community)
    
    # Calculate set differences
    common_nodes_set = nodes1.intersection(nodes2)
    only_in_1 = nodes1 - nodes2
    only_in_2 = nodes2 - nodes1
    
    # Align communities to common nodes
    labels1, labels2, common_nodes = align_communities(communities1, communities2)
    
    # Calculate all metrics
    nmi = normalized_mutual_info_score(labels1, labels2, average_method='arithmetic')
    ami = adjusted_mutual_info_score(labels1, labels2, average_method='arithmetic')
    ari = adjusted_rand_score(labels1, labels2)
    fmi = fowlkes_mallows_score(labels1, labels2)
    v_measure = v_measure_score(labels1, labels2)
    homogeneity = homogeneity_score(labels1, labels2)
    completeness = completeness_score(labels1, labels2)
    
    results = {
        'nmi': nmi,
        'ami': ami,
        'ari': ari,
        'fmi': fmi,
        'v_measure': v_measure,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'num_communities_1': len(communities1),
        'num_communities_2': len(communities2),
        'num_common_nodes': len(common_nodes_set),
        'num_only_in_1': len(only_in_1),
        'num_only_in_2': len(only_in_2),
        'total_nodes_1': len(nodes1),
        'total_nodes_2': len(nodes2)
    }
    
    if verbose:
        print("=" * 70)
        print("COMMUNITY STRUCTURE COMPARISON")
        print("=" * 70)
        print("\n📊 SIMILARITY METRICS:")
        print(f"  NMI (Normalized Mutual Information):    {nmi:.6f}")
        print(f"  AMI (Adjusted Mutual Information):      {ami:.6f}")
        print(f"  ARI (Adjusted Rand Index):              {ari:.6f}")
        print(f"  FMI (Fowlkes-Mallows Index):            {fmi:.6f}")
        print(f"  V-measure:                               {v_measure:.6f}")
        print(f"  Homogeneity:                             {homogeneity:.6f}")
        print(f"  Completeness:                            {completeness:.6f}")
        
        print("\n📈 STRUCTURE STATISTICS:")
        print(f"  Communities in structure 1:              {len(communities1)}")
        print(f"  Communities in structure 2:              {len(communities2)}")
        print(f"  Total nodes in structure 1:              {len(nodes1)}")
        print(f"  Total nodes in structure 2:              {len(nodes2)}")
        print(f"  Common nodes (used for comparison):      {len(common_nodes_set)}")
        print(f"  Nodes only in structure 1:               {len(only_in_1)}")
        print(f"  Nodes only in structure 2:               {len(only_in_2)}")
        
        print("\n💡 INTERPRETATION:")
        if nmi > 0.9:
            print("  ✓ Very high agreement between community structures")
        elif nmi > 0.7:
            print("  ✓ High agreement between community structures")
        elif nmi > 0.5:
            print("  ~ Moderate agreement between community structures")
        elif nmi > 0.3:
            print("  ~ Low agreement between community structures")
        else:
            print("  ✗ Very low agreement between community structures")
        
        print("=" * 70)
    
    return results
