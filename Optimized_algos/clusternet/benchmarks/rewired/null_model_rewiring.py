"""
Null Model-Based Network Rewiring

Adapted from Community-nullmodel project with enhancements.
https://github.com/xxx/Community-nullmodel

Provides sophisticated rewiring that preserves network properties at different orders:
- 0k: Preserves edge count
- 1k: Preserves degree distribution
- 2k: Preserves degree correlation (assortativity)
- 2.5k: Preserves degree-dependent clustering
- 3k: Preserves clustering coefficient

Key functions:
- Q_decrease_*k: Weaken community structure (increase mixing)
- Q_increase_*k: Strengthen community structure
- inner_random_*k: Shuffle within-community edges
- inter_random_*k: Shuffle between-community edges
"""

import networkx as nx
import numpy as np
import random
import copy
from typing import List, Optional, Tuple, Set


def edge_in_community(communities: List[Set], edge: Tuple[int, int]) -> bool:
    """
    Check if an edge is within a community (intra-community).
    
    Args:
        communities: List of community sets
        edge: Tuple of (node1, node2)
        
    Returns:
        True if both endpoints are in the same community
    """
    u, v = edge
    for community in communities:
        if u in community and v in community:
            return True
    return False


def _dict_degree_nodes(degree_node_list: List[Tuple[int, int]]) -> dict:
    """
    Convert list of (node, degree) to dict of {degree: [nodes]}.
    
    Args:
        degree_node_list: List of (node, degree) tuples
        
    Returns:
        Dictionary mapping degree to list of nodes with that degree
    """
    D = {}
    for degree, node in degree_node_list:
        if degree not in D:
            D[degree] = [node]
        else:
            D[degree].append(node)
    return D


# =========================================
# COMMUNITY STRUCTURE WEAKENING (Q DECREASE)
# Converts intra-community edges to inter-community
# =========================================

def Q_decrease_1k(
    G: nx.Graph,
    communities: List[List],
    nswap: int = 1,
    max_tries: int = 100,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Weaken community structure while preserving degree distribution (1k).
    
    Converts intra-community edges to inter-community edges,
    reducing modularity while keeping the same degree sequence.
    
    Args:
        G: NetworkX graph
        communities: List of communities (lists of node IDs)
        nswap: Number of successful swaps to perform
        max_tries: Maximum swap attempts
        seed: Random seed
        
    Returns:
        Graph with weakened community structure
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if G.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    if nswap > max_tries:
        raise nx.NetworkXError("nswap must be <= max_tries")
    if len(G) < 4:
        raise nx.NetworkXError("Graph needs at least 4 nodes")
    
    # Convert communities to sets for faster lookup
    comm_sets = [set(c) for c in communities]
    
    G_new = copy.deepcopy(G)
    keys, degrees = zip(*list(G_new.degree()))
    cdf = nx.utils.cumulative_distribution(list(degrees))
    
    tn = 0  # attempts
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
        
        neighbors_u = list(G_new[u])
        neighbors_x = list(G_new[x])
        
        if not neighbors_u or not neighbors_x:
            continue
            
        v = random.choice(neighbors_u)
        y = random.choice(neighbors_x)
        
        # Ensure 4 distinct nodes
        if len({u, v, x, y}) != 4:
            continue
        
        # Check: old edges must be intra-community
        if not (edge_in_community(comm_sets, (u, v)) and 
                edge_in_community(comm_sets, (x, y))):
            continue
        
        # Check: new edges must be inter-community
        if (edge_in_community(comm_sets, (u, y)) or 
            edge_in_community(comm_sets, (v, x))):
            continue
        
        # Check: new edges don't already exist
        if y in G_new[u] or v in G_new[x]:
            continue
        
        # Perform swap
        G_new.remove_edge(u, v)
        G_new.remove_edge(x, y)
        G_new.add_edge(u, y)
        G_new.add_edge(v, x)
        
        swapcount += 1
    
    return G_new


def Q_decrease_2k(
    G: nx.Graph,
    communities: List[List],
    nswap: int = 1,
    max_tries: int = 100,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Weaken community structure while preserving degree correlation (2k).
    
    Like Q_decrease_1k but also preserves degree-degree correlation
    (assortativity).
    
    Args:
        G: NetworkX graph
        communities: List of communities
        nswap: Number of successful swaps
        max_tries: Maximum attempts
        seed: Random seed
        
    Returns:
        Graph with weakened community structure
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if G.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    
    comm_sets = [set(c) for c in communities]
    
    G_new = copy.deepcopy(G)
    keys, degrees = zip(*list(G_new.degree()))
    cdf = nx.utils.cumulative_distribution(list(degrees))
    
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
        
        neighbors_u = list(G_new[u])
        neighbors_x = list(G_new[x])
        
        if not neighbors_u or not neighbors_x:
            continue
            
        v = random.choice(neighbors_u)
        y = random.choice(neighbors_x)
        
        if len({u, v, x, y}) != 4:
            continue
        
        # Old edges must be intra-community
        if not (edge_in_community(comm_sets, (u, v)) and 
                edge_in_community(comm_sets, (x, y))):
            continue
        
        # New edges must be inter-community
        if (edge_in_community(comm_sets, (u, y)) or 
            edge_in_community(comm_sets, (v, x))):
            continue
        
        # Preserve degree correlation: degrees of swapped endpoints must match
        if G_new.degree(v) != G_new.degree(y):
            continue
        
        if y in G_new[u] or v in G_new[x]:
            continue
        
        G_new.remove_edge(u, v)
        G_new.remove_edge(x, y)
        G_new.add_edge(u, y)
        G_new.add_edge(v, x)
        
        swapcount += 1
    
    return G_new


def Q_decrease_3k(
    G: nx.Graph,
    communities: List[List],
    nswap: int = 1,
    max_tries: int = 100,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Weaken community structure while preserving clustering coefficient (3k).
    
    The most restrictive variant - preserves local clustering.
    
    Args:
        G: NetworkX graph
        communities: List of communities
        nswap: Number of successful swaps
        max_tries: Maximum attempts
        seed: Random seed
        
    Returns:
        Graph with weakened community structure
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if G.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    
    comm_sets = [set(c) for c in communities]
    G0 = G  # Original for comparison
    
    G_new = copy.deepcopy(G)
    keys, degrees = zip(*list(G_new.degree()))
    cdf = nx.utils.cumulative_distribution(list(degrees))
    
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
        
        neighbors_u = list(G_new[u])
        neighbors_x = list(G_new[x])
        
        if not neighbors_u or not neighbors_x:
            continue
            
        v = random.choice(neighbors_u)
        y = random.choice(neighbors_x)
        
        if len({u, v, x, y}) != 4:
            continue
        
        if not (edge_in_community(comm_sets, (u, v)) and 
                edge_in_community(comm_sets, (x, y))):
            continue
        
        if (edge_in_community(comm_sets, (u, y)) or 
            edge_in_community(comm_sets, (v, x))):
            continue
        
        if G_new.degree(v) != G_new.degree(y):
            continue
        
        if y in G_new[u] or v in G_new[x]:
            continue
        
        # Perform swap
        G_new.remove_edge(u, v)
        G_new.remove_edge(x, y)
        G_new.add_edge(u, y)
        G_new.add_edge(v, x)
        
        # Check clustering coefficient preservation
        affected_nodes = [u, v, x, y] + list(G_new[u]) + list(G_new[v]) + list(G_new[x]) + list(G_new[y])
        affected_nodes = list(set(affected_nodes))
        
        cc_old = nx.clustering(G0, nodes=affected_nodes)
        cc_new = nx.clustering(G_new, nodes=affected_nodes)
        
        if cc_old != cc_new:
            # Revert
            G_new.remove_edge(u, y)
            G_new.remove_edge(v, x)
            G_new.add_edge(u, v)
            G_new.add_edge(x, y)
            continue
        
        swapcount += 1
    
    return G_new


# =========================================
# COMMUNITY STRUCTURE STRENGTHENING (Q INCREASE)
# Converts inter-community edges to intra-community
# =========================================

def Q_increase_1k(
    G: nx.Graph,
    communities: List[List],
    nswap: int = 1,
    max_tries: int = 100,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Strengthen community structure while preserving degree distribution (1k).
    
    Converts inter-community edges to intra-community edges,
    increasing modularity.
    
    Args:
        G: NetworkX graph
        communities: List of communities
        nswap: Number of successful swaps
        max_tries: Maximum attempts
        seed: Random seed
        
    Returns:
        Graph with strengthened community structure
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if G.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    
    comm_sets = [set(c) for c in communities]
    
    G_new = copy.deepcopy(G)
    keys, degrees = zip(*list(G_new.degree()))
    cdf = nx.utils.cumulative_distribution(list(degrees))
    
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
        
        neighbors_u = list(G_new[u])
        neighbors_x = list(G_new[x])
        
        if not neighbors_u or not neighbors_x:
            continue
            
        v = random.choice(neighbors_u)
        y = random.choice(neighbors_x)
        
        if len({u, v, x, y}) != 4:
            continue
        
        # Old edges must be INTER-community (opposite of Q_decrease)
        if (edge_in_community(comm_sets, (u, v)) or 
            edge_in_community(comm_sets, (x, y))):
            continue
        
        # New edges must be INTRA-community
        if not (edge_in_community(comm_sets, (u, y)) and 
                edge_in_community(comm_sets, (v, x))):
            continue
        
        if y in G_new[u] or v in G_new[x]:
            continue
        
        G_new.remove_edge(u, v)
        G_new.remove_edge(x, y)
        G_new.add_edge(u, y)
        G_new.add_edge(v, x)
        
        swapcount += 1
    
    return G_new


# =========================================
# EDGE SHUFFLING (MESOSCALE PRESERVATION)
# Randomize edges while keeping community structure
# =========================================

def inner_random_1k(
    G: nx.Graph,
    communities: List[List],
    nswap: int = 1,
    max_tries: int = 100,
    connected: bool = True,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Shuffle intra-community edges while preserving degree distribution.
    
    Randomizes edges WITHIN communities without changing
    community membership or degree sequence.
    
    Args:
        G: NetworkX graph  
        communities: List of communities
        nswap: Number of successful swaps
        max_tries: Maximum attempts
        connected: Ensure graph stays connected
        seed: Random seed
        
    Returns:
        Graph with shuffled intra-community edges
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if G.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    
    comm_sets = [set(c) for c in communities]
    
    G_new = copy.deepcopy(G)
    keys, degrees = zip(*list(G_new.degree()))
    cdf = nx.utils.cumulative_distribution(list(degrees))
    
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
        
        neighbors_u = list(G_new[u])
        neighbors_x = list(G_new[x])
        
        if not neighbors_u or not neighbors_x:
            continue
            
        v = random.choice(neighbors_u)
        y = random.choice(neighbors_x)
        
        if len({u, v, x, y}) != 4:
            continue
        
        # Both old and new edges must be INTRA-community
        if not (edge_in_community(comm_sets, (u, v)) and 
                edge_in_community(comm_sets, (x, y)) and
                edge_in_community(comm_sets, (u, y)) and
                edge_in_community(comm_sets, (v, x))):
            continue
        
        if y in G_new[u] or v in G_new[x]:
            continue
        
        G_new.remove_edge(u, v)
        G_new.remove_edge(x, y)
        G_new.add_edge(u, y)
        G_new.add_edge(v, x)
        
        # Check connectivity
        if connected and not nx.is_connected(G_new):
            # Revert
            G_new.remove_edge(u, y)
            G_new.remove_edge(v, x)
            G_new.add_edge(u, v)
            G_new.add_edge(x, y)
            continue
        
        swapcount += 1
    
    return G_new


def inter_random_1k(
    G: nx.Graph,
    communities: List[List],
    nswap: int = 1,
    max_tries: int = 100,
    connected: bool = True,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Shuffle inter-community edges while preserving degree distribution.
    
    Randomizes edges BETWEEN communities without changing
    degree sequence or community membership.
    
    Args:
        G: NetworkX graph
        communities: List of communities
        nswap: Number of successful swaps
        max_tries: Maximum attempts
        connected: Ensure graph stays connected
        seed: Random seed
        
    Returns:
        Graph with shuffled inter-community edges
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if G.is_directed():
        raise nx.NetworkXError("Only works on undirected graphs")
    
    comm_sets = [set(c) for c in communities]
    
    G_new = copy.deepcopy(G)
    keys, degrees = zip(*list(G_new.degree()))
    cdf = nx.utils.cumulative_distribution(list(degrees))
    
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
        
        neighbors_u = list(G_new[u])
        neighbors_x = list(G_new[x])
        
        if not neighbors_u or not neighbors_x:
            continue
            
        v = random.choice(neighbors_u)
        y = random.choice(neighbors_x)
        
        if len({u, v, x, y}) != 4:
            continue
        
        # Both old and new edges must be INTER-community
        if (edge_in_community(comm_sets, (u, v)) or 
            edge_in_community(comm_sets, (x, y)) or
            edge_in_community(comm_sets, (u, y)) or
            edge_in_community(comm_sets, (v, x))):
            continue
        
        if y in G_new[u] or v in G_new[x]:
            continue
        
        G_new.remove_edge(u, v)
        G_new.remove_edge(x, y)
        G_new.add_edge(u, y)
        G_new.add_edge(v, x)
        
        if connected and not nx.is_connected(G_new):
            G_new.remove_edge(u, y)
            G_new.remove_edge(v, x)
            G_new.add_edge(u, v)
            G_new.add_edge(x, y)
            continue
        
        swapcount += 1
    
    return G_new


# =========================================
# HIGH-LEVEL BENCHMARK FUNCTIONS
# =========================================

def weaken_community_structure(
    G: nx.Graph,
    communities: List[List],
    strength: float = 0.5,
    preserve_order: str = '1k',
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Weaken community structure by specified amount.
    
    Args:
        G: NetworkX graph
        communities: List of communities
        strength: Weakening strength (0=none, 1=maximum)
        preserve_order: Property preservation level ('1k', '2k', '3k')
        seed: Random seed
        
    Returns:
        Graph with weakened communities
    """
    m = G.number_of_edges()
    nswap = int(m * strength)
    max_tries = nswap * 10
    
    if preserve_order == '1k':
        return Q_decrease_1k(G, communities, nswap, max_tries, seed)
    elif preserve_order == '2k':
        return Q_decrease_2k(G, communities, nswap, max_tries, seed)
    elif preserve_order == '3k':
        return Q_decrease_3k(G, communities, nswap, max_tries, seed)
    else:
        raise ValueError(f"Unknown preserve_order: {preserve_order}")


def strengthen_community_structure(
    G: nx.Graph,
    communities: List[List],
    strength: float = 0.5,
    preserve_order: str = '1k',
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Strengthen community structure by specified amount.
    
    Args:
        G: NetworkX graph
        communities: List of communities
        strength: Strengthening amount (0=none, 1=maximum)
        preserve_order: Property preservation level
        seed: Random seed
        
    Returns:
        Graph with strengthened communities
    """
    m = G.number_of_edges()
    nswap = int(m * strength)
    max_tries = nswap * 10
    
    return Q_increase_1k(G, communities, nswap, max_tries, seed)


def generate_null_model_sweep(
    G: nx.Graph,
    communities: List[List],
    strengths: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    direction: str = 'weaken',
    preserve_order: str = '1k',
    seed: int = 42
) -> dict:
    """
    Generate a sweep of rewired networks at different strengths.
    
    Args:
        G: Original graph
        communities: Ground truth communities
        strengths: List of rewiring strengths
        direction: 'weaken' or 'strengthen'
        preserve_order: Property preservation level
        seed: Random seed
        
    Returns:
        Dictionary mapping strength -> rewired graph
    """
    results = {}
    
    for strength in strengths:
        if strength == 0.0:
            results[strength] = G.copy()
        else:
            if direction == 'weaken':
                results[strength] = weaken_community_structure(
                    G, communities, strength, preserve_order, seed
                )
            else:
                results[strength] = strengthen_community_structure(
                    G, communities, strength, preserve_order, seed
                )
    
    return results


def compute_mixing_parameter(G: nx.Graph, communities: List[List]) -> float:
    """
    Compute mixing parameter (μ) for a graph.
    
    μ = fraction of inter-community edges
    
    Args:
        G: NetworkX graph
        communities: List of communities
        
    Returns:
        Mixing parameter in [0, 1]
    """
    comm_sets = [set(c) for c in communities]
    
    inter_edges = 0
    total_edges = G.number_of_edges()
    
    for u, v in G.edges():
        if not edge_in_community(comm_sets, (u, v)):
            inter_edges += 1
    
    return inter_edges / total_edges if total_edges > 0 else 0.0









