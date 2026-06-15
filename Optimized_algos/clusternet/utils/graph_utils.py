"""
Graph utility functions for loading, saving, and preprocessing graphs.
"""

import networkx as nx
from typing import List, Optional, Union
import warnings


# ==========================================
# HELPER FUNCTIONS (matching main.py)
# ==========================================

def str_to_bool(s):
    """Convert string to boolean (matching main.py)."""
    if type(s) == str:
        return {'true': True, 'false': False}[s.lower()]
    else:
        return s


def process_network_file(network_file, weighted=True, directed=False):
    """
    Load a network file into a NetworkX graph (matching main.py).
    
    Args:
        network_file: Path to edge list file
        weighted: Whether to read edge weights
        directed: Whether to create directed graph
        
    Returns:
        NetworkX graph
    """
    if weighted:
        if directed:
            G = nx.read_edgelist(network_file, data=(('weight', float),), nodetype=int, create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(network_file, data=(('weight', float),), nodetype=int)
    else:
        if directed:
            G = nx.read_edgelist(network_file, nodetype=int, create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(network_file, nodetype=int)
    return G


def load_graph(file_path: str, 
               directed: bool = False,
               weighted: bool = True,
               delimiter: str = None,
               nodetype: type = int) -> nx.Graph:
    """
    Load a graph from an edge list file.
    
    File format: Each line should be: source target [weight]
    
    Args:
        file_path: Path to the edge list file
        directed: If True, create a directed graph
        weighted: If True, read edge weights (third column)
        delimiter: Delimiter for the file (auto-detected if None)
        nodetype: Type to convert node IDs to (default: int)
        
    Returns:
        NetworkX graph
        
    Example:
        >>> G = load_graph('network.dat', directed=False, weighted=True)
        >>> print(f"Loaded graph with {G.number_of_nodes()} nodes")
    """
    print(f"Loading graph from {file_path}...")
    
    # Create appropriate graph type
    G = nx.DiGraph() if directed else nx.Graph()
    
    # Read edge list
    try:
        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Split line
                if delimiter:
                    parts = line.split(delimiter)
                else:
                    parts = line.split()
                
                if len(parts) < 2:
                    warnings.warn(f"Line {line_num}: Invalid format (need at least 2 columns)")
                    continue
                
                # Parse nodes
                try:
                    source = nodetype(parts[0])
                    target = nodetype(parts[1])
                except ValueError as e:
                    warnings.warn(f"Line {line_num}: Could not parse nodes - {e}")
                    continue
                
                # Parse weight if applicable
                weight = 1.0
                if weighted and len(parts) >= 3:
                    try:
                        weight = float(parts[2])
                    except ValueError:
                        warnings.warn(f"Line {line_num}: Could not parse weight, using 1.0")
                
                # Add edge
                if weighted:
                    G.add_edge(source, target, weight=weight)
                else:
                    G.add_edge(source, target)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading graph: {str(e)}")
    
    print(f"✓ Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G


def save_communities(communities: List[List], 
                    file_path: str,
                    format: str = 'simple') -> None:
    """
    Save communities to a file.
    
    Args:
        communities: List of communities (each community is a list of nodes)
        file_path: Output file path
        format: Output format ('simple', 'detailed', or 'clusternet')
            - simple: One community per line, nodes space-separated
            - detailed: Community ID, size, then nodes
            - clusternet: Compatible with ClusterNet format (ID, score, nodes)
            
    Example:
        >>> communities = [[1, 2, 3], [4, 5, 6]]
        >>> save_communities(communities, 'output.txt', format='simple')
    """
    print(f"Saving {len(communities)} communities to {file_path}...")
    
    with open(file_path, 'w') as f:
        if format == 'simple':
            for comm in communities:
                f.write(' '.join(str(node) for node in comm) + '\n')
        
        elif format == 'detailed':
            for i, comm in enumerate(communities, 1):
                f.write(f"Community {i} (size={len(comm)}): ")
                f.write(' '.join(str(node) for node in comm) + '\n')
        
        elif format == 'clusternet':
            for i, comm in enumerate(communities, 1):
                f.write(f"{i}\t0.5\t")
                f.write('\t'.join(str(node) for node in comm) + '\n')
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    print(f"✓ Communities saved to {file_path}")


def preprocess_graph(G: nx.Graph, 
                     directed: bool = None,
                     weighted: bool = None,
                     remove_self_loops: bool = True,
                     ensure_connected: bool = False) -> nx.Graph:
    """
    Preprocess graph for community detection.
    
    Args:
        G: Input graph
        directed: If False, convert directed graph to undirected
        weighted: If False, remove edge weights
        remove_self_loops: If True, remove self-loops
        ensure_connected: If True, keep only largest connected component
        
    Returns:
        Preprocessed graph
    """
    G = G.copy()
    
    # Handle directed/undirected
    if directed is False and G.is_directed():
        # Convert to undirected (symmetrize)
        G_undirected = nx.Graph()
        for u, v, data in G.edges(data=True):
            w = data.get('weight', 1.0)
            if G_undirected.has_edge(u, v):
                G_undirected[u][v]['weight'] += w
            else:
                G_undirected.add_edge(u, v, weight=w)
        G_undirected.add_nodes_from(G.nodes())
        G = G_undirected
    
    elif directed is True and not G.is_directed():
        G = G.to_directed()
    
    # Handle weighted/unweighted
    if weighted is False:
        for u, v in G.edges():
            if 'weight' in G[u][v]:
                del G[u][v]['weight']
    elif weighted is True:
        # Ensure all edges have weights
        for u, v in G.edges():
            if 'weight' not in G[u][v]:
                G[u][v]['weight'] = 1.0
    
    # Remove self-loops
    if remove_self_loops:
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)
        if self_loops:
            print(f"  Removed {len(self_loops)} self-loops")
    
    # Keep largest component
    if ensure_connected:
        if G.is_directed():
            components = list(nx.weakly_connected_components(G))
        else:
            components = list(nx.connected_components(G))
        
        if len(components) > 1:
            largest = max(components, key=len)
            G = G.subgraph(largest).copy()
            print(f"  Kept largest component: {G.number_of_nodes()} nodes")
    
    return G


def validate_graph(G: nx.Graph, 
                   min_nodes: int = 2,
                   min_edges: int = 1,
                   allow_self_loops: bool = False) -> bool:
    """
    Validate that a graph is suitable for community detection.
    
    Args:
        G: Graph to validate
        min_nodes: Minimum number of nodes required
        min_edges: Minimum number of edges required
        allow_self_loops: Whether self-loops are allowed
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check minimum size
    if G.number_of_nodes() < min_nodes:
        raise ValueError(f"Graph has too few nodes: {G.number_of_nodes()} < {min_nodes}")
    
    if G.number_of_edges() < min_edges:
        raise ValueError(f"Graph has too few edges: {G.number_of_edges()} < {min_edges}")
    
    # Check for self-loops
    if not allow_self_loops:
        self_loops = list(nx.selfloop_edges(G))
        if self_loops:
            raise ValueError(f"Graph contains {len(self_loops)} self-loops (not allowed)")
    
    # Check for isolated nodes
    isolated = list(nx.isolates(G))
    if isolated:
        warnings.warn(f"Graph contains {len(isolated)} isolated nodes")
    
    return True


def graph_summary(G: nx.Graph) -> dict:
    """
    Get summary statistics for a graph.
    
    Args:
        G: Input graph
        
    Returns:
        Dictionary with graph statistics
    """
    stats = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'directed': G.is_directed(),
        'weighted': False,
        'density': nx.density(G),
    }
    
    # Check if weighted
    for _, _, data in G.edges(data=True):
        if 'weight' in data:
            stats['weighted'] = True
            break
    
    # Degree statistics
    degrees = [d for n, d in G.degree()]
    if degrees:
        import numpy as np
        stats['degree_mean'] = np.mean(degrees)
        stats['degree_median'] = np.median(degrees)
        stats['degree_max'] = max(degrees)
        stats['degree_min'] = min(degrees)
    
    # Connected components
    if G.is_directed():
        stats['weakly_connected_components'] = nx.number_weakly_connected_components(G)
        stats['strongly_connected_components'] = nx.number_strongly_connected_components(G)
    else:
        stats['connected_components'] = nx.number_connected_components(G)
    
    return stats


def print_graph_summary(G: nx.Graph) -> None:
    """Print a nicely formatted graph summary."""
    stats = graph_summary(G)
    
    print("\nGraph Summary:")
    print("=" * 50)
    print(f"  Nodes:        {stats['nodes']:,}")
    print(f"  Edges:        {stats['edges']:,}")
    print(f"  Directed:     {'Yes' if stats['directed'] else 'No'}")
    print(f"  Weighted:     {'Yes' if stats['weighted'] else 'No'}")
    print(f"  Density:      {stats['density']:.6f}")
    
    if 'degree_mean' in stats:
        print(f"\n  Degree Stats:")
        print(f"    Mean:       {stats['degree_mean']:.2f}")
        print(f"    Median:     {stats['degree_median']:.2f}")
        print(f"    Min/Max:    {stats['degree_min']} / {stats['degree_max']}")
    
    if 'connected_components' in stats:
        print(f"\n  Connected Components: {stats['connected_components']}")
    elif 'weakly_connected_components' in stats:
        print(f"\n  Weakly Connected:     {stats['weakly_connected_components']}")
        print(f"  Strongly Connected:   {stats['strongly_connected_components']}")
    
    print("=" * 50)

