"""
Leading Eigenvector Community Detection Algorithm

Uses the eigenvector of the modularity matrix corresponding to the 
largest positive eigenvalue to iteratively bisect the network.

Implementation matches main.py leading_eigen_vector() function exactly.
"""

import time
import networkx as nx
import igraph
from typing import List, Tuple


class LeadingEigenAlgorithm:
    """
    Leading Eigenvector Algorithm for community detection (STRICT REFERENCE MATCH with main.py).
    
    This implementation matches the reference leading_eigen_vector() function exactly:
    - Uses igraph.Graph.from_networkx()
    - Uses igraph's community_leading_eigenvector
    - Passes weights if weighted
    
    Reference:
        Newman (2006). "Finding community structure in networks using the 
        eigenvectors of matrices"
    """
    
    def __init__(self, G, weighted=True, directed=False):
        """
        Initialize Leading Eigenvector algorithm.
        
        Args:
            G: NetworkX graph
            weighted: Whether graph is weighted (default: True)
            directed: Whether graph is directed (default: False)
        """
        self.weighted = weighted
        self.directed = directed
        self.G = G
        
    def run(self) -> Tuple[List[List], float]:
        """Run the Leading Eigenvector algorithm matching main.py exactly."""
        G = self.G
        
        # Relabel nodes to sequential integers
        original_nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
        reverse_mapping = {idx: node for node, idx in node_mapping.items()}
        
        G_relabelled = nx.relabel_nodes(G, node_mapping)
        
        # Convert to igraph (matching main.py)
        g = igraph.Graph.from_networkx(G_relabelled)
        
        t1 = time.time()
        if self.weighted:
            label = g.community_leading_eigenvector(weights=g.es["weight"])
        else:
            label = g.community_leading_eigenvector()
        t2 = time.time()
        
        # Convert to communities list
        label_communities = [list(label[i]) for i in range(len(label))]
        label_communities = [[int(reverse_mapping[j]) for j in community] for community in label_communities]
        
        return label_communities, t2 - t1


# ==========================================
# STANDALONE FUNCTION (matches main.py)
# ==========================================

def leading_eigen_vector(G=None, weighted=True, directed=False, network_file=''):
    """
    Leading Eigenvector community detection - matches main.py exactly.
    
    Args:
        G: NetworkX graph (optional if network_file provided)
        weighted: Whether graph is weighted
        directed: Whether graph is directed
        network_file: Path to edge list file (optional if G provided)
        
    Returns:
        Tuple of (communities, runtime)
    """
    from clusternet.utils.graph_utils import process_network_file
    
    if network_file != '':
        G = process_network_file(network_file=network_file, directed=directed, weighted=weighted)
    
    original_nodes = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    # Relabel the graph nodes with new sequential integers
    G_relabelled = nx.relabel_nodes(G, node_mapping)
    
    g = igraph.Graph.from_networkx(G_relabelled)
    
    t1 = time.time()
    if weighted:
        label = g.community_leading_eigenvector(weights=g.es["weight"])
    else:
        label = g.community_leading_eigenvector()
    t2 = time.time()
    
    label_communities = [list(label[i]) for i in range(len(label))]
    label_communities = [[int(reverse_mapping[j]) for j in community] for community in label_communities]
    
    return label_communities, t2 - t1


# ==========================================
# WRAPPER FUNCTION FOR CLUSTERNET
# ==========================================

def leading_eigen_clustering(input_data, weighted=True, directed=False):
    """
    Main entry point for Leading Eigenvector in ClusterNet.
    
    Args:
        input_data: NetworkX graph OR file path (edgelist).
        weighted: Whether graph is weighted (default: True)
        directed: Whether graph is directed (default: False)
        
    Returns:
        List of lists: [[node1, node2], ...]
    """
    
    # --- INPUT HANDLING ---
    if isinstance(input_data, str):
        from clusternet.utils.graph_utils import process_network_file
        G = process_network_file(network_file=input_data, directed=directed, weighted=weighted)
    elif isinstance(input_data, (nx.Graph, nx.DiGraph)):
        G = input_data
    else:
        raise ValueError("Input must be a file path string or a NetworkX Graph object.")

    # --- EXECUTION ---
    algo = LeadingEigenAlgorithm(G, weighted=weighted, directed=directed)
    communities, runtime = algo.run()
    
    return communities
