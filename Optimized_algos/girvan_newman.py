"""
Girvan-Newman Community Detection Algorithm

A hierarchical divisive algorithm based on edge betweenness centrality.
Repeatedly removes the edge with highest betweenness to reveal community structure.

Implementation matches main.py girvan_newman() function exactly.
"""

import time
import networkx as nx
from typing import List, Tuple


class GirvanNewmanAlgorithm:
    """
    Girvan-Newman Algorithm for community detection (STRICT REFERENCE MATCH with main.py).
    
    This implementation matches the reference girvan_newman() function exactly:
    - Converts directed to undirected via G.to_undirected()
    - Uses networkx's nx.community.girvan_newman
    - Returns first partition from generator (next(output))
    - Supports most_valuable_edge parameter
    
    Reference:
        Girvan & Newman (2002). "Community structure in social and biological networks"
    """
    
    def __init__(self, G, weighted=True, directed=False, most_valuable_edge=None):
        """
        Initialize Girvan-Newman algorithm.
        
        Args:
            G: NetworkX graph
            weighted: Whether graph is weighted (default: True)
            directed: Whether graph is directed (default: False)
            most_valuable_edge: Function to determine edge to remove 
                               (default: highest betweenness)
        """
        self.weighted = weighted
        self.directed = directed
        self.most_valuable_edge = most_valuable_edge
        self.G = G
        
    def run(self) -> Tuple[List[List], float]:
        """Run the Girvan-Newman algorithm matching main.py exactly."""
        G = self.G
        
        # Convert directed to undirected (matching main.py)
        if self.directed:
            G = G.to_undirected()
        
        # Relabel nodes to sequential integers
        original_nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
        reverse_mapping = {idx: node for node, idx in node_mapping.items()}
        
        G_relabelled = nx.relabel_nodes(G, node_mapping)
        
        t1 = time.time()
        output = nx.community.girvan_newman(G_relabelled, most_valuable_edge=self.most_valuable_edge)
        t2 = time.time()
        
        label_communities = list(sorted(c) for c in next(output))
        label_communities = [[int(reverse_mapping[j]) for j in community] for community in label_communities]
        
        return label_communities, t2 - t1


# ==========================================
# STANDALONE FUNCTION (matches main.py)
# ==========================================

def girvan_newman(G=None, weighted=True, directed=False, network_file='', most_valuable_edge=None):
    """
    Girvan-Newman community detection - matches main.py exactly.
    
    Args:
        G: NetworkX graph (optional if network_file provided)
        weighted: Whether graph is weighted
        directed: Whether graph is directed
        network_file: Path to edge list file (optional if G provided)
        most_valuable_edge: Function to determine edge to remove (default: None)
        
    Returns:
        Tuple of (communities, runtime)
    """
    from clusternet.utils.graph_utils import process_network_file
    
    if network_file != '':
        G = process_network_file(network_file=network_file, directed=directed, weighted=weighted)
    
    if directed:
        G = G.to_undirected()
    
    original_nodes = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    # Relabel the graph nodes with new sequential integers
    G_relabelled = nx.relabel_nodes(G, node_mapping)
    
    t1 = time.time()
    output = nx.community.girvan_newman(G_relabelled, most_valuable_edge=most_valuable_edge)
    t2 = time.time()
    
    label_communities = list(sorted(c) for c in next(output))
    label_communities = [[int(reverse_mapping[j]) for j in community] for community in label_communities]
    
    return label_communities, t2 - t1


# ==========================================
# WRAPPER FUNCTION FOR CLUSTERNET
# ==========================================

def girvan_newman_clustering(input_data, weighted=True, directed=False, most_valuable_edge=None):
    """
    Main entry point for Girvan-Newman in ClusterNet.
    
    NOTE: This algorithm can be slow on large graphs due to repeated
    betweenness centrality calculations.
    
    Args:
        input_data: NetworkX graph OR file path (edgelist).
        weighted: Whether graph is weighted (default: True)
        directed: Whether graph is directed (default: False)
        most_valuable_edge: Function to determine edge to remove (default: None)
        
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
    algo = GirvanNewmanAlgorithm(G, weighted=weighted, directed=directed, most_valuable_edge=most_valuable_edge)
    communities, runtime = algo.run()
    
    return communities
