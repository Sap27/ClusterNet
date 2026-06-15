"""
Label Propagation Community Detection Algorithm

A near linear time algorithm for detecting community structure in networks.
Each node is initialized with a unique label and at every step each node
adopts the label that most of its neighbors currently have.

Implementation matches main.py label_propogation() function exactly.
Uses igraph's community_label_propagation.
"""

import time
import networkx as nx
import igraph
from typing import List, Tuple


class LabelPropagationAlgorithm:
    """
    Label Propagation Algorithm for community detection (STRICT REFERENCE MATCH with main.py).
    
    This implementation matches the reference label_propogation() function exactly:
    - Converts directed to undirected via G.to_undirected()
    - Uses igraph.Graph.from_networkx()
    - Uses igraph's community_label_propagation
    - Passes weights if weighted
    
    Reference:
        Raghavan, Albert, Kumara (2007). "Near linear time algorithm to detect 
        community structures in large-scale networks"
    """
    
    def __init__(self, G, weighted=True, directed=False, spins=None):
        """
        Initialize Label Propagation algorithm.
        
        Args:
            G: NetworkX graph
            weighted: Whether graph is weighted (default: True)
            directed: Whether graph is directed (default: False)
            spins: Not used, kept for compatibility
        """
        self.weighted = weighted
        self.directed = directed
        self.G = G
        
    def run(self) -> Tuple[List[List], float]:
        """Run the Label Propagation algorithm matching main.py exactly."""
        G = self.G
        
        # Convert directed to undirected (matching main.py)
        if self.directed:
            G = G.to_undirected()
        
        # Relabel nodes to sequential integers
        original_nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
        reverse_mapping = {idx: node for node, idx in node_mapping.items()}
        
        G_relabelled = nx.relabel_nodes(G, node_mapping)
        
        # Convert to igraph (matching main.py)
        g = igraph.Graph.from_networkx(G_relabelled)
        
        t1 = time.time()
        if self.weighted:
            label = g.community_label_propagation(weights=g.es["weight"])
        else:
            label = g.community_label_propagation()
        t2 = time.time()
        
        # Convert to communities list
        label_communities = [list(label[i]) for i in range(len(label))]
        label_communities = [[int(reverse_mapping[j]) for j in community] for community in label_communities]
        
        return label_communities, t2 - t1


# ==========================================
# STANDALONE FUNCTION (matches main.py)
# ==========================================

def label_propogation(G=None, weighted=True, directed=False, network_file='', spins=None):
    """
    Label Propagation community detection - matches main.py exactly.
    
    Args:
        G: NetworkX graph (optional if network_file provided)
        weighted: Whether graph is weighted
        directed: Whether graph is directed
        network_file: Path to edge list file (optional if G provided)
        spins: Not used, kept for compatibility with main.py signature
        
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
    
    g = igraph.Graph.from_networkx(G_relabelled)
    
    t1 = time.time()
    if weighted:
        label = g.community_label_propagation(weights=g.es["weight"])
    else:
        label = g.community_label_propagation()
    t2 = time.time()
    
    label_communities = [list(label[i]) for i in range(len(label))]
    label_communities = [[int(reverse_mapping[j]) for j in community] for community in label_communities]
    
    return label_communities, t2 - t1


# ==========================================
# WRAPPER FUNCTION FOR CLUSTERNET
# ==========================================

def label_propagation_clustering(input_data, weighted=True, directed=False):
    """
    Main entry point for Label Propagation in ClusterNet.
    
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
    algo = LabelPropagationAlgorithm(G, weighted=weighted, directed=directed)
    communities, runtime = algo.run()
    
    return communities
