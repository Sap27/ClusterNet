"""
Spin Glass Community Detection Algorithm

Uses statistical mechanics approach where nodes are treated as spins
and community structure emerges from energy minimization.

NOTE: Spin Glass requires a connected graph. This implementation 
automatically extracts the largest connected component if the graph
is not fully connected.

Implementation matches main.py spin_glass() function exactly.
"""

import time
import networkx as nx
import igraph
from typing import List, Tuple


class SpinGlassAlgorithm:
    """
    Spin Glass Algorithm for community detection (STRICT REFERENCE MATCH with main.py).
    
    This implementation matches the reference spin_glass() function exactly:
    - Converts directed to undirected via G.to_undirected()
    - Gets largest connected component
    - Uses igraph.Graph.from_networkx()
    - Only passes spins parameter and weights if weighted
    
    IMPORTANT: Requires connected graph. Automatically uses largest 
    connected component if graph is disconnected.
    
    Reference:
        Reichardt & Bornholdt (2006). "Statistical mechanics of community detection"
    """
    
    def __init__(self, G, weighted=True, directed=False, spins=25):
        """
        Initialize Spin Glass algorithm.
        
        Args:
            G: NetworkX graph
            weighted: Whether graph is weighted (default: True)
            directed: Whether graph is directed (default: False)
            spins: Number of spins (max communities) (default: 25)
        """
        self.weighted = weighted
        self.directed = directed
        self.spins = int(spins)
        self.G = G
        
    def run(self) -> Tuple[List[List], float]:
        """Run the Spin Glass algorithm matching main.py exactly."""
        G = self.G
        
        # Convert directed to undirected (matching main.py)
        if self.directed:
            G = G.to_undirected()
        
        # Get largest connected component (matching main.py)
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        
        # Relabel nodes to sequential integers
        original_nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
        reverse_mapping = {idx: node for node, idx in node_mapping.items()}
        
        G_relabelled = nx.relabel_nodes(G, node_mapping)
        
        # Convert to igraph (matching main.py)
        g = igraph.Graph.from_networkx(G_relabelled)
        
        t1 = time.time()
        if self.weighted:
            label = g.community_spinglass(weights=g.es["weight"], spins=self.spins)
        else:
            label = g.community_spinglass(spins=self.spins)
        t2 = time.time()
        
        # Convert to communities list
        label_communities = [list(label[i]) for i in range(len(label))]
        label_communities = [[int(reverse_mapping[j]) for j in community] for community in label_communities]
        
        return label_communities, t2 - t1


# ==========================================
# STANDALONE FUNCTION (matches main.py)
# ==========================================

def spin_glass(G=None, weighted=True, directed=False, network_file='', spins=25):
    """
    Spin Glass community detection - matches main.py exactly.
    
    Args:
        G: NetworkX graph (optional if network_file provided)
        weighted: Whether graph is weighted
        directed: Whether graph is directed
        network_file: Path to edge list file (optional if G provided)
        spins: Number of spins/max communities (default: 25)
        
    Returns:
        Tuple of (communities, runtime)
    """
    from clusternet.utils.graph_utils import process_network_file, str_to_bool
    
    weighted = str_to_bool(weighted)
    directed = str_to_bool(directed)
    spins = int(spins)
    
    if network_file != '':
        G = process_network_file(network_file=network_file, directed=directed, weighted=weighted)
    
    if directed:
        G = G.to_undirected()
    
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    
    original_nodes = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    # Relabel the graph nodes with new sequential integers
    G_relabelled = nx.relabel_nodes(G, node_mapping)
    
    g = igraph.Graph.from_networkx(G_relabelled)
    
    t1 = time.time()
    if weighted:
        label = g.community_spinglass(weights=g.es["weight"], spins=spins)
    else:
        label = g.community_spinglass(spins=spins)
    t2 = time.time()
    
    label_communities = [list(label[i]) for i in range(len(label))]
    label_communities = [[int(reverse_mapping[j]) for j in community] for community in label_communities]
    
    return label_communities, t2 - t1


# ==========================================
# WRAPPER FUNCTION FOR CLUSTERNET
# ==========================================

def spinglass_clustering(input_data, spins=25, weighted=True, directed=False):
    """
    Main entry point for Spin Glass in ClusterNet.
    
    NOTE: Spin Glass requires a connected graph. If the input graph is
    disconnected, the largest connected component will be used.
    
    Args:
        input_data: NetworkX graph OR file path (edgelist).
        spins: Number of spins/max communities (default: 25)
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
    algo = SpinGlassAlgorithm(G, weighted=weighted, directed=directed, spins=spins)
    communities, runtime = algo.run()
    
    return communities
