"""
Fast Greedy (Clauset-Newman-Moore) Community Detection Algorithm

A hierarchical agglomerative algorithm that greedily optimizes modularity.

Implementation matches main.py fast_greedy() function exactly.
"""

import time
import networkx as nx
import igraph
from typing import List, Tuple


class FastGreedyAlgorithm:
    """
    Fast Greedy Algorithm for community detection (STRICT REFERENCE MATCH with main.py).
    
    This implementation matches the reference fast_greedy() function exactly:
    - Converts directed to undirected via G.to_undirected()
    - Uses igraph.Graph.from_networkx()
    - Uses igraph's community_fastgreedy
    - Passes weights if weighted
    
    Parameters:
        n_clusters: If specified, cut dendrogram at this number of clusters
                    If None, use optimal modularity cut
    
    Reference:
        Clauset, Newman, Moore (2004). "Finding community structure in very large networks"
    """
    
    def __init__(self, G, weighted=True, directed=False, n_clusters=None):
        """
        Initialize Fast Greedy algorithm.
        
        Args:
            G: NetworkX graph
            weighted: Whether graph is weighted (default: True)
            directed: Whether graph is directed (default: False)
            n_clusters: Target number of clusters (if None, use optimal modularity)
        """
        self.weighted = weighted
        self.directed = directed
        self.G = G
        self.n_clusters = n_clusters
        
    def run(self) -> Tuple[List[List], float]:
        """Run the Fast Greedy algorithm matching main.py exactly."""
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
            try:
                dendrogram = g.community_fastgreedy(weights=g.es["weight"])
            except KeyError:
                dendrogram = g.community_fastgreedy()
        else:
            dendrogram = g.community_fastgreedy()
        t2 = time.time()
        
        # Cut dendrogram at specified k or optimal
        if self.n_clusters is not None:
            # Ensure n_clusters is within valid range
            max_clusters = g.vcount()
            n = min(max(1, self.n_clusters), max_clusters)
            label = dendrogram.as_clustering(n=n)
        else:
            label = dendrogram.as_clustering()
        
        # Convert to communities list
        label_communities = [list(label[i]) for i in range(len(label))]
        label_communities = [[int(reverse_mapping[j]) for j in community] for community in label_communities]
        
        return label_communities, t2 - t1


# ==========================================
# STANDALONE FUNCTION (matches main.py)
# ==========================================

def fast_greedy(G=None, weighted=True, directed=False, network_file=''):
    """
    Fast Greedy community detection - matches main.py exactly.
    
    Args:
        G: NetworkX graph (optional if network_file provided)
        weighted: Whether graph is weighted
        directed: Whether graph is directed
        network_file: Path to edge list file (optional if G provided)
        
    Returns:
        Tuple of (communities, runtime)
    """
    from clusternet.utils.graph_utils import process_network_file, str_to_bool
    
    weighted = str_to_bool(weighted)
    directed = str_to_bool(directed)
    
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
        label = g.community_fastgreedy(weights=g.es["weight"])
    else:
        label = g.community_fastgreedy()
    t2 = time.time()
    
    label = label.as_clustering()
    label_communities = [list(label[i]) for i in range(len(label))]
    label_communities = [[int(reverse_mapping[j]) for j in community] for community in label_communities]
    
    return label_communities, t2 - t1


# ==========================================
# WRAPPER FUNCTION FOR CLUSTERNET
# ==========================================

def fastgreedy_clustering(input_data, weighted=True, directed=False):
    """
    Main entry point for Fast Greedy (CNM) in ClusterNet.
    
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
    algo = FastGreedyAlgorithm(G, weighted=weighted, directed=directed)
    communities, runtime = algo.run()
    
    return communities
