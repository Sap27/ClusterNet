"""
Walktrap Community Detection Algorithm

Uses random walks to compute distances between vertices and then uses 
hierarchical clustering to identify communities.

Implementation matches main.py walktrap() function exactly.
"""

import time
import networkx as nx
import igraph
from typing import List, Tuple


class WalktrapAlgorithm:
    """
    Walktrap Community Detection (STRICT REFERENCE MATCH with main.py)

    This implementation matches the reference walktrap() function exactly:
    - Uses igraph.community_walktrap only
    - Explicit weighted / directed handling
    - For directed: Creates igraph.Graph(directed=True) and adds vertices/edges manually
    - For undirected: Uses igraph.Graph.from_networkx()
    - Explicit node relabeling + reverse mapping
    """

    def __init__(
        self,
        G: nx.Graph,
        *,
        weighted: bool = True,
        directed: bool = False,
        steps: int = 10,
    ):
        self.G = G
        self.weighted = weighted
        self.directed = directed
        self.steps = int(steps)

    def run(self) -> Tuple[List[List], float]:
        """
        Run Walktrap community detection.

        Returns:
            communities: List[List[node]]
            runtime: float (seconds)
        """
        # --- 1. RELABEL NODES TO SEQUENTIAL INTS ---
        original_nodes = list(self.G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
        reverse_mapping = {idx: node for node, idx in node_mapping.items()}

        G_relabelled = nx.relabel_nodes(self.G, node_mapping)

        # --- 2. BUILD IGRAPH (matching main.py exactly) ---
        if self.directed:
            g = igraph.Graph(directed=True)
            g.add_vertices(list(G_relabelled.nodes))
            g.add_edges(list(G_relabelled.edges))
        else:
            g = igraph.Graph.from_networkx(G_relabelled)

        # --- 3. RUN WALKTRAP ---
        t1 = time.time()
        if self.weighted:
            wtrap = g.community_walktrap(
                weights=g.es["weight"],
                steps=self.steps,
            )
        else:
            wtrap = g.community_walktrap(steps=self.steps)
        t2 = time.time()

        clust = wtrap.as_clustering()

        # --- 4. MAP BACK TO ORIGINAL NODE IDS ---
        walktrap_communities = [list(clust[i]) for i in range(len(clust))]
        walktrap_communities = [
            [int(reverse_mapping[j]) for j in community]
            for community in walktrap_communities
        ]

        return walktrap_communities, t2 - t1


# ==========================================
# STANDALONE FUNCTION (matches main.py)
# ==========================================

def walktrap(G=None, weighted=True, directed=False, network_file='', steps=10):
    """
    Walktrap community detection - matches main.py exactly.
    
    Args:
        G: NetworkX graph (optional if network_file provided)
        weighted: Whether graph is weighted
        directed: Whether graph is directed
        network_file: Path to edge list file (optional if G provided)
        steps: Number of steps in random walk (default: 10)
        
    Returns:
        Tuple of (communities, runtime)
    """
    from clusternet.utils.graph_utils import process_network_file
    
    if network_file != '':
        G = process_network_file(network_file=network_file, directed=directed, weighted=weighted)
    
    steps = int(steps)
    original_nodes = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    # Relabel the graph nodes with new sequential integers
    G_relabelled = nx.relabel_nodes(G, node_mapping)
    
    if directed:
        g = igraph.Graph(directed=True)
        g.add_vertices(list(G_relabelled.nodes))
        g.add_edges(list(G_relabelled.edges))
    else:
        g = igraph.Graph.from_networkx(G_relabelled)
    
    t1 = time.time()
    if weighted:
        wtrap = g.community_walktrap(weights=g.es["weight"], steps=steps)
    else:
        wtrap = g.community_walktrap(steps=steps)
    t2 = time.time()
    
    clust = wtrap.as_clustering()
    walktrap_communities = [list(clust[i]) for i in range(len(clust))]
    walktrap_communities = [[int(reverse_mapping[j]) for j in community] for community in walktrap_communities]
    
    return walktrap_communities, t2 - t1


# ==========================================
# WRAPPER FUNCTION FOR CLUSTERNET
# ==========================================

def walktrap_clustering(input_data, steps=10, weighted=True, directed=False):
    """
    Main entry point for Walktrap in ClusterNet.
    
    Args:
        input_data: NetworkX graph OR file path (edgelist).
        steps: Number of steps in random walk (default: 10)
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
    algo = WalktrapAlgorithm(G, weighted=weighted, directed=directed, steps=steps)
    communities, runtime = algo.run()
    
    return communities
