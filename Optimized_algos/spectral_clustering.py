"""
Spectral Clustering for Community Detection

Uses eigenvalues and eigenvectors of graph matrices (Laplacian) to 
partition nodes into communities.

Implementation matches main.py spectral_clustering() function exactly.
"""

import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn.cluster import SpectralClustering
from typing import List, Tuple, Union


class SpectralClusteringAlgorithm:
    """
    Spectral Clustering (STRICT REFERENCE MATCH with main.py)

    This implementation matches the reference spectral_clustering() function exactly:
    - Gets largest connected component (weakly for directed, strongly for undirected)
    - Relabels nodes to sequential integers
    - Builds adjacency matrix using nx.adjacency_matrix()
    - Uses sklearn SpectralClustering with affinity='precomputed'
    """

    def __init__(
        self,
        G: nx.Graph,
        *,
        weighted: bool = True,
        directed: bool = False,
        n_clusters: int = 30,
        n_components: int = 10,
    ):
        self.G = G
        self.weighted = weighted
        self.directed = directed
        self.n_clusters = int(n_clusters)
        self.n_components = int(n_components)

    def run(self) -> Tuple[List[List[int]], float]:
        """
        Run spectral clustering matching main.py exactly.

        Returns:
            cluster_list: List[List[node]]
            runtime: float (seconds)
        """
        G = self.G

        # --- 1. GET LARGEST CONNECTED COMPONENT (matching main.py) ---
        if self.directed:
            weakly_connected_components = nx.weakly_connected_components(G)
            largest_wcc = max(weakly_connected_components, key=len)
            G = G.subgraph(largest_wcc).copy()
        else:
            connected_components = nx.connected_components(G)
            largest_cc = max(connected_components, key=len)
            G = G.subgraph(largest_cc).copy()

        # --- 2. RELABEL NODES TO SEQUENTIAL INTEGERS ---
        original_nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
        reverse_mapping = {idx: node for node, idx in node_mapping.items()}

        relabeled_G = nx.relabel_nodes(G, node_mapping)

        # --- 3. BUILD ADJACENCY MATRIX (matching main.py exactly) ---
        adj_mat = nx.adjacency_matrix(relabeled_G)
        adj_mat = sp.csr_matrix(adj_mat)
        adj_mat.indices = adj_mat.indices.astype('int32')
        adj_mat.indptr = adj_mat.indptr.astype(np.int32)

        # --- 4. RUN SPECTRAL CLUSTERING (matching main.py timing) ---
        t1 = time.time()
        sc = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            n_components=self.n_components
        )
        t2 = time.time()
        clusters = sc.fit_predict(adj_mat)

        # --- 5. FORMAT OUTPUT (matching main.py exactly) ---
        cluster_list = [[] for _ in range(sc.n_clusters)]
        for idx, cluster_label in enumerate(clusters):
            cluster_list[cluster_label].append(idx)
        
        cluster_list = [[int(reverse_mapping[j]) for j in i] for i in cluster_list]

        return cluster_list, t2 - t1


# ==========================================
# STANDALONE FUNCTION (matches main.py)
# ==========================================

def spectral_clustering(
    G: nx.Graph = None,
    weighted: bool = True,
    directed: bool = False,
    network_file: str = '',
    n_clusters: int = 30,
    n_components: int = 10
) -> Tuple[List[List[int]], float]:
    """
    Spectral clustering community detection - matches main.py exactly.
    
    Args:
        G: NetworkX graph (optional if network_file provided)
        weighted: Whether graph is weighted
        directed: Whether graph is directed
        network_file: Path to edge list file (optional if G provided)
        n_clusters: Number of clusters (default: 30)
        n_components: Number of eigenvector components (default: 10)
        
    Returns:
        Tuple of (communities, runtime)
    """
    # Input handling
    if network_file != '':
        from clusternet.utils.graph_utils import process_network_file, str_to_bool
        weighted = str_to_bool(weighted)
        directed = str_to_bool(directed)
        G = process_network_file(network_file=network_file, directed=directed, weighted=weighted)
    
    n_clusters = int(n_clusters)
    n_components = int(n_components)

    # Use the algorithm class
    algo = SpectralClusteringAlgorithm(
        G,
        weighted=weighted,
        directed=directed,
        n_clusters=n_clusters,
        n_components=n_components
    )
    return algo.run()


# ==========================================
# WRAPPER FUNCTION FOR CLUSTERNET
# ==========================================

def spectral_clustering_wrapper(
    input_data: Union[str, nx.Graph],
    n_clusters: int = 30,
    n_components: int = 10,
    weighted: bool = True,
    directed: bool = False
) -> List[List[int]]:
    """
    ClusterNet entry point for Spectral Clustering.
    
    Args:
        input_data: NetworkX graph OR file path (edgelist)
        n_clusters: Number of clusters (default: 30)
        n_components: Number of eigenvector components (default: 10)
        weighted: Whether graph is weighted (default: True)
        directed: Whether graph is directed (default: False)
        
    Returns:
        List of communities: [[node1, node2, ...], ...]
    """
    if isinstance(input_data, str):
        communities, _ = spectral_clustering(
            network_file=input_data,
            weighted=weighted,
            directed=directed,
            n_clusters=n_clusters,
            n_components=n_components
        )
    elif isinstance(input_data, (nx.Graph, nx.DiGraph)):
        communities, _ = spectral_clustering(
            G=input_data,
            weighted=weighted,
            directed=directed,
            n_clusters=n_clusters,
            n_components=n_components
        )
    else:
        raise ValueError("Input must be a file path string or a NetworkX Graph object.")
    
    return communities
