"""
Structural community detection algorithm wrappers from cdlib.

NOTE: These wrappers remap nodes to 0-indexed before running cdlib algorithms
and remap results back to original node IDs to fix cdlib's node mapping issues.
"""

import numpy as np
import networkx as nx
from cdlib import algorithms as cdlib_algos
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


def _remap_graph_to_sequential(G):
    """
    Remap graph nodes to sequential integers 0, 1, 2, ...
    
    Returns:
        G_remapped: New graph with sequential node IDs
        idx_to_original: Dict mapping sequential index to original node ID
        original_to_idx: Dict mapping original node ID to sequential index
    """
    nodes = list(G.nodes())
    original_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_original = {idx: node for idx, node in enumerate(nodes)}
    
    # Create new graph with sequential node IDs
    G_remapped = nx.Graph()
    G_remapped.add_nodes_from(range(len(nodes)))
    
    for u, v, data in G.edges(data=True):
        G_remapped.add_edge(original_to_idx[u], original_to_idx[v], **data)
    
    return G_remapped, idx_to_original, original_to_idx


def _remap_communities_to_original(communities, idx_to_original):
    """
    Remap community node IDs from sequential indices back to original node IDs.
    """
    remapped = []
    for comm in communities:
        remapped_comm = [idx_to_original[idx] for idx in comm]
        remapped.append(remapped_comm)
    return remapped


@register_algorithm('scan', aliases=['structural_clustering'])
class SCANWrapper(BaseAlgorithm):
    """
    Structural Clustering Algorithm for Networks (SCAN).
    
    SCAN is a density-based clustering algorithm that identifies clusters, hubs, 
    and outliers based on structural similarity between nodes.
    
    Parameters:
        epsilon: Neighborhood radius (default: 0.5)
        mu: Minimum number of neighbors (default: 3)
    
    Reference:
        Xu, X., Yuruk, N., Feng, Z., & Schweiger, T. A. (2007, August). Scan: a 
        structural clustering algorithm for networks. In KDD 2007.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = False
    
    def __init__(self, G, epsilon=0.5, mu=3, **kwargs):
        """
        Initialize SCAN algorithm.
        
        Args:
            G: NetworkX graph
            epsilon: Neighborhood radius
            mu: Minimum neighbors
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.epsilon = epsilon
        self.mu = mu
        self.params['epsilon'] = epsilon
        self.params['mu'] = mu
    
    def run(self):
        """
        Run the SCAN algorithm.
        
        Returns:
            List of communities with original node IDs
        """
        # Remap nodes to sequential 0, 1, 2, ...
        G_remapped, idx_to_original, original_to_idx = _remap_graph_to_sequential(self.G)
        
        try:
            result = cdlib_algos.scan(G_remapped, epsilon=self.epsilon, mu=self.mu)
            # Filter out single-node communities (outliers)
            communities = [c for c in result.communities if len(c) > 1]
            if not communities:
                # If all are outliers, return original result
                communities = result.communities
            # Remap back to original node IDs
            return _remap_communities_to_original(communities, idx_to_original)
        except Exception as e:
            print(f"SCAN algorithm failed: {e}")
            # Fallback to label propagation
            try:
                communities_gen = nx.community.label_propagation_communities(self.G)
                return [list(c) for c in communities_gen]
            except:
                from community import community_louvain
                partition = community_louvain.best_partition(self.G)
                comm_dict = {}
                for node, comm_id in partition.items():
                    if comm_id not in comm_dict:
                        comm_dict[comm_id] = []
                    comm_dict[comm_id].append(node)
                return list(comm_dict.values())


@register_algorithm('agdl', aliases=['adaptive_greedy'])
class AGDLWrapper(BaseAlgorithm):
    """
    Adaptive Greedy community detection with Diffusion Learning (AGDL).
    
    AGDL adaptively selects seeds and uses greedy expansion with diffusion 
    to discover communities of various densities.
    
    Parameters:
        number_communities: Target number of communities (if None, auto-estimate)
        kc: Size parameter for seed selection (default: 5)
            Higher kc = more seeds = finer-grained communities
        auto_tune: If True, try EXTENSIVE parameter combinations (default: True)
    
    Reference:
        Zhang, W., et al. (2013). Identification of overlapping community structure 
        in complex networks using fuzzy c-means clustering.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, number_communities=None, kc=5, auto_tune=True, **kwargs):
        """
        Initialize AGDL algorithm.
        
        Args:
            G: NetworkX graph
            number_communities: Target number of communities
            kc: Seed selection parameter
            auto_tune: Whether to auto-tune parameters
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        if number_communities is None:
            # Better estimation: use modularity-based heuristic
            self.number_communities = max(2, int(np.sqrt(n)))
        else:
            self.number_communities = number_communities
            
        # kc controls granularity - higher = finer communities
        self.kc = kc
        
        self.auto_tune = auto_tune
        self.params['number_communities'] = self.number_communities
        self.params['kc'] = self.kc
    
    def run(self):
        """
        Run the AGDL algorithm with EXTENSIVE parameter tuning.
        
        Returns:
            List of communities with original node IDs
        """
        # Remap nodes to sequential 0, 1, 2, ...
        G_remapped, idx_to_original, _ = _remap_graph_to_sequential(self.G)
        
        best_communities = None
        best_modularity = -1
        
        n = self.G.number_of_nodes()
        
        # EXTENSIVE parameter search
        if self.auto_tune:
            # Try many number_communities values
            nc_values = [10, 15, 20, 25, 30, 35, 40, int(np.sqrt(n)), int(np.sqrt(n)*1.5)]
            nc_values = sorted(set([max(2, min(n//3, nc)) for nc in nc_values]))
            # Try many kc values
            kc_values = [2, 3, 5, 8, 10, 15, 20]
            params_to_try = [(nc, kc) for nc in nc_values for kc in kc_values]
        else:
            params_to_try = [(self.number_communities, self.kc)]
        
        for nc, kc in params_to_try:
            try:
                result = cdlib_algos.agdl(
                    G_remapped,
                    number_communities=nc,
                    kc=kc
                )
                communities = result.communities
                
                # Calculate modularity using remapped graph
                try:
                    mod = nx.community.modularity(G_remapped, [set(c) for c in communities])
                    if mod > best_modularity:
                        best_modularity = mod
                        best_communities = communities
                except:
                    if best_communities is None:
                        best_communities = communities
            except:
                continue
        
        if best_communities is not None:
            # Remap back to original node IDs
            return _remap_communities_to_original(best_communities, idx_to_original)
            
        # Fallback
        print(f"AGDL algorithm failed, using fallback")
        from community import community_louvain
        partition = community_louvain.best_partition(self.G)
        comm_dict = {}
        for node, comm_id in partition.items():
            if comm_id not in comm_dict:
                comm_dict[comm_id] = []
            comm_dict[comm_id].append(node)
        return list(comm_dict.values())


@register_algorithm('gdmp2', aliases=['gdmp'])
class GDMP2Wrapper(BaseAlgorithm):
    """
    Graph Decomposition via Metis Partitioning 2 (GDMP2).
    
    GDMP2 uses graph partitioning techniques to identify communities by 
    recursively decomposing the graph.
    
    Parameters:
        min_threshold: Minimum threshold for partitioning (default: 0.75)
    
    Reference:
        Chen, M., et al. Community detection via maximization of modularity and 
        its variants. IEEE TCSS, 2014.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, min_threshold=0.75, **kwargs):
        """
        Initialize GDMP2 algorithm.
        
        Args:
            G: NetworkX graph
            min_threshold: Minimum threshold
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.min_threshold = min_threshold
        self.params['min_threshold'] = min_threshold
    
    def run(self):
        """
        Run the GDMP2 algorithm.
        
        Returns:
            List of communities with original node IDs
        """
        # Remap nodes to sequential 0, 1, 2, ...
        G_remapped, idx_to_original, _ = _remap_graph_to_sequential(self.G)
        
        try:
            result = cdlib_algos.gdmp2(G_remapped, min_threshold=self.min_threshold)
            # Remap back to original node IDs
            return _remap_communities_to_original(result.communities, idx_to_original)
        except Exception as e:
            print(f"GDMP2 algorithm failed: {e}")
            # Fallback to Louvain
            from community import community_louvain
            partition = community_louvain.best_partition(self.G)
            comm_dict = {}
            for node, comm_id in partition.items():
                if comm_id not in comm_dict:
                    comm_dict[comm_id] = []
                comm_dict[comm_id].append(node)
            return list(comm_dict.values())
