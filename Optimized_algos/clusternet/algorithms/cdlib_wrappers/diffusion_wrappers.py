"""
Diffusion-based community detection algorithm wrappers from cdlib.

NOTE: These wrappers remap nodes to 0-indexed before running cdlib algorithms
and remap results back to original node IDs to fix cdlib's node mapping issues.
"""

import networkx as nx
from cdlib import algorithms as cdlib_algos
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm
import numpy as np


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


@register_algorithm('der', aliases=['diffusion_entropy_reducer'])
class DERWrapper(BaseAlgorithm):
    """
    Diffusion Entropy Reducer (DER) for community detection.
    
    DER uses diffusion dynamics and entropy to identify communities by 
    finding partitions that minimize the entropy of diffusion.
    
    Note: DER tends to find few, large communities. For finer-grained
    communities, consider using other algorithms like label_propagation.
    
    Parameters:
        walk_len: Length of random walks (default: 3, recommended: 3-10)
                  Longer walks can capture larger-scale structure.
        threshold: Threshold for community assignment (default: 0.00001)
                   Lower values = stricter assignment.
        auto_tune: If True, automatically select walk_len based on network (default: True)
    
    Reference:
        Kozdoba, M., & Mannor, S. (2013). Community detection via measure space 
        embedding. NIPS 2013.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, walk_len=None, threshold=0.00001, auto_tune=True, **kwargs):
        """
        Initialize DER algorithm.
        
        Args:
            G: NetworkX graph
            walk_len: Length of random walks (if None, auto-select based on network)
            threshold: Threshold for community assignment
            auto_tune: Whether to auto-tune parameters
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        
        # Auto-tune walk_len based on network diameter/size
        if walk_len is None and auto_tune:
            n = G.number_of_nodes()
            avg_degree = 2 * G.number_of_edges() / n if n > 0 else 1
            # Longer walks for sparser networks
            self.walk_len = max(3, min(10, int(np.log(n) / np.log(avg_degree + 1))))
        else:
            self.walk_len = walk_len if walk_len is not None else 3
            
        self.threshold = threshold
        self.params['walk_len'] = self.walk_len
        self.params['threshold'] = threshold
    
    def run(self):
        """
        Run the DER algorithm.
        
        Returns:
            List of communities with original node IDs
        """
        # Remap nodes to sequential 0, 1, 2, ...
        G_remapped, idx_to_original, _ = _remap_graph_to_sequential(self.G)
        
        try:
            result = cdlib_algos.der(
                G_remapped,
                walk_len=self.walk_len,
                threshold=self.threshold
            )
            communities = result.communities
            
            # If DER returns too few communities, try with different parameters
            n = self.G.number_of_nodes()
            expected_min = max(2, int(np.sqrt(n) / 5))
            
            if len(communities) < expected_min:
                # Try longer walks
                for wl in [5, 7, 10]:
                    if wl != self.walk_len:
                        try:
                            result2 = cdlib_algos.der(G_remapped, walk_len=wl, threshold=self.threshold)
                            if len(result2.communities) > len(communities):
                                communities = result2.communities
                        except:
                            pass
            
            # Remap back to original node IDs
            return _remap_communities_to_original(communities, idx_to_original)
            
        except Exception as e:
            print(f"DER algorithm failed: {e}")
            # Fallback to label propagation (also diffusion-based)
            try:
                result = cdlib_algos.label_propagation(G_remapped)
                return _remap_communities_to_original(result.communities, idx_to_original)
            except:
                from community import community_louvain
                partition = community_louvain.best_partition(self.G)
                comm_dict = {}
                for node, comm_id in partition.items():
                    if comm_id not in comm_dict:
                        comm_dict[comm_id] = []
                    comm_dict[comm_id].append(node)
                return list(comm_dict.values())


@register_algorithm('async_fluid', aliases=['fluid', 'asynchronous_fluid'])
class AsyncFluidWrapper(BaseAlgorithm):
    """
    Asynchronous Fluid Communities algorithm.
    
    This algorithm is based on the simple idea of fluids (communities) interacting 
    in an environment (network), expanding and pushing each other. It's based on 
    propagation dynamics and is fast for large networks.
    
    Parameters:
        k: Number of communities to find (if None, auto-estimate)
        max_iter: Maximum iterations (default: 100)
    
    Reference:
        Parés F., Garcia-Gasulla D. et al. "Fluid Communities: A Competitive and 
        Highly Scalable Community Detection Algorithm". ComplexNetworks 2017.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = False
    
    def __init__(self, G, k=None, **kwargs):
        """
        Initialize Async Fluid algorithm.
        
        Args:
            G: NetworkX graph
            k: Number of communities (if None, estimate from network size)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        # Better k estimation using modularity-based heuristic
        if k is None:
            n = G.number_of_nodes()
            m = G.number_of_edges()
            # Use sqrt(n/2) as baseline, adjusted by density
            density = 2 * m / (n * (n - 1)) if n > 1 else 0
            self.k = max(2, int(np.sqrt(n / 2) * (1 + density)))
        else:
            self.k = k
        self.params['k'] = self.k
    
    def run(self):
        """
        Run the Async Fluid algorithm.
        
        Returns:
            List of communities with original node IDs
        """
        # Remap nodes to sequential 0, 1, 2, ...
        G_remapped, idx_to_original, _ = _remap_graph_to_sequential(self.G)
        
        try:
            result = cdlib_algos.async_fluid(G_remapped, k=self.k)
            # Remap back to original node IDs
            return _remap_communities_to_original(result.communities, idx_to_original)
        except Exception as e:
            print(f"Async Fluid algorithm failed: {e}")
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
