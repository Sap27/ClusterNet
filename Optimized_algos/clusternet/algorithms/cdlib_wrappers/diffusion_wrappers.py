"""
Diffusion-based community detection algorithm wrappers from cdlib.
"""

from cdlib import algorithms as cdlib_algos
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('der', aliases=['diffusion_entropy_reducer'])
class DERWrapper(BaseAlgorithm):
    """
    Diffusion Entropy Reducer (DER) for community detection.
    
    DER uses diffusion dynamics and entropy to identify communities by 
    finding partitions that minimize the entropy of diffusion.
    
    Parameters:
        walk_len: Length of random walks (default: 3)
        threshold: Threshold for community assignment (default: 0.00001)
    
    Reference:
        Kozdoba, M., & Mannor, S. (2013). Community detection via measure space 
        embedding. NIPS 2013.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, walk_len=3, threshold=0.00001, **kwargs):
        """
        Initialize DER algorithm.
        
        Args:
            G: NetworkX graph
            walk_len: Length of random walks
            threshold: Threshold for community assignment
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.walk_len = walk_len
        self.threshold = threshold
        self.params['walk_len'] = walk_len
        self.params['threshold'] = threshold
    
    def run(self):
        """
        Run the DER algorithm.
        
        Returns:
            List of communities
        """
        try:
            result = cdlib_algos.der(
                self.G,
                walk_len=self.walk_len,
                threshold=self.threshold
            )
            return result.communities
        except Exception as e:
            print(f"DER algorithm failed: {e}")
            # Fallback to label propagation (also diffusion-based)
            try:
                result = cdlib_algos.label_propagation(self.G)
                return result.communities
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
        k: Number of communities to find
    
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
        # Estimate k if not provided
        if k is None:
            self.k = max(2, int(G.number_of_nodes() ** 0.5 / 2))
        else:
            self.k = k
        self.params['k'] = self.k
    
    def run(self):
        """
        Run the Async Fluid algorithm.
        
        Returns:
            List of communities
        """
        try:
            result = cdlib_algos.async_fluid(self.G, k=self.k)
            return result.communities
        except Exception as e:
            print(f"Async Fluid algorithm failed: {e}")
            # Fallback to label propagation
            try:
                import networkx as nx
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

