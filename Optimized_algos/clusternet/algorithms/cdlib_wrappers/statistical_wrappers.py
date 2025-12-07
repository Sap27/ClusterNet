"""
Statistical inference-based community detection algorithm wrappers from cdlib.
"""

from cdlib import algorithms as cdlib_algos
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm
import networkx as nx


@register_algorithm('em', aliases=['expectation_maximization'])
class EMWrapper(BaseAlgorithm):
    """
    Expectation-Maximization algorithm for community detection.
    
    This algorithm uses EM to fit a statistical model to the network structure.
    
    Parameters:
        k: Number of communities to find
    
    Reference:
        Newman, M. E., & Leicht, E. A. (2007). Mixture models and exploratory 
        analysis in networks. PNAS, 104(23), 9564-9569.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = False
    
    def __init__(self, G, k=None, **kwargs):
        """
        Initialize EM algorithm.
        
        Args:
            G: NetworkX graph
            k: Number of communities (if None, will try to auto-detect)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.k = k if k is not None else max(2, int(G.number_of_nodes() ** 0.5 / 2))
        self.params['k'] = self.k
    
    def run(self):
        """
        Run the EM algorithm.
        
        Returns:
            List of communities
        """
        try:
            result = cdlib_algos.em(self.G, k=self.k)
            return result.communities
        except Exception as e:
            print(f"EM algorithm failed: {e}")
            # Return each node as its own community as fallback
            return [[node] for node in self.G.nodes()]


@register_algorithm('sbm', aliases=['sbm_dl', 'stochastic_block_model'])
class SBMWrapper(BaseAlgorithm):
    """
    Stochastic Block Model inference using minimum description length.
    
    This algorithm uses graph-tool's efficient SBM inference based on 
    Bayesian inference and the minimum description length principle.
    
    Reference:
        Peixoto, T. P. (2014). Hierarchical block structures and high-resolution 
        model selection in large networks. Physical Review X, 4(011047).
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, **kwargs):
        """
        Initialize SBM algorithm.
        
        Args:
            G: NetworkX graph
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
    
    def run(self):
        """
        Run the SBM inference.
        
        Returns:
            List of communities
        """
        try:
            result = cdlib_algos.sbm_dl(self.G)
            return result.communities
        except Exception as e:
            print(f"SBM algorithm failed (graph-tool may not be installed): {e}")
            # Fallback to simple modularity
            from community import community_louvain
            partition = community_louvain.best_partition(self.G.to_undirected())
            comm_dict = {}
            for node, comm_id in partition.items():
                if comm_id not in comm_dict:
                    comm_dict[comm_id] = []
                comm_dict[comm_id].append(node)
            return list(comm_dict.values())


@register_algorithm('sbm_nested', aliases=['nested_sbm', 'sbm_dl_nested'])
class NestedSBMWrapper(BaseAlgorithm):
    """
    Nested Stochastic Block Model inference.
    
    This algorithm infers a hierarchical structure using nested SBM with 
    the minimum description length principle.
    
    Reference:
        Peixoto, T. P. (2014). Hierarchical block structures and high-resolution 
        model selection in large networks. Physical Review X, 4(011047).
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, **kwargs):
        """
        Initialize nested SBM algorithm.
        
        Args:
            G: NetworkX graph
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
    
    def run(self):
        """
        Run the nested SBM inference.
        
        Returns:
            List of communities (at the finest level of hierarchy)
        """
        try:
            result = cdlib_algos.sbm_dl_nested(self.G)
            return result.communities
        except Exception as e:
            print(f"Nested SBM algorithm failed (graph-tool may not be installed): {e}")
            # Fallback to regular SBM
            try:
                result = cdlib_algos.sbm_dl(self.G)
                return result.communities
            except:
                from community import community_louvain
                partition = community_louvain.best_partition(self.G.to_undirected())
                comm_dict = {}
                for node, comm_id in partition.items():
                    if comm_id not in comm_dict:
                        comm_dict[comm_id] = []
                    comm_dict[comm_id].append(node)
                return list(comm_dict.values())

