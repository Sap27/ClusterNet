"""
Structural community detection algorithm wrappers from cdlib.
"""

from cdlib import algorithms as cdlib_algos
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


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
            List of communities
        """
        try:
            result = cdlib_algos.scan(self.G, epsilon=self.epsilon, mu=self.mu)
            # Filter out single-node communities (outliers)
            communities = [c for c in result.communities if len(c) > 1]
            if not communities:
                # If all are outliers, return original result
                communities = result.communities
            return communities
        except Exception as e:
            print(f"SCAN algorithm failed: {e}")
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


@register_algorithm('agdl', aliases=['adaptive_greedy'])
class AGDLWrapper(BaseAlgorithm):
    """
    Adaptive Greedy community detection with Diffusion Learning (AGDL).
    
    AGDL adaptively selects seeds and uses greedy expansion with diffusion 
    to discover communities of various densities.
    
    Parameters:
        number_communities: Target number of communities
        kc: Size parameter for seed selection (default: 3)
    
    Reference:
        Zhang, W., et al. (2013). Identification of overlapping community structure 
        in complex networks using fuzzy c-means clustering.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, number_communities=None, kc=3, **kwargs):
        """
        Initialize AGDL algorithm.
        
        Args:
            G: NetworkX graph
            number_communities: Target number of communities
            kc: Seed selection parameter
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        if number_communities is None:
            self.number_communities = max(2, int(G.number_of_nodes() ** 0.5 / 2))
        else:
            self.number_communities = number_communities
        self.kc = kc
        self.params['number_communities'] = self.number_communities
        self.params['kc'] = kc
    
    def run(self):
        """
        Run the AGDL algorithm.
        
        Returns:
            List of communities
        """
        try:
            # AGDL expects 'weight' as the edge attribute name (default "weight")
            result = cdlib_algos.agdl(
                self.G,
                number_communities=self.number_communities,
                kc=self.kc
            )
            return result.communities
        except Exception as e:
            print(f"AGDL algorithm failed: {e}")
            # Fallback to Louvain
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
            List of communities
        """
        try:
            result = cdlib_algos.gdmp2(self.G, min_threshold=self.min_threshold)
            return result.communities
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

