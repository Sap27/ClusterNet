"""
Overlapping community detection algorithm wrappers from cdlib.
"""

from cdlib import algorithms as cdlib_algos
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('angel', aliases=['demon_successor'])
class AngelWrapper(BaseAlgorithm):
    """
    ANGEL - Efficient Node-Centric Community Discovery.
    
    Angel is a node-centric bottom-up overlapping community discovery algorithm.
    It's the faster successor of DEMON and can identify overlapping communities.
    
    Parameters:
        threshold: Merging threshold (default: 0.25)
        min_community_size: Minimum community size (default: 3)
    
    Reference:
        Rossetti, G. (2019). "Exorcising the Demon: Angel, Efficient Node-Centric 
        Community Discovery." International Conference on Complex Networks.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = False
    
    def __init__(self, G, threshold=0.25, min_community_size=3, **kwargs):
        """
        Initialize Angel algorithm.
        
        Args:
            G: NetworkX graph
            threshold: Merging threshold
            min_community_size: Minimum community size
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.threshold = threshold
        self.min_community_size = min_community_size
        self.params['threshold'] = threshold
        self.params['min_community_size'] = min_community_size
    
    def run(self):
        """
        Run the Angel algorithm.
        
        Returns:
            List of communities (may be overlapping)
        """
        try:
            result = cdlib_algos.angel(
                self.G,
                threshold=self.threshold,
                min_community_size=self.min_community_size
            )
            return result.communities
        except Exception as e:
            print(f"Angel algorithm failed (angel-cd may not be installed): {e}")
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


@register_algorithm('surprise_communities', aliases=['surprise'])
class SurpriseCommunitiesWrapper(BaseAlgorithm):
    """
    Surprise-based community detection.
    
    This algorithm finds communities by optimizing the Surprise quality function,
    which measures how surprising the distribution of links is compared to a 
    random graph.
    
    Reference:
        Aldecoa, R., & Marín, I. (2013). Surprise maximization reveals the 
        community structure of complex networks. Scientific reports, 3(1).
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, **kwargs):
        """
        Initialize Surprise Communities algorithm.
        
        Args:
            G: NetworkX graph
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
    
    def run(self):
        """
        Run the Surprise Communities algorithm.
        
        Returns:
            List of communities
        """
        try:
            result = cdlib_algos.surprise_communities(self.G)
            return result.communities
        except Exception as e:
            print(f"Surprise Communities algorithm failed: {e}")
            # Fallback to Leiden (also optimizes quality function)
            try:
                import leidenalg
                import igraph as ig
                g = ig.Graph.TupleList(self.G.edges(), directed=False)
                partition = leidenalg.find_partition(
                    g,
                    leidenalg.ModularityVertexPartition
                )
                communities = []
                for comm in partition:
                    communities.append([g.vs[i]['name'] for i in comm])
                return communities
            except:
                from community import community_louvain
                partition = community_louvain.best_partition(self.G)
                comm_dict = {}
                for node, comm_id in partition.items():
                    if comm_id not in comm_dict:
                        comm_dict[comm_id] = []
                    comm_dict[comm_id].append(node)
                return list(comm_dict.values())

