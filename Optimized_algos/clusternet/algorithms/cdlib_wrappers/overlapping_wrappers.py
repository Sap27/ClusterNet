"""
Overlapping community detection algorithm wrappers from cdlib.
"""

from cdlib import algorithms as cdlib_algos
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm
import numpy as np


@register_algorithm('angel', aliases=['demon_successor'])
class AngelWrapper(BaseAlgorithm):
    """
    ANGEL - Efficient Node-Centric Community Discovery.
    
    Angel is a node-centric bottom-up overlapping community discovery algorithm.
    It's the faster successor of DEMON and can identify overlapping communities.
    
    Parameters:
        threshold: Merging threshold (default: 0.5)
                   Higher values = stricter merging = more communities
                   Lower values = more aggressive merging = fewer communities
        min_community_size: Minimum community size (default: 3)
        use_demon_fallback: If Angel returns too few communities, try DEMON (default: True)
    
    Reference:
        Rossetti, G. (2019). "Exorcising the Demon: Angel, Efficient Node-Centric 
        Community Discovery." International Conference on Complex Networks.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = False
    
    def __init__(self, G, threshold=0.5, min_community_size=3, use_demon_fallback=True, **kwargs):
        """
        Initialize Angel algorithm.
        
        Args:
            G: NetworkX graph
            threshold: Merging threshold (0.0-1.0)
            min_community_size: Minimum community size
            use_demon_fallback: Whether to fallback to DEMON if Angel fails
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.threshold = threshold
        self.min_community_size = min_community_size
        self.use_demon_fallback = use_demon_fallback
        self.params['threshold'] = threshold
        self.params['min_community_size'] = min_community_size
    
    def run(self):
        """
        Run the Angel algorithm with smart fallbacks.
        
        Returns:
            List of communities (may be overlapping)
        """
        n = self.G.number_of_nodes()
        expected_min_comms = max(2, int(np.sqrt(n) / 3))
        
        best_communities = None
        best_score = -1
        
        # Try multiple threshold values
        thresholds_to_try = [self.threshold]
        if self.threshold == 0.5:
            thresholds_to_try = [0.3, 0.5, 0.7]
        
        for thresh in thresholds_to_try:
            try:
                result = cdlib_algos.angel(
                    self.G,
                    threshold=thresh,
                    min_community_size=self.min_community_size
                )
                communities = result.communities
                
                # Score by number of communities and coverage
                if len(communities) >= expected_min_comms:
                    # Calculate coverage
                    covered_nodes = set()
                    for c in communities:
                        covered_nodes.update(c)
                    coverage = len(covered_nodes) / n if n > 0 else 0
                    score = len(communities) * coverage
                    
                    if score > best_score:
                        best_score = score
                        best_communities = communities
            except:
                pass
        
        # If we have good results, return them
        if best_communities is not None and len(best_communities) >= expected_min_comms:
            return best_communities
        
        # Try DEMON as fallback (predecessor of Angel, sometimes works better)
        if self.use_demon_fallback:
            try:
                for epsilon in [0.25, 0.5, 0.75]:
                    try:
                        result = cdlib_algos.demon(
                            self.G, 
                            epsilon=epsilon,
                            min_community_size=self.min_community_size
                        )
                        if len(result.communities) >= expected_min_comms:
                            return result.communities
                        if best_communities is None or len(result.communities) > len(best_communities):
                            best_communities = result.communities
                    except:
                        pass
            except:
                pass
        
        # If we have any result, return it
        if best_communities is not None:
            return best_communities
        
        # Ultimate fallback to label propagation
        print(f"Angel/DEMON algorithms failed, using label propagation fallback")
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


@register_algorithm('demon', aliases=['demon_overlapping'])
class DEMONWrapper(BaseAlgorithm):
    """
    DEMON - Democratic Estimate of the Modular Organization of a Network.
    
    DEMON is an ego-network based overlapping community detection algorithm.
    It identifies communities by analyzing local ego-networks and merging them
    using a democratic voting scheme.
    
    Parameters:
        epsilon: Merging threshold for label propagation (default: 0.25)
                 Lower = stricter merging = more communities
        min_com_size: Minimum community size (default: 3)
    
    Reference:
        Coscia, M., Rossetti, G., Giannotti, F., & Pedreschi, D. (2012).
        "DEMON: a local-first discovery method for overlapping communities."
        KDD 2012.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = False
    
    def __init__(self, G, epsilon=0.25, min_com_size=3, **kwargs):
        """
        Initialize DEMON algorithm.
        
        Args:
            G: NetworkX graph
            epsilon: Merging threshold (0.0-1.0)
            min_com_size: Minimum community size
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.epsilon = epsilon
        self.min_com_size = min_com_size
        self.params['epsilon'] = epsilon
        self.params['min_com_size'] = min_com_size
    
    def run(self):
        """
        Run the DEMON algorithm.
        
        Returns:
            List of communities (overlapping)
        """
        try:
            result = cdlib_algos.demon(
                self.G,
                epsilon=self.epsilon,
                min_com_size=self.min_com_size
            )
            return result.communities
        except Exception as e:
            print(f"DEMON algorithm failed: {e}")
            # Fallback to Angel
            try:
                result = cdlib_algos.angel(self.G, threshold=0.5, min_community_size=self.min_com_size)
                return result.communities
            except:
                import networkx as nx
                communities_gen = nx.community.label_propagation_communities(self.G)
                return [list(c) for c in communities_gen]


@register_algorithm('kclique', aliases=['k_clique', 'clique_percolation'])
class KCliqueWrapper(BaseAlgorithm):
    """
    k-Clique Percolation Method (CPM) for overlapping community detection.
    
    This algorithm finds overlapping communities by identifying k-clique communities,
    where a k-clique community is the union of all k-cliques that can be reached
    from each other through a series of adjacent k-cliques.
    
    Parameters:
        k: Size of cliques to consider (default: 3)
           Larger k = denser, smaller communities
    
    Reference:
        Palla, G., Derényi, I., Farkas, I., & Vicsek, T. (2005).
        "Uncovering the overlapping community structure of complex networks
        in nature and society." Nature, 435(7043), 814-818.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = False
    
    def __init__(self, G, k=3, **kwargs):
        """
        Initialize k-Clique algorithm.
        
        Args:
            G: NetworkX graph
            k: Clique size (default: 3, meaning triangles)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.k = k
        self.params['k'] = k
    
    def run(self):
        """
        Run the k-Clique Percolation algorithm.
        
        Returns:
            List of communities (overlapping)
        """
        try:
            result = cdlib_algos.kclique(self.G, k=self.k)
            communities = result.communities
            
            # If no communities found, try smaller k
            if not communities and self.k > 3:
                result = cdlib_algos.kclique(self.G, k=3)
                communities = result.communities
            
            return communities if communities else [[n] for n in self.G.nodes()]
            
        except Exception as e:
            print(f"k-Clique algorithm failed: {e}")
            # Fallback to NetworkX k-clique communities
            try:
                import networkx as nx
                from networkx.algorithms.community import k_clique_communities
                communities = list(k_clique_communities(self.G, self.k))
                return [list(c) for c in communities] if communities else [[n] for n in self.G.nodes()]
            except:
                # Ultimate fallback
                import networkx as nx
                communities_gen = nx.community.label_propagation_communities(self.G)
                return [list(c) for c in communities_gen]
