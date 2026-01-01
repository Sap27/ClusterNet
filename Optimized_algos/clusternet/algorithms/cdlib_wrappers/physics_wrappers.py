"""
Physics-inspired community detection algorithm wrappers from cdlib.
"""

from cdlib import algorithms as cdlib_algos
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('cpm', aliases=['constant_potts_model'])
class CPMWrapper(BaseAlgorithm):
    """
    Constant Potts Model (CPM) for community detection.
    
    CPM uses the Potts model from statistical physics with a resolution parameter
    to control community granularity. Similar to Leiden but with different optimization.
    
    Parameters:
        resolution_parameter: Resolution parameter (default: 1.0)
    
    Reference:
        Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011). Narrow scope for 
        resolution-limit-free community detection. Physical Review E, 84(016114).
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, resolution_parameter=1.0, **kwargs):
        """
        Initialize CPM algorithm.
        
        Args:
            G: NetworkX graph
            resolution_parameter: Resolution parameter
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.resolution_parameter = resolution_parameter
        self.params['resolution_parameter'] = resolution_parameter
    
    def run(self):
        """
        Run the CPM algorithm.
        
        Returns:
            List of communities
        """
        try:
            result = cdlib_algos.cpm(self.G, resolution_parameter=self.resolution_parameter)
            return result.communities
        except Exception as e:
            print(f"CPM algorithm failed: {e}")
            # Fallback to Leiden with same resolution
            try:
                import leidenalg
                import igraph as ig
                # Convert to igraph
                g = ig.Graph.TupleList(self.G.edges(), directed=self.directed)
                partition = leidenalg.find_partition(
                    g, 
                    leidenalg.CPMVertexPartition,
                    resolution_parameter=self.resolution_parameter
                )
                communities = []
                for comm in partition:
                    communities.append([g.vs[i]['name'] for i in comm])
                return communities
            except:
                from community import community_louvain
                partition = community_louvain.best_partition(
                    self.G.to_undirected(), 
                    resolution=self.resolution_parameter
                )
                comm_dict = {}
                for node, comm_id in partition.items():
                    if comm_id not in comm_dict:
                        comm_dict[comm_id] = []
                    comm_dict[comm_id].append(node)
                return list(comm_dict.values())


@register_algorithm('rb_pots', aliases=['reichardt_bornholdt_pots'])
class RBPotsWrapper(BaseAlgorithm):
    """
    Reichardt-Bornholdt Potts model for community detection.
    
    This algorithm optimizes a Potts model energy function for community detection
    in weighted networks.
    
    Parameters:
        resolution_parameter: Resolution parameter (default: 1.0)
        weights: Edge weights (default: None, uses 'weight' attribute if present)
    
    Reference:
        Reichardt, J., & Bornholdt, S. (2006). Statistical mechanics of community 
        detection. Physical Review E, 74(016110).
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, resolution_parameter=1.0, weights=None, **kwargs):
        """
        Initialize RB Potts algorithm.
        
        Args:
            G: NetworkX graph
            resolution_parameter: Resolution parameter
            weights: Edge weights
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.resolution_parameter = resolution_parameter
        self.weights = weights
        self.params['resolution_parameter'] = resolution_parameter
        self.params['weights'] = weights
    
    def run(self):
        """
        Run the RB Potts algorithm.
        
        Returns:
            List of communities
        """
        try:
            result = cdlib_algos.rb_pots(
                self.G,
                resolution_parameter=self.resolution_parameter,
                weights=self.weights
            )
            return result.communities
        except Exception as e:
            print(f"RB Potts algorithm failed: {e}")
            # Fallback to Leiden
            try:
                import leidenalg
                import igraph as ig
                g = ig.Graph.TupleList(self.G.edges(), directed=self.directed)
                partition = leidenalg.find_partition(
                    g,
                    leidenalg.RBConfigurationVertexPartition,
                    resolution_parameter=self.resolution_parameter
                )
                communities = []
                for comm in partition:
                    communities.append([g.vs[i]['name'] for i in comm])
                return communities
            except:
                from community import community_louvain
                partition = community_louvain.best_partition(self.G.to_undirected())
                comm_dict = {}
                for node, comm_id in partition.items():
                    if comm_id not in comm_dict:
                        comm_dict[comm_id] = []
                    comm_dict[comm_id].append(node)
                return list(comm_dict.values())


@register_algorithm('rber_pots', aliases=['reichardt_bornholdt_er_pots'])
class RBERPotsWrapper(BaseAlgorithm):
    """
    Reichardt-Bornholdt Erdos-Renyi Potts model for community detection.
    
    This algorithm optimizes a Potts model energy function assuming an 
    Erdos-Renyi random graph null model.
    
    Parameters:
        resolution_parameter: Resolution parameter (default: 1.0)
        weights: Edge weights (default: None, uses 'weight' attribute if present)
        node_sizes: Node sizes for aggregate graphs (default: None)
    
    Reference:
        Reichardt, J., & Bornholdt, S. (2006). Statistical mechanics of community 
        detection. Physical Review E, 74(016110).
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, resolution_parameter=1.0, weights=None, node_sizes=None, **kwargs):
        """
        Initialize RBER Potts algorithm.
        
        Args:
            G: NetworkX graph
            resolution_parameter: Resolution parameter
            weights: Edge weights
            node_sizes: Node sizes
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.resolution_parameter = resolution_parameter
        self.weights = weights
        self.node_sizes = node_sizes
        self.params['resolution_parameter'] = resolution_parameter
        self.params['weights'] = weights
        self.params['node_sizes'] = node_sizes
    
    def run(self):
        """
        Run the RBER Potts algorithm.
        
        Returns:
            List of communities
        """
        try:
            result = cdlib_algos.rber_pots(
                self.G,
                resolution_parameter=self.resolution_parameter,
                weights=self.weights,
                node_sizes=self.node_sizes
            )
            return result.communities
        except Exception as e:
            print(f"RBER Potts algorithm failed: {e}")
            # Fallback to RB Pots
            try:
                result = cdlib_algos.rb_pots(
                    self.G,
                    resolution_parameter=self.resolution_parameter,
                    weights=self.weights
                )
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


@register_algorithm('surprise_communities', aliases=['surprise'])
class SurpriseCommunitiesWrapper(BaseAlgorithm):
    """
    Surprise-based community detection.
    
    This algorithm finds communities by optimizing the Surprise quality function,
    which measures how surprising the distribution of links is compared to a 
    random graph. Unlike modularity, Surprise is better at detecting small communities.
    
    Note: Returns DISJOINT partitions (not overlapping).
    
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
            List of communities (disjoint)
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

