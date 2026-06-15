"""
Physics-inspired community detection algorithm wrappers from cdlib.

NOTE: These wrappers remap nodes to 0-indexed before running cdlib algorithms
and remap results back to original node IDs to fix cdlib's node mapping issues.
"""

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


@register_algorithm('cpm', aliases=['constant_potts_model'])
class CPMWrapper(BaseAlgorithm):
    """
    Constant Potts Model (CPM) for community detection.
    
    CPM uses the Potts model from statistical physics with a resolution parameter
    to control community granularity.
    
    Parameters:
        resolution_parameter: Resolution parameter (default: 0.1)
            Lower values = fewer, larger communities
            Higher values = more, smaller communities
    
    Reference:
        Traag, V. A., Van Dooren, P., & Nesterov, Y. (2011). Narrow scope for 
        resolution-limit-free community detection. Physical Review E, 84(016114).
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, resolution_parameter=0.1, **kwargs):
        """
        Initialize CPM algorithm.
        
        Args:
            G: NetworkX graph
            resolution_parameter: Resolution parameter (default: 0.1)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.resolution_parameter = resolution_parameter
        self.params['resolution_parameter'] = resolution_parameter
    
    def run(self):
        """
        Run the CPM algorithm.
        
        Returns:
            List of communities with original node IDs
        """
        # Remap nodes to sequential 0, 1, 2, ...
        G_remapped, idx_to_original, _ = _remap_graph_to_sequential(self.G)
        
        try:
            result = cdlib_algos.cpm(G_remapped, resolution_parameter=self.resolution_parameter)
            # Remap communities back to original node IDs
            return _remap_communities_to_original(result.communities, idx_to_original)
        except Exception as e:
            print(f"CPM algorithm failed: {e}")
            # Fallback to Louvain
            from community import community_louvain
            partition = community_louvain.best_partition(
                self.G.to_undirected() if self.directed else self.G,
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
            List of communities with original node IDs
        """
        # Remap nodes to sequential 0, 1, 2, ...
        G_remapped, idx_to_original, _ = _remap_graph_to_sequential(self.G)
        
        try:
            result = cdlib_algos.rb_pots(
                G_remapped,
                resolution_parameter=self.resolution_parameter,
                weights=self.weights
            )
            # Remap communities back to original node IDs
            return _remap_communities_to_original(result.communities, idx_to_original)
        except Exception as e:
            print(f"RB Potts algorithm failed: {e}")
            # Fallback to Louvain
            from community import community_louvain
            partition = community_louvain.best_partition(
                self.G.to_undirected() if self.directed else self.G,
                resolution=self.resolution_parameter
            )
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
            List of communities with original node IDs
        """
        # Remap nodes to sequential 0, 1, 2, ...
        G_remapped, idx_to_original, _ = _remap_graph_to_sequential(self.G)
        
        try:
            result = cdlib_algos.rber_pots(
                G_remapped,
                resolution_parameter=self.resolution_parameter,
                weights=self.weights,
                node_sizes=self.node_sizes
            )
            # Remap communities back to original node IDs
            return _remap_communities_to_original(result.communities, idx_to_original)
        except Exception as e:
            print(f"RBER Potts algorithm failed: {e}")
            # Fallback to RB Pots with remapping
            try:
                result = cdlib_algos.rb_pots(
                    G_remapped,
                    resolution_parameter=self.resolution_parameter,
                    weights=self.weights
                )
                return _remap_communities_to_original(result.communities, idx_to_original)
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
            List of communities (disjoint) with original node IDs
        """
        # Remap nodes to sequential 0, 1, 2, ...
        G_remapped, idx_to_original, _ = _remap_graph_to_sequential(self.G)
        
        try:
            result = cdlib_algos.surprise_communities(G_remapped)
            # Remap communities back to original node IDs
            return _remap_communities_to_original(result.communities, idx_to_original)
        except Exception as e:
            print(f"Surprise Communities algorithm failed: {e}")
            # Fallback to Louvain
            from community import community_louvain
            partition = community_louvain.best_partition(self.G)
            comm_dict = {}
            for node, comm_id in partition.items():
                if comm_id not in comm_dict:
                    comm_dict[comm_id] = []
                comm_dict[comm_id].append(node)
            return list(comm_dict.values())
