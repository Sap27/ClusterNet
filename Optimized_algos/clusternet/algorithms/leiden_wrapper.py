"""
Wrapper for Leiden algorithm using leidenalg library.
"""

import networkx as nx
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('leiden')
class LeidenWrapper(BaseAlgorithm):
    """
    Leiden algorithm for community detection.
    
    The Leiden algorithm is an improved version of Louvain that guarantees
    well-connected communities. It includes a refinement phase that prevents
    disconnected communities.
    
    This wrapper uses the optimized leidenalg library (C++ implementation).
    
    Parameters:
        resolution: Resolution parameter (default: 1.0)
                    Higher = more smaller communities
                    Lower = fewer larger communities
    
    Reference:
        Traag et al. (2019). "From Louvain to Leiden: guaranteeing well-connected communities"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, resolution=1.0, **kwargs):
        """
        Initialize Leiden algorithm.
        
        Args:
            G: NetworkX graph
            resolution: Resolution parameter (default: 1.0)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.resolution = resolution
        self.params.update({
            'resolution': resolution
        })
    
    def run(self):
        """
        Run the Leiden algorithm using leidenalg.
        
        Returns:
            List of communities
        """
        try:
            import leidenalg
            import igraph as ig
            
            # Convert networkx to igraph
            if self.G.is_directed():
                G_undirected = self.G.to_undirected()
            else:
                G_undirected = self.G
            
            # Create igraph graph
            g = ig.Graph.from_networkx(G_undirected)
            
            # Check for weights
            weights = None
            if 'weight' in g.es.attributes():
                weights = g.es['weight']
            
            # Run Leiden with resolution parameter
            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights=weights,
                resolution_parameter=self.resolution
            )
            
            # Convert to list of communities
            # Get node names from igraph
            if '_nx_name' in g.vs.attributes():
                names = g.vs['_nx_name']
            else:
                names = list(range(g.vcount()))
            
            communities = []
            for comm in partition:
                communities.append([names[i] for i in comm])
            
            return communities
            
        except ImportError:
            # Fallback to cdlib if leidenalg not available
            try:
                from cdlib import algorithms as cdlib_algos
                result = cdlib_algos.leiden(self.G)
                return result.communities
            except Exception as e:
                print(f"Leiden algorithm failed: {e}")
                # Ultimate fallback to Louvain
                from community import community_louvain
                partition = community_louvain.best_partition(self.G)
                comm_dict = {}
                for node, comm_id in partition.items():
                    if comm_id not in comm_dict:
                        comm_dict[comm_id] = []
                    comm_dict[comm_id].append(node)
                return list(comm_dict.values())
