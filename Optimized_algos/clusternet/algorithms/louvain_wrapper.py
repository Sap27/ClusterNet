"""
Wrapper for Louvain algorithm.
"""

import sys
import os

# Add parent directory to path to import original algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from louvain import LouvainAlgorithm
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('louvain', aliases=['louvain_method'])
class LouvainWrapper(BaseAlgorithm):
    """
    Louvain method for community detection.
    
    The Louvain method is a modularity-based algorithm that hierarchically
    optimizes the modularity measure. It handles directed/undirected and
    weighted/unweighted graphs.
    
    Parameters:
        resolution: Resolution parameter (default: 1.0)
            Higher values lead to more communities
    
    Reference:
        Blondel et al. (2008). "Fast unfolding of communities in large networks"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, resolution=1.0, **kwargs):
        """
        Initialize Louvain algorithm.
        
        Args:
            G: NetworkX graph
            resolution: Resolution parameter
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.resolution = resolution
        self.params['resolution'] = resolution
    
    def run(self):
        """
        Run the Louvain algorithm.
        
        Returns:
            List of communities
        """
        # Create instance of original algorithm
        louvain = LouvainAlgorithm(self.G, resolution=self.resolution)
        
        # Run algorithm
        result = louvain.run()
        
        # Check if result is already a list of communities or a partition dict
        if isinstance(result, list):
            # Already formatted as list of communities
            return result
        else:
            # Convert partition dict to list of lists
            communities = self._partition_to_communities(result)
            return communities
    
    def _partition_to_communities(self, partition):
        """Convert partition dictionary to list of lists."""
        comm_dict = {}
        for node, comm_id in partition.items():
            if comm_id not in comm_dict:
                comm_dict[comm_id] = []
            comm_dict[comm_id].append(node)
        
        return list(comm_dict.values())

