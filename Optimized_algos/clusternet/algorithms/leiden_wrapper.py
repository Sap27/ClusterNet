"""
Wrapper for Leiden algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from leiden import LeidenAlgorithm
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('leiden')
class LeidenWrapper(BaseAlgorithm):
    """
    Leiden algorithm for community detection.
    
    The Leiden algorithm is an improved version of Louvain that guarantees
    well-connected communities. It includes a refinement phase that prevents
    disconnected communities.
    
    Parameters:
        resolution: Resolution parameter (default: 1.0)
        randomness: Randomness parameter theta (default: 0.01)
    
    Reference:
        Traag et al. (2019). "From Louvain to Leiden: guaranteeing well-connected communities"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, resolution=1.0, randomness=0.01, **kwargs):
        """
        Initialize Leiden algorithm.
        
        Args:
            G: NetworkX graph
            resolution: Resolution parameter
            randomness: Randomness parameter (theta)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.resolution = resolution
        self.randomness = randomness
        self.params.update({
            'resolution': resolution,
            'randomness': randomness
        })
    
    def run(self):
        """
        Run the Leiden algorithm.
        
        Returns:
            List of communities
        """
        leiden = LeidenAlgorithm(self.G, 
                                resolution=self.resolution,
                                randomness=self.randomness)
        
        result = leiden.run()
        
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

