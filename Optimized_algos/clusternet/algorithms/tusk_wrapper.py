"""
Wrapper for Tusk algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tusk import tusk_community_detection
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('tusk', aliases=['tusk_dmi'])
class TuskWrapper(BaseAlgorithm):
    """
    Tusk algorithm for community detection.
    
    Tusk uses a combination of network transformation and clustering
    methods to detect communities effectively in complex networks.
    
    Parameters:
        num_com: Target number of communities (default: auto-detect)
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, num_com=None, **kwargs):
        """
        Initialize Tusk algorithm.
        
        Args:
            G: NetworkX graph
            num_com: Number of communities (None for auto-detection)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.num_com = num_com
        self.params['num_com'] = num_com
    
    def run(self):
        """
        Run the Tusk algorithm.
        
        Returns:
            List of communities
        """
        communities = tusk_community_detection(
            G=self.G,
            num_com=self.num_com
        )
        
        return communities

