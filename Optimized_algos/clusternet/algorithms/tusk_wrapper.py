"""
Wrapper for Tusk algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tusk import tusk_clustering
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('tusk', aliases=['tusk_dmi'])
class TuskWrapper(BaseAlgorithm):
    """
    Tusk algorithm for community detection.
    
    Tusk uses a combination of network transformation and clustering
    methods to detect communities effectively in complex networks.
    
    Parameters:
        k_components: Target number of components/clusters (default: 20)
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, k_components=20, **kwargs):
        """
        Initialize Tusk algorithm.
        
        Args:
            G: NetworkX graph
            k_components: Number of components (default: 20)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.k_components = k_components
        self.params['k_components'] = k_components
    
    def run(self):
        """
        Run the Tusk algorithm.
        
        Returns:
            List of communities
        """
        communities = tusk_clustering(
            input_data=self.G,
            k_components=self.k_components
        )
        
        return communities

