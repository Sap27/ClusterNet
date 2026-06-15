"""
Wrapper for SCORE (Spectral Clustering On Ratios-of-Eigenvectors) algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from SCORE import SCOREAlgorithm
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('score', aliases=['spectral_score'])
class SCOREWrapper(BaseAlgorithm):
    """
    SCORE algorithm for community detection.
    
    SCORE (Spectral Clustering On Ratios-of-Eigenvectors) is particularly robust
    for directed graphs with high degree heterogeneity. It normalizes node degrees
    using ratios of leading eigenvectors.
    
    Parameters:
        n_clusters: Number of communities to detect (default: auto-detect)
    
    Reference:
        Jin, J. (2015). "Fast Community Detection by SCORE"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, n_clusters=None, **kwargs):
        """
        Initialize SCORE algorithm.
        
        Args:
            G: NetworkX graph
            n_clusters: Number of communities (None for auto-detection)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.n_clusters = n_clusters
        self.params['n_clusters'] = n_clusters
    
    def run(self):
        """
        Run the SCORE algorithm.
        
        Returns:
            List of communities
        """
        score = SCOREAlgorithm(self.G, n_clusters=self.n_clusters)
        communities = score.run()
        
        return communities

