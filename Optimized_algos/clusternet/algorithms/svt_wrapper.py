"""
Wrapper for SVT (Singular Value Thresholding) algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tianle import tianle
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('svt', aliases=['tianle', 'svt_feature'])
class SVTWrapper(BaseAlgorithm):
    """
    SVT-based community detection algorithm.
    
    This algorithm uses Singular Value Thresholding (SVT) for feature extraction
    followed by hierarchical clustering to detect communities.
    
    Parameters:
        svd_k: Number of singular values to use (default: 50)
        n_clusters: Initial number of clusters (default: 50)
        module_size: Target module size (default: 40)
        n_neighbors: Number of neighbors for connectivity (default: 100)
    
    Reference:
        Based on matrix completion and spectral methods
    """
    
    SUPPORTS_DIRECTED = False  # Converts directed to undirected
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, 
                 svd_k=50, 
                 n_clusters=50, 
                 module_size=40,
                 n_neighbors=100,
                 **kwargs):
        """
        Initialize SVT algorithm.
        
        Args:
            G: NetworkX graph
            svd_k: Number of singular values
            n_clusters: Initial number of clusters
            module_size: Target module size
            n_neighbors: Number of neighbors for connectivity
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.svd_k = svd_k
        self.n_clusters = n_clusters
        self.module_size = module_size
        self.n_neighbors = n_neighbors
        
        self.params.update({
            'svd_k': svd_k,
            'n_clusters': n_clusters,
            'module_size': module_size,
            'n_neighbors': n_neighbors
        })
    
    def run(self):
        """
        Run the SVT algorithm.
        
        Returns:
            List of communities
        """
        # The tianle function expects a graph and parameters
        communities = tianle(
            output_filename='',  # Don't save to file
            G=self.G,
            directed=self.directed,
            svd_k=self.svd_k,
            n_clusters=self.n_clusters,
            module_size=self.module_size,
            n_neighbors=self.n_neighbors
        )
        
        return communities

