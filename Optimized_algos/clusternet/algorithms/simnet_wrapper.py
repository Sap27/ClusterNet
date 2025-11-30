"""
Wrapper for SimNet algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from simnet import run_simnet_pipeline
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('simnet', aliases=['similarity_network'])
class SimNetWrapper(BaseAlgorithm):
    """
    SimNet algorithm for community detection.
    
    SimNet uses network denoising followed by hierarchical clustering
    to detect communities. It combines spectral methods with agglomerative
    clustering for robust community detection.
    
    Parameters:
        initial_clusters: Initial number of clusters (default: 28)
        cluster_size_threshold: Maximum cluster size before subdivision (default: 50)
        denoise_lambda: Denoising parameter lambda (default: 1.0)
        denoise_beta: Denoising parameter beta (default: 0.1)
        verbose: Print progress information (default: True)
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, 
                 initial_clusters=28,
                 cluster_size_threshold=50,
                 denoise_lambda=1.0,
                 denoise_beta=0.1,
                 verbose=True,
                 **kwargs):
        """
        Initialize SimNet algorithm.
        
        Args:
            G: NetworkX graph
            initial_clusters: Initial number of clusters
            cluster_size_threshold: Max cluster size before subdivision
            denoise_lambda: Denoising parameter
            denoise_beta: Denoising parameter
            verbose: Print progress
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.initial_clusters = initial_clusters
        self.cluster_size_threshold = cluster_size_threshold
        self.denoise_lambda = denoise_lambda
        self.denoise_beta = denoise_beta
        self.verbose = verbose
        
        self.params.update({
            'initial_clusters': initial_clusters,
            'cluster_size_threshold': cluster_size_threshold,
            'denoise_lambda': denoise_lambda,
            'denoise_beta': denoise_beta
        })
    
    def run(self):
        """
        Run the SimNet algorithm.
        
        Returns:
            List of communities
        """
        communities = run_simnet_pipeline(
            G=self.G,
            initial_clusters=self.initial_clusters,
            cluster_size_threshold=self.cluster_size_threshold,
            denoise_lambda=self.denoise_lambda,
            denoise_beta=self.denoise_beta,
            verbose=self.verbose
        )
        
        return communities

