"""
Wrapper for SimNet algorithm with GPU support.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm

# Try to import GPU version first, fall back to CPU version
try:
    from simnet_gpu import run_simnet_pipeline_gpu, GPU_AVAILABLE
    USE_GPU_VERSION = True
except ImportError:
    from simnet import run_simnet_pipeline
    USE_GPU_VERSION = False
    GPU_AVAILABLE = False


@register_algorithm('simnet', aliases=['similarity_network'])
class SimNetWrapper(BaseAlgorithm):
    """
    SimNet algorithm for community detection with GPU support.
    
    SimNet uses network denoising followed by hierarchical clustering
    to detect communities. It combines spectral methods with agglomerative
    clustering for robust community detection.
    
    GPU acceleration (CuPy) is used automatically if available for the
    computationally expensive denoising step.
    
    Parameters:
        initial_clusters: Initial number of clusters (default: 28)
        cluster_size_threshold: Maximum cluster size before subdivision (default: 50)
        denoise_lambda: Denoising parameter lambda (default: 1.0)
        denoise_beta: Denoising parameter beta (default: 0.1)
        use_gpu: Use GPU if available (default: 'auto' - True if GPU detected)
        verbose: Print progress information (default: True)
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, 
                 initial_clusters=28,
                 cluster_size_threshold=50,
                 denoise_lambda=1.0,
                 denoise_beta=0.1,
                 use_gpu='auto',
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
            use_gpu: 'auto' (default), True, or False
            verbose: Print progress
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.initial_clusters = initial_clusters
        self.cluster_size_threshold = cluster_size_threshold
        self.denoise_lambda = denoise_lambda
        self.denoise_beta = denoise_beta
        self.verbose = verbose
        
        # Determine GPU usage
        if use_gpu == 'auto':
            self.use_gpu = GPU_AVAILABLE
        else:
            self.use_gpu = use_gpu and GPU_AVAILABLE
        
        self.params.update({
            'initial_clusters': initial_clusters,
            'cluster_size_threshold': cluster_size_threshold,
            'denoise_lambda': denoise_lambda,
            'denoise_beta': denoise_beta,
            'use_gpu': self.use_gpu
        })
    
    def run(self):
        """
        Run the SimNet algorithm (GPU-accelerated if available).
        
        Returns:
            List of communities
        """
        if USE_GPU_VERSION:
            # Use GPU version with automatic CPU fallback
            communities = run_simnet_pipeline_gpu(
                G=self.G,
                initial_clusters=self.initial_clusters,
                cluster_size_threshold=self.cluster_size_threshold,
                denoise_lambda=self.denoise_lambda,
                denoise_beta=self.denoise_beta,
                use_gpu=self.use_gpu,
                verbose=self.verbose
            )
        else:
            # Use CPU-only version
            communities = run_simnet_pipeline(
                G=self.G,
                initial_clusters=self.initial_clusters,
                cluster_size_threshold=self.cluster_size_threshold,
                denoise_lambda=self.denoise_lambda,
                denoise_beta=self.denoise_beta,
                verbose=self.verbose
            )
        
        return communities

