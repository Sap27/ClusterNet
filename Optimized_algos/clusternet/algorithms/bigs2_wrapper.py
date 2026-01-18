"""
Wrapper for BiGS2 (Multi-stage SCORE) algorithm.

BiGS2 uses multi-stage spectral clustering to detect communities,
recursively splitting large communities until size constraints are met.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('bigs2', aliases=['bi_graph_segmentation', 'multi_stage_score'])
class BiGS2Wrapper(BaseAlgorithm):
    """
    BiGS2 (Multi-stage SCORE) algorithm for community detection.
    
    Uses spectral clustering with SCORE algorithm to recursively split
    large communities until all are within size constraints.
    
    Parameters:
        n_clusters: Initial number of clusters (default: auto-estimate)
        min_size: Minimum community size (default: 3)
        max_size: Maximum community size (default: 100)
        max_iter: Maximum refinement iterations (default: 15)
        m_subclusters: Number of subclusters when splitting (default: 5)
        use_gpu: Use GPU acceleration if available (default: False)
        gpu_threshold: Minimum matrix size for GPU (default: 1000)
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, n_clusters=None, min_size=3, max_size=100, 
                 max_iter=15, m_subclusters=5, use_gpu=False, 
                 gpu_threshold=1000, **kwargs):
        """
        Initialize BiGS2 algorithm.
        
        Args:
            G: NetworkX graph
            n_clusters: Initial number of clusters (None for auto-estimate)
            min_size: Minimum community size
            max_size: Maximum community size
            max_iter: Maximum refinement iterations
            m_subclusters: Number of subclusters for splitting
            use_gpu: Use GPU acceleration
            gpu_threshold: Minimum matrix size for GPU
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.n_clusters = n_clusters
        self.min_size = min_size
        self.max_size = max_size
        self.max_iter = max_iter
        self.m_subclusters = m_subclusters
        self.use_gpu = use_gpu
        self.gpu_threshold = gpu_threshold
        
        self.params.update({
            'n_clusters': n_clusters,
            'min_size': min_size,
            'max_size': max_size,
            'max_iter': max_iter,
            'm_subclusters': m_subclusters,
            'use_gpu': use_gpu,
            'gpu_threshold': gpu_threshold
        })
    
    def run(self):
        """
        Run the BiGS2 algorithm.
        
        Returns:
            List of communities
        """
        try:
            # Import the bigs2-hybrid module
            import importlib.util
            bigs2_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "bigs2-hybrid.py"
            )
            spec = importlib.util.spec_from_file_location("bigs2_hybrid", bigs2_path)
            bigs2_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bigs2_module)
            
            # Run clustering
            communities = bigs2_module.bigs2_clustering(
                G=self.G,
                n_clusters=self.n_clusters,
                min_size=self.min_size,
                max_size=self.max_size,
                max_iter=self.max_iter,
                m_subclusters=self.m_subclusters,
                use_gpu=self.use_gpu,
                gpu_threshold=self.gpu_threshold,
                verbose=False
            )
            
            return communities
            
        except Exception as e:
            print(f"Warning: BiGS2 algorithm failed: {e}")
            # Fallback: use simple SCORE
            try:
                from clusternet.algorithms.score_wrapper import SCOREWrapper
                wrapper = SCOREWrapper(self.G)
                return wrapper.run()
            except:
                # Ultimate fallback: all nodes as one community
                return [list(self.G.nodes())]
