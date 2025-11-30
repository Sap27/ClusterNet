"""
Wrapper for BiGS2 (Bi-level Graph Segmentation) algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('bigs2', aliases=['bi_graph_segmentation'])
class BiGS2Wrapper(BaseAlgorithm):
    """
    BiGS2 (Bi-level Graph Segmentation) algorithm for community detection.
    
    BiGS2 is a hybrid algorithm that combines multiple strategies for
    detecting communities in complex networks.
    
    Parameters:
        min_size: Minimum community size (default: 3)
        max_size: Maximum community size (default: 100)
        max_iter: Maximum iterations (default: 100)
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, min_size=3, max_size=100, max_iter=100, **kwargs):
        """
        Initialize BiGS2 algorithm.
        
        Args:
            G: NetworkX graph
            min_size: Minimum community size
            max_size: Maximum community size
            max_iter: Maximum iterations
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.min_size = min_size
        self.max_size = max_size
        self.max_iter = max_iter
        
        self.params.update({
            'min_size': min_size,
            'max_size': max_size,
            'max_iter': max_iter
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
            spec = importlib.util.spec_from_file_location(
                "bigs2_hybrid",
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                           "bigs2-hybrid.py")
            )
            bigs2_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bigs2_module)
            
            communities = bigs2_module.bigs2_community_detection(
                G=self.G,
                min_size=self.min_size,
                max_size=self.max_size,
                Max_iter=self.max_iter
            )
            
            return communities
        except Exception as e:
            print(f"Warning: BiGS2 algorithm failed: {e}")
            # Fallback: return all nodes as one community
            return [list(self.G.nodes())]

