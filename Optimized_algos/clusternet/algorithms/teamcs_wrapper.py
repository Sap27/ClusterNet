"""
Wrapper for TeamCS algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from TeamCS import main as teamcs_main
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('teamcs', aliases=['team_cs'])
class TeamCSWrapper(BaseAlgorithm):
    """
    TeamCS algorithm for community detection.
    
    TeamCS uses a combination of Infomap and hierarchical refinement
    for detecting communities. It can operate in recursive mode for
    hierarchical community structure.
    
    Parameters:
        recursive: Whether to use recursive refinement (default: True)
        size_threshold: Clusters larger than this will be split (default: 100)
        min_size: Minimum cluster size (default: 3)
        percentile: Sparsification percentile (default: 40)
        n_jobs_outer: Parallel jobs for clusters (default: -1)
        n_jobs_inner: Parallel jobs within clusters (default: 1)
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, recursive=True, size_threshold=100, min_size=3, 
                 percentile=40, n_jobs_outer=-1, n_jobs_inner=1,weighted=True, directed=False, **kwargs):
        """
        Initialize TeamCS algorithm.
        
        Args:
            G: NetworkX graph
            recursive: Use recursive refinement
            size_threshold: Max cluster size before splitting
            min_size: Minimum cluster size
            percentile: Sparsification percentile
            n_jobs_outer: Parallel workers for clusters
            n_jobs_inner: Parallel workers within clusters
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.recursive = recursive
        self.size_threshold = size_threshold
        self.min_size = min_size
        self.percentile = percentile
        self.n_jobs_outer = n_jobs_outer
        self.n_jobs_inner = n_jobs_inner
        self.weighted = weighted
        self.directed = directed
        self.params.update({
            'recursive': recursive,
            'size_threshold': size_threshold,
            'min_size': min_size,
            'percentile': percentile,
            'n_jobs_outer': n_jobs_outer,
            'n_jobs_inner': n_jobs_inner
        })
    
    def run(self):
        """
        Run the TeamCS algorithm.
        
        Returns:
            List of communities
        """
        communities = teamcs_main(
            G=self.G,
            recursive=self.recursive,
            size_threshold=self.size_threshold,
            min_size=self.min_size,
            percentile=self.percentile,
            n_jobs_outer=self.n_jobs_outer,
            n_jobs_inner=self.n_jobs_inner,
            weighted=self.weighted,
            directed=self.directed
        )
        
        return communities

