"""
Wrapper for TeamCS algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from TeamCS import team_cs_community_detection
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
        max_workers: Number of parallel workers (default: -1 for all cores)
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, recursive=True, max_workers=-1, **kwargs):
        """
        Initialize TeamCS algorithm.
        
        Args:
            G: NetworkX graph
            recursive: Use recursive refinement
            max_workers: Number of parallel workers
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.recursive = recursive
        self.max_workers = max_workers
        
        self.params.update({
            'recursive': recursive,
            'max_workers': max_workers
        })
    
    def run(self):
        """
        Run the TeamCS algorithm.
        
        Returns:
            List of communities
        """
        communities = team_cs_community_detection(
            G=self.G,
            Recursive=self.recursive,
            max_workers=self.max_workers
        )
        
        return communities

