"""
Wrapper for Walktrap algorithm.
"""

import sys
import os

# Add parent directory to path to import original algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from walktrap import WalktrapAlgorithm, walktrap
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('walktrap', aliases=['walktrap_method', 'random_walk'])
class WalktrapWrapper(BaseAlgorithm):
    """
    Walktrap method for community detection using random walks.
    
    Uses random walks to compute distances between vertices and then uses 
    hierarchical clustering to identify communities.
    
    Implementation matches main.py walktrap() function exactly:
    - Uses igraph.community_walktrap
    - For directed: Creates igraph.Graph(directed=True) and adds vertices/edges manually
    - For undirected: Uses igraph.Graph.from_networkx()
    
    Parameters:
        steps: Length of random walks (default: 10)
        weighted: Whether graph is weighted (default: True)
        directed: Whether graph is directed (default: False)
    
    Reference:
        Pons & Latapy (2005). "Computing Communities in Large Networks Using Random Walks"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, steps=10, weighted=True, directed=False, **kwargs):
        """
        Initialize Walktrap algorithm.
        
        Args:
            G: NetworkX graph
            steps: Length of random walks (default: 10)
            weighted: Whether graph is weighted (default: True)
            directed: Whether graph is directed (default: False)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.steps = int(steps)
        self.weighted = weighted
        self.directed = directed
        self.params['steps'] = self.steps
        self.params['weighted'] = weighted
        self.params['directed'] = directed
    
    def run(self):
        """
        Run the Walktrap algorithm (matching main.py exactly).
        
        Returns:
            List of communities
        """
        algo = WalktrapAlgorithm(
            self.G, 
            weighted=self.weighted, 
            directed=self.directed, 
            steps=self.steps
        )
        communities, runtime = algo.run()
        return communities
