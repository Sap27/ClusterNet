"""
Wrapper for Fast Greedy (CNM) algorithm.
"""

import sys
import os

# Add parent directory to path to import original algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fastgreedy import FastGreedyAlgorithm, fast_greedy
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('fastgreedy', aliases=['fast_greedy', 'cnm', 'clauset_newman_moore'])
class FastGreedyWrapper(BaseAlgorithm):
    """
    Fast Greedy (Clauset-Newman-Moore) method for community detection.
    
    A hierarchical agglomerative algorithm that greedily optimizes modularity.
    
    Implementation matches main.py fast_greedy() function exactly:
    - Converts directed to undirected via G.to_undirected()
    - Uses igraph.Graph.from_networkx()
    - Uses igraph's community_fastgreedy
    - Passes weights if weighted
    
    Parameters:
        weighted: Whether graph is weighted (default: True)
        directed: Whether graph is directed (default: False)
    
    Reference:
        Clauset, Newman, Moore (2004). "Finding community structure in very large networks"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, weighted=True, directed=False, **kwargs):
        """
        Initialize Fast Greedy algorithm.
        
        Args:
            G: NetworkX graph
            weighted: Whether graph is weighted (default: True)
            directed: Whether graph is directed (default: False)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.weighted = weighted
        self.directed = directed
        self.params['weighted'] = weighted
        self.params['directed'] = directed
    
    def run(self):
        """
        Run the Fast Greedy algorithm (matching main.py exactly).
        
        Returns:
            List of communities
        """
        algo = FastGreedyAlgorithm(
            self.G,
            weighted=self.weighted,
            directed=self.directed
        )
        communities, runtime = algo.run()
        return communities
