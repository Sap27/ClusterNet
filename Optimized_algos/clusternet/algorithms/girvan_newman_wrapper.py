"""
Wrapper for Girvan-Newman algorithm.
"""

import sys
import os

# Add parent directory to path to import original algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from girvan_newman import GirvanNewmanAlgorithm, girvan_newman
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('girvan_newman', aliases=['gn', 'edge_betweenness'])
class GirvanNewmanWrapper(BaseAlgorithm):
    """
    Girvan-Newman method for community detection.
    
    A hierarchical divisive algorithm based on edge betweenness centrality.
    Repeatedly removes the edge with highest betweenness to reveal community structure.
    
    Implementation matches main.py girvan_newman() function exactly:
    - Converts directed to undirected via G.to_undirected()
    - Uses networkx's nx.community.girvan_newman
    - Returns first partition from generator (next(output))
    - Supports most_valuable_edge parameter
    
    NOTE: This algorithm can be slow on large graphs due to repeated
    betweenness centrality calculations.
    
    Parameters:
        weighted: Whether graph is weighted (default: True)
        directed: Whether graph is directed (default: False)
        most_valuable_edge: Function to determine edge to remove (default: None)
    
    Reference:
        Girvan & Newman (2002). "Community structure in social and biological networks"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, weighted=True, directed=False, most_valuable_edge=None, **kwargs):
        """
        Initialize Girvan-Newman algorithm.
        
        Args:
            G: NetworkX graph
            weighted: Whether graph is weighted (default: True)
            directed: Whether graph is directed (default: False)
            most_valuable_edge: Function to determine edge to remove (default: None)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.weighted = weighted
        self.directed = directed
        self.most_valuable_edge = most_valuable_edge
        self.params['weighted'] = weighted
        self.params['directed'] = directed
    
    def run(self):
        """
        Run the Girvan-Newman algorithm (matching main.py exactly).
        
        Returns:
            List of communities
        """
        algo = GirvanNewmanAlgorithm(
            self.G,
            weighted=self.weighted,
            directed=self.directed,
            most_valuable_edge=self.most_valuable_edge
        )
        communities, runtime = algo.run()
        return communities
