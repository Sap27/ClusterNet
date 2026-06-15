"""
Wrapper for Leading Eigenvector algorithm.
"""

import sys
import os

# Add parent directory to path to import original algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from leading_eigen import LeadingEigenAlgorithm, leading_eigen_vector
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('leading_eigenvector', aliases=['leading_eigen', 'newman_eigen'])
class LeadingEigenWrapper(BaseAlgorithm):
    """
    Leading Eigenvector method for community detection.
    
    Uses the eigenvector of the modularity matrix corresponding to the 
    largest positive eigenvalue to iteratively bisect the network.
    
    Implementation matches main.py leading_eigen_vector() function exactly:
    - Uses igraph.Graph.from_networkx()
    - Uses igraph's community_leading_eigenvector
    - Passes weights if weighted
    
    Parameters:
        weighted: Whether graph is weighted (default: True)
        directed: Whether graph is directed (default: False)
    
    Reference:
        Newman (2006). "Finding community structure in networks using the 
        eigenvectors of matrices"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, weighted=True, directed=False, **kwargs):
        """
        Initialize Leading Eigenvector algorithm.
        
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
        Run the Leading Eigenvector algorithm (matching main.py exactly).
        
        Returns:
            List of communities
        """
        algo = LeadingEigenAlgorithm(
            self.G,
            weighted=self.weighted,
            directed=self.directed
        )
        communities, runtime = algo.run()
        return communities
