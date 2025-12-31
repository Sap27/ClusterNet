"""
Wrapper for Spin Glass algorithm.
"""

import sys
import os

# Add parent directory to path to import original algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from spinglass import SpinGlassAlgorithm, spin_glass
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('spinglass', aliases=['spin_glass', 'potts_model'])
class SpinGlassWrapper(BaseAlgorithm):
    """
    Spin Glass method for community detection.
    
    Uses statistical mechanics approach where nodes are treated as spins
    and community structure emerges from energy minimization.
    
    Implementation matches main.py spin_glass() function exactly:
    - Converts directed to undirected via G.to_undirected()
    - Gets largest connected component
    - Uses igraph.Graph.from_networkx()
    - Uses igraph's community_spinglass with spins parameter
    
    NOTE: Requires connected graph. Automatically uses largest connected 
    component if graph is disconnected.
    
    Parameters:
        spins: Number of spins/max communities (default: 25)
        weighted: Whether graph is weighted (default: True)
        directed: Whether graph is directed (default: False)
    
    Reference:
        Reichardt & Bornholdt (2006). "Statistical mechanics of community detection"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    REQUIRES_CONNECTED = True
    
    def __init__(self, G, spins=25, weighted=True, directed=False, **kwargs):
        """
        Initialize Spin Glass algorithm.
        
        Args:
            G: NetworkX graph
            spins: Number of spins/max communities (default: 25)
            weighted: Whether graph is weighted (default: True)
            directed: Whether graph is directed (default: False)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.spins = int(spins)
        self.weighted = weighted
        self.directed = directed
        self.params['spins'] = self.spins
        self.params['weighted'] = weighted
        self.params['directed'] = directed
    
    def run(self):
        """
        Run the Spin Glass algorithm (matching main.py exactly).
        
        Returns:
            List of communities
        """
        algo = SpinGlassAlgorithm(
            self.G,
            weighted=self.weighted,
            directed=self.directed,
            spins=self.spins
        )
        communities, runtime = algo.run()
        return communities
