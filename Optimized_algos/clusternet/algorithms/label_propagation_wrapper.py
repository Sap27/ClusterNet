"""
Wrapper for Label Propagation algorithm.
"""

import sys
import os

# Add parent directory to path to import original algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from label_propagation import LabelPropagationAlgorithm, label_propogation
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('label_propagation', aliases=['lpa', 'label_prop'])
class LabelPropagationWrapper(BaseAlgorithm):
    """
    Label Propagation method for community detection.
    
    A near linear time algorithm for detecting community structure in networks.
    Each node is initialized with a unique label and at every step each node
    adopts the label that most of its neighbors currently have.
    
    Implementation matches main.py label_propogation() function exactly:
    - Converts directed to undirected via G.to_undirected()
    - Uses igraph.Graph.from_networkx()
    - Uses igraph's community_label_propagation
    - Passes weights if weighted
    
    Parameters:
        weighted: Whether graph is weighted (default: True)
        directed: Whether graph is directed (default: False)
    
    Reference:
        Raghavan, Albert, Kumara (2007). "Near linear time algorithm to detect 
        community structures in large-scale networks"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, weighted=True, directed=False, **kwargs):
        """
        Initialize Label Propagation algorithm.
        
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
        Run the Label Propagation algorithm (matching main.py exactly).
        
        Returns:
            List of communities
        """
        algo = LabelPropagationAlgorithm(
            self.G,
            weighted=self.weighted,
            directed=self.directed
        )
        communities, runtime = algo.run()
        return communities
