"""
Base class for community detection algorithms.
"""

from abc import ABC, abstractmethod
import networkx as nx
from typing import List, Dict, Any, Optional


class BaseAlgorithm(ABC):
    """
    Abstract base class for all community detection algorithms.
    
    All algorithms should inherit from this class and implement the `run()` method.
    
    Attributes:
        G: NetworkX graph object
        directed: Whether the graph is directed
        weighted: Whether the graph is weighted
        params: Dictionary of algorithm-specific parameters
    """
    
    def __init__(self, G: nx.Graph, **kwargs):
        """
        Initialize the algorithm.
        
        Args:
            G: NetworkX graph
            **kwargs: Algorithm-specific parameters
        """
        self.G = G
        self.directed = G.is_directed()
        self.weighted = self._is_weighted(G)
        self.params = kwargs
        
        # Store original node labels for remapping
        self.original_nodes = list(G.nodes())
        
    def _is_weighted(self, G: nx.Graph) -> bool:
        """Check if graph has edge weights."""
        for _, _, data in G.edges(data=True):
            if 'weight' in data:
                return True
        return False
    
    @abstractmethod
    def run(self) -> List[List]:
        """
        Run the community detection algorithm.
        
        Returns:
            List of communities, where each community is a list of node IDs
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get algorithm information.
        
        Returns:
            Dictionary with algorithm metadata
        """
        return {
            'name': self.__class__.__name__,
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges(),
            'directed': self.directed,
            'weighted': self.weighted,
            'parameters': self.params
        }
    
    def validate_communities(self, communities: List[List]) -> bool:
        """
        Validate that communities are properly formed.
        
        Args:
            communities: List of communities to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not communities:
            return False
        
        # Check that all communities are non-empty
        if any(len(c) == 0 for c in communities):
            return False
        
        # Check that all nodes are valid
        all_nodes = set()
        for comm in communities:
            all_nodes.update(comm)
        
        return all_nodes.issubset(set(self.original_nodes))
    
    def preprocess_graph(self, G: nx.Graph) -> nx.Graph:
        """
        Preprocess graph for the algorithm.
        Default implementation returns the graph as-is.
        Subclasses can override for specific preprocessing needs.
        
        Args:
            G: Input graph
            
        Returns:
            Preprocessed graph
        """
        return G.copy()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nodes={self.G.number_of_nodes()}, edges={self.G.number_of_edges()})"

