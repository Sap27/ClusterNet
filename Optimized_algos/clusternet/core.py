"""
Core community detection interface.
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Union
import warnings

from clusternet.registry import ALGORITHM_REGISTRY, get_algorithm_class
from clusternet.utils.graph_utils import preprocess_graph


class CommunityDetector:
    """
    Unified interface for community detection algorithms.
    
    This class provides a consistent API for running different community detection
    algorithms on various types of graphs (directed/undirected, weighted/unweighted).
    
    Example:
        >>> from clusternet import CommunityDetector
        >>> import networkx as nx
        >>> 
        >>> G = nx.karate_club_graph()
        >>> detector = CommunityDetector(algorithm='louvain', resolution=1.0)
        >>> communities = detector.detect(G)
        >>> print(f"Found {len(communities)} communities")
    """
    
    def __init__(self, algorithm: str = 'louvain', **kwargs):
        """
        Initialize the community detector.
        
        Args:
            algorithm: Name of the algorithm to use
            **kwargs: Algorithm-specific parameters
            
        Raises:
            ValueError: If algorithm is not recognized
        """
        self.algorithm_name = algorithm.lower()
        self.params = kwargs
        
        # Get algorithm class from registry
        self.algorithm_class = get_algorithm_class(self.algorithm_name)
        if self.algorithm_class is None:
            available = list(ALGORITHM_REGISTRY.keys())
            raise ValueError(
                f"Algorithm '{algorithm}' not found. "
                f"Available algorithms: {', '.join(available)}"
            )
    
    def detect(self, 
               G: Optional[nx.Graph] = None,
               input_file: Optional[str] = None,
               directed: bool = False,
               weighted: bool = True,
               output_file: Optional[str] = None) -> List[List]:
        """
        Detect communities in a graph.
        
        Args:
            G: NetworkX graph (if None, must provide input_file)
            input_file: Path to edge list file (if None, must provide G)
            directed: Whether to treat the graph as directed
            weighted: Whether to consider edge weights
            output_file: Optional path to save communities
            
        Returns:
            List of communities, where each community is a list of node IDs
            
        Raises:
            ValueError: If neither G nor input_file is provided
        """
        # Load graph if needed
        if G is None and input_file is None:
            raise ValueError("Must provide either 'G' or 'input_file'")
        
        if G is None:
            from clusternet.utils.graph_utils import load_graph
            G = load_graph(input_file, directed=directed, weighted=weighted)
        
        # Preprocess graph
        G = preprocess_graph(G, directed=directed, weighted=weighted)
        
        # Create algorithm instance
        algo_instance = self.algorithm_class(G, **self.params)
        
        # Run algorithm
        try:
            communities = algo_instance.run()
        except Exception as e:
            warnings.warn(f"Algorithm '{self.algorithm_name}' failed: {str(e)}")
            raise
        
        # Validate results
        if not algo_instance.validate_communities(communities):
            warnings.warn("Generated communities failed validation")
        
        # Save if requested
        if output_file:
            from clusternet.utils.graph_utils import save_communities
            save_communities(communities, output_file)
        
        return communities
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about the selected algorithm.
        
        Returns:
            Dictionary with algorithm metadata
        """
        return {
            'name': self.algorithm_name,
            'class': self.algorithm_class.__name__,
            'parameters': self.params,
            'supports_directed': getattr(self.algorithm_class, 'SUPPORTS_DIRECTED', True),
            'supports_weighted': getattr(self.algorithm_class, 'SUPPORTS_WEIGHTED', True),
        }
    
    def __repr__(self) -> str:
        return f"CommunityDetector(algorithm='{self.algorithm_name}', params={self.params})"


class BatchDetector:
    """
    Run multiple algorithms on the same graph for comparison.
    
    Example:
        >>> from clusternet import BatchDetector
        >>> import networkx as nx
        >>> 
        >>> G = nx.karate_club_graph()
        >>> detector = BatchDetector(algorithms=['louvain', 'leiden', 'infomap'])
        >>> results = detector.detect_all(G)
        >>> 
        >>> for algo, communities in results.items():
        >>>     print(f"{algo}: {len(communities)} communities")
    """
    
    def __init__(self, algorithms: List[str], **common_params):
        """
        Initialize batch detector.
        
        Args:
            algorithms: List of algorithm names to run
            **common_params: Common parameters for all algorithms
        """
        self.algorithms = algorithms
        self.common_params = common_params
        self.detectors = {}
        
        # Create detector for each algorithm
        for algo in algorithms:
            try:
                self.detectors[algo] = CommunityDetector(algorithm=algo, **common_params)
            except ValueError as e:
                warnings.warn(f"Skipping algorithm '{algo}': {str(e)}")
    
    def detect_all(self, 
                   G: Optional[nx.Graph] = None,
                   input_file: Optional[str] = None,
                   directed: bool = False,
                   weighted: bool = True) -> Dict[str, List[List]]:
        """
        Run all algorithms on the graph.
        
        Args:
            G: NetworkX graph (if None, must provide input_file)
            input_file: Path to edge list file
            directed: Whether to treat as directed
            weighted: Whether to consider weights
            
        Returns:
            Dictionary mapping algorithm names to their communities
        """
        results = {}
        
        for algo_name, detector in self.detectors.items():
            try:
                communities = detector.detect(
                    G=G, 
                    input_file=input_file,
                    directed=directed,
                    weighted=weighted
                )
                results[algo_name] = communities
                print(f"✓ {algo_name}: {len(communities)} communities")
            except Exception as e:
                print(f"✗ {algo_name}: Failed - {str(e)}")
                results[algo_name] = None
        
        return results
    
    def __repr__(self) -> str:
        return f"BatchDetector(algorithms={self.algorithms})"

