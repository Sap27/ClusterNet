"""
ClusterNet: A Unified Framework for Community Detection

ClusterNet provides a consistent interface for running various community detection
algorithms on directed/undirected and weighted/unweighted graphs.

Example usage:
    >>> import networkx as nx
    >>> from clusternet import CommunityDetector
    >>> 
    >>> # Create a graph
    >>> G = nx.karate_club_graph()
    >>> 
    >>> # Detect communities using Louvain
    >>> detector = CommunityDetector(algorithm='louvain')
    >>> communities = detector.detect(G)
    >>> 
    >>> # Or use a specific algorithm directly
    >>> from clusternet.algorithms import LouvainAlgorithm
    >>> louvain = LouvainAlgorithm(G, resolution=1.0)
    >>> communities = louvain.run()
"""

__version__ = "1.0.0"
__author__ = "ClusterNet Team"
__all__ = [
    "CommunityDetector",
    "list_algorithms",
    "load_graph",
    "save_communities",
]

from clusternet.core import CommunityDetector
from clusternet.utils.graph_utils import load_graph, save_communities
from clusternet.registry import list_algorithms

# Import algorithms to trigger registration
import clusternet.algorithms

# Optional: Try to import GPU support
try:
    import cupy
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

