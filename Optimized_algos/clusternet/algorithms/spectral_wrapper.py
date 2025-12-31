"""
Wrapper for Spectral Clustering algorithm.
"""

import sys
import os

# Add parent directory to path to import original algorithms
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from spectral_clustering import SpectralClusteringAlgorithm, spectral_clustering
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('spectral', aliases=['spectral_clustering', 'spectral_modularity'])
class SpectralWrapper(BaseAlgorithm):
    """
    Spectral Clustering method for community detection.
    
    Uses eigenvalues and eigenvectors of graph matrices (Laplacian) to 
    partition nodes into communities.
    
    Implementation matches main.py spectral_clustering() function exactly:
    - Gets largest connected component
    - Uses sklearn SpectralClustering with affinity='precomputed'
    - Uses adjacency matrix from nx.adjacency_matrix()
    
    Parameters:
        n_clusters: Number of clusters (default: 30)
        n_components: Number of eigenvector components (default: 10)
        weighted: Whether graph is weighted (default: True)
        directed: Whether graph is directed (default: False)
    
    Reference:
        Ng, Jordan, Weiss (2002). "On Spectral Clustering: Analysis and an algorithm"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, n_clusters=30, n_components=10, 
                 weighted=True, directed=False, **kwargs):
        """
        Initialize Spectral Clustering algorithm.
        
        Args:
            G: NetworkX graph
            n_clusters: Number of clusters (default: 30)
            n_components: Number of eigenvector components (default: 10)
            weighted: Whether graph is weighted (default: True)
            directed: Whether graph is directed (default: False)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.n_clusters = int(n_clusters)
        self.n_components = int(n_components)
        self.weighted = weighted
        self.directed = directed
        
        self.params['n_clusters'] = self.n_clusters
        self.params['n_components'] = self.n_components
        self.params['weighted'] = weighted
        self.params['directed'] = directed
    
    def run(self):
        """
        Run the Spectral Clustering algorithm (matching main.py exactly).
        
        Returns:
            List of communities
        """
        algo = SpectralClusteringAlgorithm(
            self.G,
            weighted=self.weighted,
            directed=self.directed,
            n_clusters=self.n_clusters,
            n_components=self.n_components
        )
        communities, runtime = algo.run()
        return communities
