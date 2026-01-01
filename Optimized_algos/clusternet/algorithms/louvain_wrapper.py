"""
Wrapper for Louvain algorithm using python-louvain library.
"""

import networkx as nx
from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('louvain', aliases=['louvain_method'])
class LouvainWrapper(BaseAlgorithm):
    """
    Louvain method for community detection.
    
    The Louvain method is a modularity-based algorithm that hierarchically
    optimizes the modularity measure.
    
    This wrapper uses the optimized python-louvain library (C implementation).
    
    Parameters:
        resolution: Resolution parameter (default: 1.0)
            Higher values lead to more communities
            Lower values lead to fewer, larger communities
    
    Reference:
        Blondel et al. (2008). "Fast unfolding of communities in large networks"
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, resolution=1.0, **kwargs):
        """
        Initialize Louvain algorithm.
        
        Args:
            G: NetworkX graph
            resolution: Resolution parameter (default: 1.0)
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.resolution = resolution
        self.params['resolution'] = resolution
    
    def run(self):
        """
        Run the Louvain algorithm using python-louvain.
        
        Returns:
            List of communities
        """
        try:
            from community import community_louvain
            
            # Handle directed graphs (convert to undirected)
            if self.G.is_directed():
                G_undirected = self.G.to_undirected()
            else:
                G_undirected = self.G
            
            # Get weights if available
            weight = 'weight' if nx.is_weighted(G_undirected) else None
            
            # Run Louvain with resolution parameter
            partition = community_louvain.best_partition(
                G_undirected,
                weight=weight,
                resolution=self.resolution,
                random_state=42  # For reproducibility
            )
            
            # Convert partition dict to list of communities
            comm_dict = {}
            for node, comm_id in partition.items():
                if comm_id not in comm_dict:
                    comm_dict[comm_id] = []
                comm_dict[comm_id].append(node)
            
            return list(comm_dict.values())
            
        except ImportError:
            # Fallback to pure Python implementation if python-louvain not installed
            print("Warning: python-louvain not installed. Using slower pure Python implementation.")
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            from louvain import LouvainAlgorithm
            
            louvain = LouvainAlgorithm(self.G, resolution=self.resolution)
            return louvain.run()
