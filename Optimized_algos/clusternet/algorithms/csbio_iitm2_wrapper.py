"""
Wrapper for CSBIO-IITM2 (Ensemble Louvain-based) algorithm.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm
import networkx as nx
import tempfile


@register_algorithm('csbio_iitm2', aliases=['ensemble_louvain', 'csbio'])
class CSBIOWrapper(BaseAlgorithm):
    """
    CSBIO-IITM2 Ensemble Louvain algorithm for community detection.
    
    This algorithm uses ensemble clustering with the Louvain method across
    multiple resolution parameters, followed by hierarchical clustering
    to produce robust communities.
    
    Features:
    - Memory-optimized implementation
    - Ensemble approach for stability
    - Works on weighted/unweighted graphs
    - Automatically converts directed to undirected
    
    Parameters:
        resolutions: List of resolution values to use (default: None = auto range 0.1-1.0)
        strength: Strength parameter for ensemble clustering (default: 2)
    
    Reference:
        Ensemble Louvain method with hierarchical refinement
    """
    
    SUPPORTS_DIRECTED = True  # Converts to undirected internally
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, resolutions=None, strength=2, **kwargs):
        """
        Initialize CSBIO-IITM2 algorithm.
        
        Args:
            G: NetworkX graph
            resolutions: List of resolution values (None for default range)
            strength: Strength parameter for ensemble clustering
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        self.resolutions = resolutions
        self.strength = strength
        
        self.params.update({
            'resolutions': resolutions,
            'strength': strength
        })
    
    def run(self):
        """
        Run the CSBIO-IITM2 ensemble algorithm.
        
        Returns:
            List of communities
        """
        # Import the algorithm functions
        from csbio_iitm2_memory_optimized import (
            calculate_modularity,
            creating_ensemble_matrix,
            ensemble_hierarchical_clustering
        )
        import gc
        
        print(f"Running CSBIO-IITM2 on graph with {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
        # Preprocess: Convert directed to undirected if needed
        if self.G.is_directed():
            print("  Converting directed graph to undirected...")
            G = self.G.to_undirected()
        else:
            G = self.G.copy()
        
        # Ensure graph has weights
        for u, v in G.edges():
            if 'weight' not in G[u][v]:
                G[u][v]['weight'] = 1.0
        
        # Store original nodes and create sequential mapping
        original_nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
        reverse_mapping = {idx: node for node, idx in node_mapping.items()}
        
        # Relabel graph
        relabeled_G = nx.relabel_nodes(G, node_mapping)
        del G, node_mapping
        gc.collect()
        
        # Define resolution range
        if self.resolutions is None:
            resolutions = [0.1 * i for i in range(1, 11)]  # 0.1 to 1.0
        else:
            resolutions = self.resolutions
        
        print(f"  Calculating modularity for {len(resolutions)} resolutions...")
        
        # Calculate modularity for each resolution
        modularity_results = {}
        for i, resolution in enumerate(resolutions):
            if i % 2 == 0:
                print(f"    Resolution {i+1}/{len(resolutions)}: {resolution:.2f}")
            
            partition, mod_values = calculate_modularity(relabeled_G, resolution)
            modularity_results[resolution] = (partition, mod_values)
            
            if i % 3 == 0:
                gc.collect()
        
        print("  Creating ensemble matrix...")
        ensemble_matrix = creating_ensemble_matrix(modularity_results)
        
        del modularity_results
        gc.collect()
        
        print("  Performing hierarchical clustering...")
        final_clusters = ensemble_hierarchical_clustering(ensemble_matrix, relabeled_G, strength=self.strength)
        
        del ensemble_matrix
        gc.collect()
        
        # Map back to original node IDs
        communities = [[int(reverse_mapping[node]) for node in cluster] 
                      for cluster in final_clusters]
        
        # Cleanup
        del relabeled_G, reverse_mapping, final_clusters
        gc.collect()
        
        print(f"  ✓ Found {len(communities)} communities")
        
        return communities

