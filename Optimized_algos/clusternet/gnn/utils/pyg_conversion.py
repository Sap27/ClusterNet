"""
PyTorch Geometric conversion utilities.

Handles conversion between NetworkX graphs and PyG Data objects.
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
from typing import Optional, Union


def nx_to_pyg(
    G: nx.Graph,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None
) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data.
    
    Args:
        G: NetworkX graph
        features: Optional node features [N, F]
        labels: Optional node labels [N]
        
    Returns:
        PyG Data object
    """
    # Get edge index from networkx
    pyg_data = from_networkx(G)
    
    # Handle features
    if features is not None:
        x = torch.tensor(features, dtype=torch.float)
    elif hasattr(pyg_data, 'x') and pyg_data.x is not None:
        x = pyg_data.x.float()
    else:
        x = generate_features(G)
    
    # Ensure 2D
    if x.dim() == 1:
        x = x.unsqueeze(1)
    
    # Build data object
    data = Data(
        x=x,
        edge_index=pyg_data.edge_index,
        num_nodes=G.number_of_nodes()
    )
    
    # Add labels if provided
    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.long)
    
    # Add edge weights if present
    if G.is_weighted():
        weights = []
        for u, v in G.edges():
            w = G[u][v].get('weight', 1.0)
            weights.append(w)
        # Make bidirectional (PyG uses both directions)
        weights = weights + weights
        data.edge_attr = torch.tensor(weights, dtype=torch.float)
    
    return data


def pyg_to_nx(data: Data) -> nx.Graph:
    """
    Convert PyTorch Geometric Data to NetworkX graph.
    
    Args:
        data: PyG Data object
        
    Returns:
        NetworkX graph
    """
    return to_networkx(data, to_undirected=True)


def generate_features(
    G: nx.Graph,
    dim: int = 32,
    feature_type: str = 'structural'
) -> torch.Tensor:
    """
    Generate node features for a graph.
    
    Args:
        G: NetworkX graph
        dim: Feature dimension
        feature_type: Type of features ('structural', 'random', 'degree')
        
    Returns:
        Feature tensor [N, dim]
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}
    
    if feature_type == 'random':
        return torch.randn(n, dim)
    
    elif feature_type == 'degree':
        degrees = torch.tensor([G.degree(node) for node in nodes], dtype=torch.float)
        # Normalize and expand
        degrees = degrees / (degrees.max() + 1e-8)
        features = degrees.unsqueeze(1).expand(-1, dim)
        # Add noise for diversity
        features = features + 0.1 * torch.randn(n, dim)
        return features
    
    else:  # structural
        features = np.zeros((n, dim))
        
        # Degree (normalized)
        degrees = dict(G.degree())
        max_deg = max(degrees.values()) if degrees else 1
        for node, deg in degrees.items():
            features[node_idx[node], 0] = deg / max_deg
        
        # Clustering coefficient
        clustering = nx.clustering(G)
        for node, cc in clustering.items():
            features[node_idx[node], 1] = cc
        
        # PageRank
        try:
            pagerank = nx.pagerank(G)
            max_pr = max(pagerank.values())
            for node, pr in pagerank.items():
                features[node_idx[node], 2] = pr / max_pr
        except:
            pass
        
        # Eigenvector centrality
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=100)
            max_ev = max(eigenvector.values())
            for node, ev in eigenvector.items():
                features[node_idx[node], 3] = ev / max_ev
        except:
            pass
        
        # Random features for remaining dimensions
        features[:, 4:] = np.random.randn(n, dim - 4) * 0.1
        
        return torch.tensor(features, dtype=torch.float)









