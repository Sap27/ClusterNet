"""
Perturbation Functions for Robustness Testing

Provides various perturbation methods for:
- Edge structure (random, targeted, community-mixing)
- Node features (mean shift, variance scaling, dropout)
- Biological-specific attacks (hub, pathway)
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx
from typing import List, Optional, Tuple, Union


# =========================================
# EDGE PERTURBATIONS
# =========================================

def perturb_edges_random(
    G: Union[nx.Graph, Data],
    remove_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Union[nx.Graph, Data]:
    """
    Randomly remove edges from the graph.
    
    Args:
        G: NetworkX graph or PyG Data
        remove_ratio: Fraction of edges to remove
        seed: Random seed
        
    Returns:
        Perturbed graph (same type as input)
    """
    if seed is not None:
        np.random.seed(seed)
    
    is_pyg = isinstance(G, Data)
    
    if is_pyg:
        G_nx = to_networkx(G, to_undirected=True)
    else:
        G_nx = G.copy()
    
    edges = list(G_nx.edges())
    n_remove = int(len(edges) * remove_ratio)
    
    indices = np.random.choice(len(edges), n_remove, replace=False)
    for idx in indices:
        G_nx.remove_edge(*edges[idx])
    
    if is_pyg:
        edge_index = torch.tensor(list(G_nx.edges())).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        result = G.clone()
        result.edge_index = edge_index
        return result
    
    return G_nx


def perturb_edges_targeted(
    G: Union[nx.Graph, Data],
    remove_ratio: float = 0.1,
    target: str = 'betweenness',
    seed: Optional[int] = None
) -> Union[nx.Graph, Data]:
    """
    Remove edges based on centrality (targeted attack).
    
    Args:
        G: NetworkX graph or PyG Data
        remove_ratio: Fraction of edges to remove
        target: Centrality measure ('betweenness', 'degree')
        seed: Random seed
        
    Returns:
        Perturbed graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    is_pyg = isinstance(G, Data)
    
    if is_pyg:
        G_nx = to_networkx(G, to_undirected=True)
    else:
        G_nx = G.copy()
    
    n_remove = int(G_nx.number_of_edges() * remove_ratio)
    
    for _ in range(n_remove):
        if G_nx.number_of_edges() == 0:
            break
        
        if target == 'betweenness':
            centrality = nx.edge_betweenness_centrality(G_nx)
        elif target == 'degree':
            # Product of endpoint degrees
            centrality = {
                (u, v): G_nx.degree(u) * G_nx.degree(v)
                for u, v in G_nx.edges()
            }
        else:
            centrality = {e: 1 for e in G_nx.edges()}
        
        # Remove highest centrality edge
        max_edge = max(centrality, key=centrality.get)
        G_nx.remove_edge(*max_edge)
    
    if is_pyg:
        edge_index = torch.tensor(list(G_nx.edges())).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        result = G.clone()
        result.edge_index = edge_index
        return result
    
    return G_nx


def perturb_edges_add_noise(
    G: Union[nx.Graph, Data],
    add_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Union[nx.Graph, Data]:
    """
    Add random spurious edges.
    
    Args:
        G: NetworkX graph or PyG Data
        add_ratio: Fraction of new edges to add
        seed: Random seed
        
    Returns:
        Perturbed graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    is_pyg = isinstance(G, Data)
    
    if is_pyg:
        G_nx = to_networkx(G, to_undirected=True)
    else:
        G_nx = G.copy()
    
    n = G_nx.number_of_nodes()
    n_add = int(G_nx.number_of_edges() * add_ratio)
    
    nodes = list(G_nx.nodes())
    added = 0
    max_attempts = n_add * 10
    
    for _ in range(max_attempts):
        if added >= n_add:
            break
        
        u, v = np.random.choice(nodes, 2, replace=False)
        if not G_nx.has_edge(u, v):
            G_nx.add_edge(u, v)
            added += 1
    
    if is_pyg:
        edge_index = torch.tensor(list(G_nx.edges())).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        result = G.clone()
        result.edge_index = edge_index
        return result
    
    return G_nx


def perturb_edges_community_mixing(
    G: Union[nx.Graph, Data],
    communities: List[List[int]],
    mix_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Union[nx.Graph, Data]:
    """
    Add cross-community edges to increase mixing.
    
    Args:
        G: NetworkX graph or PyG Data
        communities: List of communities
        mix_ratio: Fraction of new cross-community edges
        seed: Random seed
        
    Returns:
        Perturbed graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    is_pyg = isinstance(G, Data)
    
    if is_pyg:
        G_nx = to_networkx(G, to_undirected=True)
    else:
        G_nx = G.copy()
    
    n_add = int(G_nx.number_of_edges() * mix_ratio)
    
    if len(communities) < 2:
        return G if not is_pyg else G.clone()
    
    added = 0
    max_attempts = n_add * 10
    
    for _ in range(max_attempts):
        if added >= n_add:
            break
        
        # Pick two different communities
        c1_idx, c2_idx = np.random.choice(len(communities), 2, replace=False)
        
        # Pick random nodes
        u = np.random.choice(communities[c1_idx])
        v = np.random.choice(communities[c2_idx])
        
        if not G_nx.has_edge(u, v):
            G_nx.add_edge(u, v)
            added += 1
    
    if is_pyg:
        edge_index = torch.tensor(list(G_nx.edges())).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        result = G.clone()
        result.edge_index = edge_index
        return result
    
    return G_nx


# =========================================
# FEATURE PERTURBATIONS
# =========================================

def perturb_features_mean(
    features: Union[np.ndarray, torch.Tensor],
    shift: float = 1.0,
    seed: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Shift feature means by adding Gaussian noise.
    
    Args:
        features: Node features [N, F]
        shift: Mean of Gaussian noise
        seed: Random seed
        
    Returns:
        Perturbed features
    """
    if seed is not None:
        np.random.seed(seed)
    
    is_tensor = isinstance(features, torch.Tensor)
    
    if is_tensor:
        noise = torch.normal(
            mean=shift * torch.ones_like(features),
            std=torch.ones_like(features)
        )
        return features + noise
    else:
        noise = np.random.normal(loc=shift, scale=1.0, size=features.shape)
        return features + noise


def perturb_features_variance(
    features: Union[np.ndarray, torch.Tensor],
    scale: float = 2.0
) -> Union[np.ndarray, torch.Tensor]:
    """
    Scale feature variance.
    
    Args:
        features: Node features [N, F]
        scale: Variance scaling factor
        
    Returns:
        Perturbed features
    """
    is_tensor = isinstance(features, torch.Tensor)
    
    if is_tensor:
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        return centered * scale + mean
    else:
        mean = features.mean(axis=0, keepdims=True)
        centered = features - mean
        return centered * scale + mean


def perturb_features_dropout(
    features: Union[np.ndarray, torch.Tensor],
    drop_ratio: float = 0.1,
    seed: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Randomly mask features (dropout).
    
    Args:
        features: Node features [N, F]
        drop_ratio: Fraction of features to mask
        seed: Random seed
        
    Returns:
        Perturbed features
    """
    if seed is not None:
        np.random.seed(seed)
    
    is_tensor = isinstance(features, torch.Tensor)
    
    if is_tensor:
        mask = torch.rand_like(features) > drop_ratio
        return features * mask.float()
    else:
        mask = np.random.random(features.shape) > drop_ratio
        return features * mask


def perturb_features_gaussian(
    features: Union[np.ndarray, torch.Tensor],
    noise_scale: float = 0.1,
    seed: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Add Gaussian noise to features.
    
    Args:
        features: Node features [N, F]
        noise_scale: Standard deviation of noise
        seed: Random seed
        
    Returns:
        Perturbed features
    """
    if seed is not None:
        np.random.seed(seed)
    
    is_tensor = isinstance(features, torch.Tensor)
    
    if is_tensor:
        noise = torch.randn_like(features) * noise_scale
        return features + noise
    else:
        noise = np.random.randn(*features.shape) * noise_scale
        return features + noise


# =========================================
# BIOLOGICAL-SPECIFIC PERTURBATIONS
# =========================================

def perturb_hub_nodes(
    G: nx.Graph,
    remove_ratio: float = 0.1,
    hub_percentile: float = 0.9,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Remove edges from hub nodes (high-degree nodes).
    Simulates knockout experiments in biological networks.
    
    Args:
        G: NetworkX graph
        remove_ratio: Fraction of hub edges to remove
        hub_percentile: Percentile threshold for hub definition
        seed: Random seed
        
    Returns:
        Perturbed graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    G_perturbed = G.copy()
    
    # Identify hub nodes
    degrees = dict(G_perturbed.degree())
    threshold = np.percentile(list(degrees.values()), hub_percentile * 100)
    hubs = [n for n, d in degrees.items() if d >= threshold]
    
    # Collect hub edges
    hub_edges = []
    for hub in hubs:
        for neighbor in G_perturbed.neighbors(hub):
            if (hub, neighbor) not in hub_edges and (neighbor, hub) not in hub_edges:
                hub_edges.append((hub, neighbor))
    
    # Remove random subset
    n_remove = int(len(hub_edges) * remove_ratio)
    indices = np.random.choice(len(hub_edges), min(n_remove, len(hub_edges)), replace=False)
    
    for idx in indices:
        u, v = hub_edges[idx]
        if G_perturbed.has_edge(u, v):
            G_perturbed.remove_edge(u, v)
    
    return G_perturbed


def perturb_pathway(
    G: nx.Graph,
    pathway_nodes: List[int],
    remove_ratio: float = 0.5,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Disrupt a specific pathway by removing internal edges.
    Simulates pathway disruption experiments.
    
    Args:
        G: NetworkX graph
        pathway_nodes: Nodes belonging to the pathway
        remove_ratio: Fraction of pathway edges to remove
        seed: Random seed
        
    Returns:
        Perturbed graph
    """
    if seed is not None:
        np.random.seed(seed)
    
    G_perturbed = G.copy()
    pathway_set = set(pathway_nodes)
    
    # Find pathway internal edges
    pathway_edges = []
    for u, v in G_perturbed.edges():
        if u in pathway_set and v in pathway_set:
            pathway_edges.append((u, v))
    
    # Remove random subset
    n_remove = int(len(pathway_edges) * remove_ratio)
    if n_remove > 0 and pathway_edges:
        indices = np.random.choice(len(pathway_edges), min(n_remove, len(pathway_edges)), replace=False)
        for idx in indices:
            u, v = pathway_edges[idx]
            if G_perturbed.has_edge(u, v):
                G_perturbed.remove_edge(u, v)
    
    return G_perturbed

