"""
LFR Benchmark Generator with Perturbation Support

Generates LFR benchmark graphs with configurable parameters
and built-in perturbation capabilities.
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class LFRParams:
    """LFR benchmark parameters."""
    n: int = 1000                    # Number of nodes
    tau1: float = 2.0                # Power law exponent for degree
    tau2: float = 1.1                # Power law exponent for community size
    mu: float = 0.1                  # Mixing parameter
    average_degree: int = 25         # Average degree
    max_degree: Optional[int] = None # Max degree (default: 0.1*n)
    min_community: Optional[int] = None  # Min community size
    max_community: Optional[int] = None  # Max community size
    seed: Optional[int] = None


class LFRBenchmark:
    """
    LFR Benchmark Generator with perturbation support.
    
    Features:
    - Configurable LFR parameters
    - Built-in feature generation
    - Perturbation methods (edges, features)
    - PyTorch Geometric compatibility
    """
    
    def __init__(self, params: Optional[LFRParams] = None, **kwargs):
        """
        Initialize LFR benchmark.
        
        Args:
            params: LFRParams object or keyword arguments
        """
        if params is None:
            params = LFRParams(**kwargs)
        self.params = params
        
        # Set defaults based on n
        if self.params.max_degree is None:
            self.params.max_degree = int(0.1 * self.params.n)
        if self.params.min_community is None:
            self.params.min_community = max(10, int(0.05 * self.params.n))
        if self.params.max_community is None:
            self.params.max_community = int(0.2 * self.params.n)
        
        # Generate graph
        self.G = None
        self.communities = None
        self.labels = None
        self.features = None
        
    def generate(self, feat_dim: int = 32) -> Tuple[nx.Graph, List[List[int]], np.ndarray]:
        """
        Generate LFR benchmark graph.
        
        Args:
            feat_dim: Dimension of node features
            
        Returns:
            G: NetworkX graph
            communities: List of communities
            labels: Node label array
        """
        # Set seed if provided
        if self.params.seed is not None:
            np.random.seed(self.params.seed)
        
        # Generate LFR graph
        self.G = nx.generators.community.LFR_benchmark_graph(
            n=self.params.n,
            tau1=self.params.tau1,
            tau2=self.params.tau2,
            mu=self.params.mu,
            average_degree=self.params.average_degree,
            max_degree=self.params.max_degree,
            min_community=self.params.min_community,
            max_community=self.params.max_community,
            seed=self.params.seed
        )
        
        # Remove self-loops
        self.G.remove_edges_from(nx.selfloop_edges(self.G))
        
        # Extract communities
        self._extract_communities()
        
        # Generate features
        self.features = self._generate_features(feat_dim)
        
        return self.G, self.communities, self.labels
    
    def _extract_communities(self):
        """Extract community structure from LFR graph."""
        community_dict = nx.get_node_attributes(self.G, 'community')
        
        # Get unique communities
        unique_comms = {}
        for node, comm_set in community_dict.items():
            comm_id = tuple(sorted(comm_set))
            if comm_id not in unique_comms:
                unique_comms[comm_id] = len(unique_comms)
        
        # Assign labels and build community lists
        self.labels = np.zeros(self.params.n, dtype=np.int64)
        comm_members = {}
        
        for node, comm_set in community_dict.items():
            comm_id = tuple(sorted(comm_set))
            label = unique_comms[comm_id]
            self.labels[node] = label
            
            if label not in comm_members:
                comm_members[label] = []
            comm_members[label].append(node)
        
        self.communities = [comm_members[i] for i in sorted(comm_members.keys())]
    
    def _generate_features(self, feat_dim: int) -> np.ndarray:
        """Generate node features based on community structure."""
        n_communities = len(self.communities)
        
        # Each community gets a base feature vector
        community_bases = np.random.randn(n_communities, feat_dim)
        
        # Each node gets base + noise
        features = np.zeros((self.params.n, feat_dim))
        for i, node in enumerate(self.G.nodes()):
            label = self.labels[node]
            features[node] = community_bases[label] + 0.1 * np.random.randn(feat_dim)
        
        return features
    
    def to_pyg(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        if self.G is None:
            raise RuntimeError("Generate graph first with generate()")
        
        # Get edge index
        pyg_data = from_networkx(self.G)
        
        # Create Data object
        data = Data(
            x=torch.tensor(self.features, dtype=torch.float),
            edge_index=pyg_data.edge_index,
            y=torch.tensor(self.labels, dtype=torch.long),
            num_nodes=self.params.n
        )
        
        return data
    
    # =========================================
    # PERTURBATION METHODS
    # =========================================
    
    def perturb_edges_random(self, remove_ratio: float = 0.1) -> nx.Graph:
        """
        Randomly remove edges.
        
        Args:
            remove_ratio: Fraction of edges to remove
            
        Returns:
            Perturbed graph
        """
        G_perturbed = self.G.copy()
        edges = list(G_perturbed.edges())
        n_remove = int(len(edges) * remove_ratio)
        
        edges_to_remove = np.random.choice(len(edges), n_remove, replace=False)
        for idx in edges_to_remove:
            G_perturbed.remove_edge(*edges[idx])
        
        return G_perturbed
    
    def perturb_edges_targeted(self, remove_ratio: float = 0.1) -> nx.Graph:
        """
        Remove edges by betweenness centrality (targeted attack).
        
        Args:
            remove_ratio: Fraction of edges to remove
            
        Returns:
            Perturbed graph
        """
        G_perturbed = self.G.copy()
        n_remove = int(G_perturbed.number_of_edges() * remove_ratio)
        
        for _ in range(n_remove):
            if G_perturbed.number_of_edges() == 0:
                break
            
            # Compute edge betweenness
            edge_betweenness = nx.edge_betweenness_centrality(G_perturbed)
            
            # Remove edge with highest betweenness
            max_edge = max(edge_betweenness, key=edge_betweenness.get)
            G_perturbed.remove_edge(*max_edge)
        
        return G_perturbed
    
    def perturb_edges_community_mixing(self, mix_ratio: float = 0.1) -> nx.Graph:
        """
        Add cross-community edges (increase effective μ).
        
        Args:
            mix_ratio: Fraction of new cross-community edges to add
            
        Returns:
            Perturbed graph
        """
        G_perturbed = self.G.copy()
        n_add = int(G_perturbed.number_of_edges() * mix_ratio)
        
        # Get nodes by community
        comm_nodes = {i: set(comm) for i, comm in enumerate(self.communities)}
        
        added = 0
        max_attempts = n_add * 10
        attempts = 0
        
        while added < n_add and attempts < max_attempts:
            attempts += 1
            
            # Pick two different communities
            c1, c2 = np.random.choice(len(self.communities), 2, replace=False)
            
            # Pick random nodes from each
            n1 = np.random.choice(list(comm_nodes[c1]))
            n2 = np.random.choice(list(comm_nodes[c2]))
            
            # Add edge if not exists
            if not G_perturbed.has_edge(n1, n2):
                G_perturbed.add_edge(n1, n2)
                added += 1
        
        return G_perturbed
    
    def perturb_features_mean(self, shift: float = 1.0) -> np.ndarray:
        """
        Shift feature means.
        
        Args:
            shift: Mean shift amount
            
        Returns:
            Perturbed features
        """
        noise = np.random.normal(loc=shift, scale=1.0, size=self.features.shape)
        return self.features + noise
    
    def perturb_features_variance(self, scale: float = 2.0) -> np.ndarray:
        """
        Scale feature variance.
        
        Args:
            scale: Variance scaling factor
            
        Returns:
            Perturbed features
        """
        mean = self.features.mean(axis=0)
        centered = self.features - mean
        scaled = centered * scale
        return scaled + mean
    
    def perturb_features_dropout(self, drop_ratio: float = 0.1) -> np.ndarray:
        """
        Randomly mask features.
        
        Args:
            drop_ratio: Fraction of features to mask
            
        Returns:
            Perturbed features
        """
        mask = np.random.random(self.features.shape) > drop_ratio
        return self.features * mask


def generate_lfr_suite(
    mu_values: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    n_values: List[int] = [1000],
    feat_dim: int = 32,
    seed: int = 42
) -> Dict[str, Tuple[nx.Graph, List, np.ndarray]]:
    """
    Generate a suite of LFR benchmarks.
    
    Args:
        mu_values: List of mixing parameters
        n_values: List of network sizes
        feat_dim: Feature dimension
        seed: Random seed
        
    Returns:
        Dictionary mapping benchmark names to (graph, communities, labels)
    """
    benchmarks = {}
    
    for n in n_values:
        for mu in mu_values:
            name = f"lfr_n{n}_mu{mu:.1f}"
            
            benchmark = LFRBenchmark(
                n=n,
                mu=mu,
                seed=seed
            )
            G, communities, labels = benchmark.generate(feat_dim=feat_dim)
            
            benchmarks[name] = {
                'graph': G,
                'communities': communities,
                'labels': labels,
                'features': benchmark.features,
                'params': {'n': n, 'mu': mu}
            }
    
    return benchmarks









