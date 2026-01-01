"""
Stochastic Block Model (SBM) Benchmark Generator

Supports:
- Standard SBM
- Degree-Corrected SBM
- Hierarchical/Nested SBM
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SBMParams:
    """SBM benchmark parameters."""
    n: int = 1000                    # Number of nodes
    k: int = 5                       # Number of communities
    p_in: float = 0.3                # Intra-community edge probability
    p_out: float = 0.05              # Inter-community edge probability
    sizes: Optional[List[int]] = None  # Community sizes (uniform if None)
    degree_corrected: bool = False   # Use degree-corrected SBM
    seed: Optional[int] = None


class SBMBenchmark:
    """
    Stochastic Block Model Benchmark Generator.
    
    Features:
    - Standard and degree-corrected variants
    - Configurable community sizes
    - Perturbation support
    """
    
    def __init__(self, params: Optional[SBMParams] = None, **kwargs):
        """
        Initialize SBM benchmark.
        
        Args:
            params: SBMParams object or keyword arguments
        """
        if params is None:
            params = SBMParams(**kwargs)
        self.params = params
        
        # Set uniform sizes if not provided
        if self.params.sizes is None:
            base_size = self.params.n // self.params.k
            self.params.sizes = [base_size] * self.params.k
            # Handle remainder
            remainder = self.params.n - sum(self.params.sizes)
            for i in range(remainder):
                self.params.sizes[i] += 1
        
        self.G = None
        self.communities = None
        self.labels = None
        self.features = None
    
    def generate(self, feat_dim: int = 32) -> Tuple[nx.Graph, List[List[int]], np.ndarray]:
        """
        Generate SBM benchmark graph.
        
        Args:
            feat_dim: Dimension of node features
            
        Returns:
            G: NetworkX graph
            communities: List of communities
            labels: Node label array
        """
        if self.params.seed is not None:
            np.random.seed(self.params.seed)
        
        # Build probability matrix
        k = self.params.k
        p_matrix = np.full((k, k), self.params.p_out)
        np.fill_diagonal(p_matrix, self.params.p_in)
        
        if self.params.degree_corrected:
            self.G = self._generate_dc_sbm(p_matrix)
        else:
            self.G = nx.stochastic_block_model(
                self.params.sizes,
                p_matrix,
                seed=self.params.seed
            )
        
        # Build communities and labels
        self._build_communities()
        
        # Generate features
        self.features = self._generate_features(feat_dim)
        
        return self.G, self.communities, self.labels
    
    def _generate_dc_sbm(self, p_matrix: np.ndarray) -> nx.Graph:
        """Generate degree-corrected SBM."""
        G = nx.Graph()
        
        # Generate degree sequence (power law within each community)
        total_nodes = sum(self.params.sizes)
        G.add_nodes_from(range(total_nodes))
        
        # Assign nodes to communities
        node_communities = []
        for i, size in enumerate(self.params.sizes):
            node_communities.extend([i] * size)
        
        # Generate power-law degree propensities
        theta = np.random.pareto(2.0, total_nodes) + 1
        theta = theta / theta.max()  # Normalize
        
        # Generate edges
        for i in range(total_nodes):
            for j in range(i + 1, total_nodes):
                ci, cj = node_communities[i], node_communities[j]
                p = p_matrix[ci, cj] * theta[i] * theta[j]
                p = min(p, 1.0)  # Ensure valid probability
                
                if np.random.random() < p:
                    G.add_edge(i, j)
        
        return G
    
    def _build_communities(self):
        """Build community structure."""
        self.communities = []
        self.labels = np.zeros(self.params.n, dtype=np.int64)
        
        node_idx = 0
        for i, size in enumerate(self.params.sizes):
            community = list(range(node_idx, node_idx + size))
            self.communities.append(community)
            self.labels[community] = i
            node_idx += size
    
    def _generate_features(self, feat_dim: int) -> np.ndarray:
        """Generate node features based on community structure."""
        n_communities = len(self.communities)
        
        # Community base vectors
        community_bases = np.random.randn(n_communities, feat_dim)
        
        # Node features = base + noise
        features = np.zeros((self.params.n, feat_dim))
        for node in range(self.params.n):
            label = self.labels[node]
            features[node] = community_bases[label] + 0.1 * np.random.randn(feat_dim)
        
        return features
    
    def to_pyg(self) -> Data:
        """Convert to PyTorch Geometric Data object."""
        if self.G is None:
            raise RuntimeError("Generate graph first with generate()")
        
        pyg_data = from_networkx(self.G)
        
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
    
    def perturb_to_disassortative(self) -> nx.Graph:
        """
        Create disassortative version (p_in < p_out).
        
        Returns:
            Disassortative graph
        """
        # Swap probabilities and regenerate
        old_p_in = self.params.p_in
        old_p_out = self.params.p_out
        
        self.params.p_in = old_p_out
        self.params.p_out = old_p_in
        
        G_disassort, _, _ = self.generate()
        
        # Restore
        self.params.p_in = old_p_in
        self.params.p_out = old_p_out
        
        return G_disassort
    
    def vary_snr(self, snr: float) -> nx.Graph:
        """
        Generate graph with specified signal-to-noise ratio.
        
        SNR = (p_in - p_out) / sqrt(p_out)
        
        Args:
            snr: Desired SNR value
            
        Returns:
            Graph with specified SNR
        """
        # Solve for p_in given p_out and SNR
        p_out = self.params.p_out
        p_in = snr * np.sqrt(p_out) + p_out
        p_in = min(p_in, 1.0)
        
        # Temporarily change and generate
        old_p_in = self.params.p_in
        self.params.p_in = p_in
        
        G_new, _, _ = self.generate()
        
        self.params.p_in = old_p_in
        
        return G_new


class HierarchicalSBM(SBMBenchmark):
    """
    Hierarchical/Nested SBM for multi-scale community structure.
    """
    
    def __init__(
        self,
        n: int = 1000,
        levels: int = 2,
        branching: int = 3,
        p_in_base: float = 0.5,
        p_decay: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize hierarchical SBM.
        
        Args:
            n: Number of nodes
            levels: Number of hierarchy levels
            branching: Branching factor at each level
            p_in_base: Base intra-community probability
            p_decay: Decay factor for inter-level probabilities
            seed: Random seed
        """
        self.levels = levels
        self.branching = branching
        self.p_in_base = p_in_base
        self.p_decay = p_decay
        
        # Total communities at finest level
        k = branching ** levels
        
        super().__init__(n=n, k=k, seed=seed)
        
        self.hierarchy = None
    
    def generate(self, feat_dim: int = 32) -> Tuple[nx.Graph, List[List[int]], np.ndarray]:
        """Generate hierarchical SBM."""
        if self.params.seed is not None:
            np.random.seed(self.params.seed)
        
        # Build hierarchical probability matrix
        p_matrix = self._build_hierarchical_p_matrix()
        
        self.G = nx.stochastic_block_model(
            self.params.sizes,
            p_matrix,
            seed=self.params.seed
        )
        
        self._build_communities()
        self._build_hierarchy()
        self.features = self._generate_features(feat_dim)
        
        return self.G, self.communities, self.labels
    
    def _build_hierarchical_p_matrix(self) -> np.ndarray:
        """Build probability matrix with hierarchical structure."""
        k = self.params.k
        p_matrix = np.zeros((k, k))
        
        for i in range(k):
            for j in range(k):
                # Find common ancestor level
                level = self._common_ancestor_level(i, j)
                
                if level == self.levels:  # Same community
                    p_matrix[i, j] = self.p_in_base
                else:
                    # Probability decays with hierarchy distance
                    p_matrix[i, j] = self.p_in_base * (self.p_decay ** (self.levels - level))
        
        return p_matrix
    
    def _common_ancestor_level(self, i: int, j: int) -> int:
        """Find the level of common ancestor for two communities."""
        if i == j:
            return self.levels
        
        for level in range(self.levels, 0, -1):
            divisor = self.branching ** (self.levels - level)
            if i // divisor == j // divisor:
                return level
        
        return 0
    
    def _build_hierarchy(self):
        """Build hierarchical community structure."""
        self.hierarchy = {}
        
        for level in range(self.levels + 1):
            n_clusters = self.branching ** level
            clusters_at_level = []
            
            cluster_size = self.params.k // n_clusters
            
            for c in range(n_clusters):
                # Merge fine-grained communities
                start = c * cluster_size
                end = start + cluster_size
                merged = []
                for comm in self.communities[start:end]:
                    merged.extend(comm)
                clusters_at_level.append(merged)
            
            self.hierarchy[level] = clusters_at_level
    
    def get_communities_at_level(self, level: int) -> List[List[int]]:
        """Get communities at specified hierarchy level."""
        if self.hierarchy is None:
            raise RuntimeError("Generate graph first")
        return self.hierarchy.get(level, self.communities)


def generate_sbm_suite(
    n_values: List[int] = [1000],
    k_values: List[int] = [5, 10],
    snr_values: List[float] = [1.0, 2.0, 4.0],
    feat_dim: int = 32,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Generate a suite of SBM benchmarks.
    
    Args:
        n_values: Network sizes
        k_values: Number of communities
        snr_values: Signal-to-noise ratios
        feat_dim: Feature dimension
        seed: Random seed
        
    Returns:
        Dictionary of benchmarks
    """
    benchmarks = {}
    
    for n in n_values:
        for k in k_values:
            for snr in snr_values:
                # Calculate p_in from SNR
                p_out = 0.05
                p_in = snr * np.sqrt(p_out) + p_out
                p_in = min(p_in, 1.0)
                
                name = f"sbm_n{n}_k{k}_snr{snr:.1f}"
                
                benchmark = SBMBenchmark(
                    n=n,
                    k=k,
                    p_in=p_in,
                    p_out=p_out,
                    seed=seed
                )
                G, communities, labels = benchmark.generate(feat_dim=feat_dim)
                
                benchmarks[name] = {
                    'graph': G,
                    'communities': communities,
                    'labels': labels,
                    'features': benchmark.features,
                    'params': {'n': n, 'k': k, 'snr': snr, 'p_in': p_in, 'p_out': p_out}
                }
    
    return benchmarks

