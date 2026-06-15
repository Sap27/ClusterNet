"""
Rewired Real Network Benchmarks

Creates semi-synthetic benchmarks by rewiring real networks
to inject known community structure while preserving topology.

Based on: "A Comprehensive Survey of Community Detection Methods" (2020)
https://onlinelibrary.wiley.com/doi/full/10.1155/2020/7096230
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RewiredParams:
    """Rewired benchmark parameters."""
    rewiring_fraction: float = 0.5   # Fraction of edges to rewire (0=real, 1=synthetic)
    num_communities: Optional[int] = None  # Target number of communities
    preserve_degrees: bool = True    # Try to preserve degree distribution
    seed: Optional[int] = None


class RewiredBenchmark:
    """
    Rewired Real Network Benchmark Generator.
    
    Takes a real network and gradually imposes community structure
    through controlled edge rewiring. This creates benchmarks that:
    - Preserve realistic network properties (at low rewiring)
    - Have known ground truth communities (at high rewiring)
    - Allow studying the synthetic-to-real gap
    
    Usage:
        benchmark = RewiredBenchmark(real_graph)
        G_rewired, communities = benchmark.rewire(fraction=0.5)
    """
    
    def __init__(
        self,
        G: nx.Graph,
        params: Optional[RewiredParams] = None,
        **kwargs
    ):
        """
        Initialize with a real network.
        
        Args:
            G: Real-world NetworkX graph
            params: RewiredParams or keyword arguments
        """
        if params is None:
            params = RewiredParams(**kwargs)
        self.params = params
        
        # Store original graph
        self.G_original = G.copy()
        self.n = G.number_of_nodes()
        self.m = G.number_of_edges()
        
        # Ensure nodes are 0-indexed
        if min(G.nodes()) != 0 or max(G.nodes()) != self.n - 1:
            mapping = {node: i for i, node in enumerate(G.nodes())}
            self.G_original = nx.relabel_nodes(self.G_original, mapping)
        
        # Target communities (will be assigned)
        self.target_communities = None
        self.node_to_community = None
        
    def _assign_target_communities(self, num_communities: Optional[int] = None):
        """
        Assign target community structure.
        
        Uses spectral clustering on original graph to get
        reasonable community assignments.
        """
        if num_communities is None:
            num_communities = self.params.num_communities or int(np.sqrt(self.n / 10))
        
        try:
            from sklearn.cluster import SpectralClustering
            
            adj = nx.adjacency_matrix(self.G_original).astype(float)
            
            sc = SpectralClustering(
                n_clusters=num_communities,
                affinity='precomputed',
                n_components=min(num_communities, self.n - 1)
            )
            labels = sc.fit_predict(adj)
            
        except Exception:
            # Fallback: random assignment
            labels = np.random.randint(0, num_communities, self.n)
        
        # Build community structure
        self.node_to_community = {i: labels[i] for i in range(self.n)}
        self.target_communities = {}
        for node, comm in self.node_to_community.items():
            if comm not in self.target_communities:
                self.target_communities[comm] = []
            self.target_communities[comm].append(node)
        
        self.target_communities = [
            self.target_communities[i] 
            for i in sorted(self.target_communities.keys())
        ]
    
    def rewire(
        self,
        fraction: Optional[float] = None,
        num_communities: Optional[int] = None
    ) -> Tuple[nx.Graph, List[List[int]]]:
        """
        Rewire the network to impose community structure.
        
        Args:
            fraction: Rewiring fraction (0=original, 1=fully imposed)
            num_communities: Number of target communities
            
        Returns:
            G_rewired: Rewired graph
            communities: Target communities (ground truth)
        """
        if self.params.seed is not None:
            np.random.seed(self.params.seed)
        
        if fraction is None:
            fraction = self.params.rewiring_fraction
        
        # Assign target communities if needed
        if self.target_communities is None:
            self._assign_target_communities(num_communities)
        
        # Start with original
        G = self.G_original.copy()
        
        # Number of edges to rewire
        n_rewire = int(self.m * fraction)
        
        # Identify inter-community edges (candidates for removal)
        inter_edges = []
        for u, v in G.edges():
            if self.node_to_community[u] != self.node_to_community[v]:
                inter_edges.append((u, v))
        
        # Rewire: remove inter-community, add intra-community
        rewired = 0
        max_attempts = n_rewire * 10
        attempts = 0
        
        while rewired < n_rewire and attempts < max_attempts:
            attempts += 1
            
            if not inter_edges:
                break
            
            # Remove random inter-community edge
            idx = np.random.randint(len(inter_edges))
            u, v = inter_edges.pop(idx)
            
            if not G.has_edge(u, v):
                continue
                
            G.remove_edge(u, v)
            
            # Add random intra-community edge
            # Pick a random community
            comm_idx = np.random.randint(len(self.target_communities))
            comm = self.target_communities[comm_idx]
            
            if len(comm) < 2:
                # Restore the removed edge and try again
                G.add_edge(u, v)
                continue
            
            # Find a non-existing edge within community
            found = False
            for _ in range(10):  # Limited attempts
                a, b = np.random.choice(comm, 2, replace=False)
                if not G.has_edge(a, b):
                    G.add_edge(a, b)
                    found = True
                    break
            
            if found:
                rewired += 1
            else:
                # Restore the removed edge
                G.add_edge(u, v)
        
        return G, self.target_communities
    
    def rewire_sweep(
        self,
        fractions: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        num_communities: Optional[int] = None
    ) -> Dict[float, Tuple[nx.Graph, List[List[int]]]]:
        """
        Generate graphs at multiple rewiring levels.
        
        Args:
            fractions: List of rewiring fractions
            num_communities: Number of communities
            
        Returns:
            Dictionary mapping fraction -> (graph, communities)
        """
        results = {}
        
        # Assign communities once
        if self.target_communities is None:
            self._assign_target_communities(num_communities)
        
        for frac in fractions:
            G_rewired, communities = self.rewire(fraction=frac)
            results[frac] = (G_rewired, communities)
        
        return results
    
    def compute_topology_preservation(self, G_rewired: nx.Graph) -> Dict[str, float]:
        """
        Compute how well topology is preserved after rewiring.
        
        Args:
            G_rewired: Rewired graph
            
        Returns:
            Dictionary of preservation metrics
        """
        metrics = {}
        
        # Degree distribution correlation
        deg_orig = sorted([d for n, d in self.G_original.degree()])
        deg_rewired = sorted([d for n, d in G_rewired.degree()])
        
        if len(deg_orig) == len(deg_rewired):
            metrics['degree_correlation'] = np.corrcoef(deg_orig, deg_rewired)[0, 1]
        else:
            metrics['degree_correlation'] = 0.0
        
        # Clustering coefficient
        cc_orig = nx.average_clustering(self.G_original)
        cc_rewired = nx.average_clustering(G_rewired)
        metrics['clustering_ratio'] = cc_rewired / cc_orig if cc_orig > 0 else 1.0
        
        # Density
        density_orig = nx.density(self.G_original)
        density_rewired = nx.density(G_rewired)
        metrics['density_ratio'] = density_rewired / density_orig if density_orig > 0 else 1.0
        
        # Edge preservation
        orig_edges = set(self.G_original.edges())
        rewired_edges = set(G_rewired.edges())
        common = len(orig_edges & rewired_edges)
        metrics['edge_preservation'] = common / len(orig_edges) if orig_edges else 1.0
        
        return metrics


def load_standard_networks() -> Dict[str, nx.Graph]:
    """
    Load standard real networks for benchmarking.
    
    Returns:
        Dictionary of network name -> NetworkX graph
    """
    networks = {}
    
    # Karate club
    networks['karate'] = nx.karate_club_graph()
    
    # Football
    try:
        networks['football'] = nx.read_gml(
            'http://www-personal.umich.edu/~mejn/netdata/football.gml'
        )
    except:
        pass
    
    # Les Miserables
    try:
        networks['lesmis'] = nx.les_miserables_graph()
    except:
        pass
    
    # Dolphins
    try:
        networks['dolphins'] = nx.read_gml(
            'http://www-personal.umich.edu/~mejn/netdata/dolphins.gml'
        )
    except:
        pass
    
    return networks


def generate_rewired_suite(
    networks: Optional[Dict[str, nx.Graph]] = None,
    fractions: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    seed: int = 42
) -> Dict[str, Dict]:
    """
    Generate rewired benchmark suite.
    
    Args:
        networks: Dictionary of real networks (uses defaults if None)
        fractions: Rewiring fractions to test
        seed: Random seed
        
    Returns:
        Dictionary of benchmarks
    """
    if networks is None:
        networks = {'karate': nx.karate_club_graph()}
    
    benchmarks = {}
    
    for name, G in networks.items():
        benchmark = RewiredBenchmark(G, seed=seed)
        results = benchmark.rewire_sweep(fractions=fractions)
        
        for frac, (G_rewired, communities) in results.items():
            bench_name = f"{name}_rewired_{int(frac*100)}"
            
            # Compute topology metrics
            topo_metrics = benchmark.compute_topology_preservation(G_rewired)
            
            benchmarks[bench_name] = {
                'graph': G_rewired,
                'communities': communities,
                'labels': np.array([
                    next(i for i, c in enumerate(communities) if node in c)
                    for node in range(G_rewired.number_of_nodes())
                ]),
                'params': {
                    'source': name,
                    'rewiring_fraction': frac
                },
                'topology_preservation': topo_metrics
            }
    
    return benchmarks









