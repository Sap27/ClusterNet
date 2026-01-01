"""
Motif Enrichment Analysis for Biological Networks

Analyzes community structure based on network motif preservation.
Key insight: Good biological communities should be enriched for 
functional motifs (e.g., feed-forward loops in GRNs).
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter
from dataclasses import dataclass


@dataclass
class MotifResult:
    """Result of motif analysis for a community."""
    community_id: int
    motif_counts: Dict[str, int]
    expected_counts: Dict[str, float]
    enrichment_scores: Dict[str, float]
    z_scores: Dict[str, float]


class MotifAnalyzer:
    """
    Motif-based community quality analyzer.
    
    Computes motif enrichment within communities compared to
    random null model. Useful for evaluating biological network
    community structure.
    
    Supported motifs:
    - triangle: 3-cliques (transitive closure)
    - square: 4-cycles
    - feedforward: Feed-forward loops (directed)
    - bifan: Bi-fan motifs (directed)
    """
    
    # Motif patterns (as frozensets of edges for undirected)
    MOTIF_TYPES = ['triangle', 'square', 'feedforward', 'bifan']
    
    def __init__(
        self,
        G: nx.Graph,
        directed: bool = False,
        n_random: int = 100,
        seed: Optional[int] = None
    ):
        """
        Initialize motif analyzer.
        
        Args:
            G: NetworkX graph
            directed: Whether to analyze as directed graph
            n_random: Number of random graphs for null model
            seed: Random seed
        """
        self.G = G
        self.directed = directed
        self.n_random = n_random
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
        
        # Pre-compute global motif counts
        self._global_motifs = None
    
    def count_triangles(self, nodes: Optional[Set[int]] = None) -> int:
        """
        Count triangles in graph or subgraph.
        
        Args:
            nodes: Optional subset of nodes
            
        Returns:
            Number of triangles
        """
        if nodes is None:
            return sum(nx.triangles(self.G).values()) // 3
        
        # Subgraph triangles
        subgraph = self.G.subgraph(nodes)
        return sum(nx.triangles(subgraph).values()) // 3
    
    def count_squares(self, nodes: Optional[Set[int]] = None) -> int:
        """
        Count 4-cycles (squares) in graph or subgraph.
        
        Args:
            nodes: Optional subset of nodes
            
        Returns:
            Number of 4-cycles
        """
        if nodes is not None:
            G = self.G.subgraph(nodes)
        else:
            G = self.G
        
        # Count 4-cycles by looking for paths of length 2
        count = 0
        nodes_list = list(G.nodes())
        
        for i, u in enumerate(nodes_list):
            neighbors_u = set(G.neighbors(u))
            for v in nodes_list[i+1:]:
                if v in neighbors_u:
                    continue  # Skip if already connected
                
                # Count common neighbors (potential 4-cycles)
                neighbors_v = set(G.neighbors(v))
                common = neighbors_u & neighbors_v
                
                # Each pair of common neighbors forms a 4-cycle
                n_common = len(common)
                count += n_common * (n_common - 1) // 2
        
        return count // 2  # Each 4-cycle counted twice
    
    def count_feedforward_loops(self, nodes: Optional[Set[int]] = None) -> int:
        """
        Count feed-forward loops in directed graph.
        
        A feed-forward loop: A -> B, A -> C, B -> C
        
        Args:
            nodes: Optional subset of nodes
            
        Returns:
            Number of FFLs
        """
        if not self.directed:
            return 0
        
        if nodes is not None:
            G = self.G.subgraph(nodes)
        else:
            G = self.G
        
        if not isinstance(G, nx.DiGraph):
            return 0
        
        count = 0
        for A in G.nodes():
            out_A = set(G.successors(A))
            for B in out_A:
                out_B = set(G.successors(B))
                # C must be in both out_A and out_B
                common = out_A & out_B
                count += len(common)
        
        return count
    
    def count_bifans(self, nodes: Optional[Set[int]] = None) -> int:
        """
        Count bi-fan motifs in directed graph.
        
        A bi-fan: A -> C, A -> D, B -> C, B -> D
        (Two sources with two common targets)
        
        Args:
            nodes: Optional subset of nodes
            
        Returns:
            Number of bi-fans
        """
        if not self.directed:
            return 0
        
        if nodes is not None:
            G = self.G.subgraph(nodes)
        else:
            G = self.G
        
        if not isinstance(G, nx.DiGraph):
            return 0
        
        count = 0
        nodes_list = list(G.nodes())
        
        for i, A in enumerate(nodes_list):
            out_A = set(G.successors(A))
            for B in nodes_list[i+1:]:
                out_B = set(G.successors(B))
                # Common targets
                common = out_A & out_B
                n_common = len(common)
                # Each pair of common targets forms a bi-fan
                count += n_common * (n_common - 1) // 2
        
        return count
    
    def count_all_motifs(self, nodes: Optional[Set[int]] = None) -> Dict[str, int]:
        """
        Count all motif types.
        
        Args:
            nodes: Optional subset of nodes
            
        Returns:
            Dictionary of motif type -> count
        """
        counts = {
            'triangle': self.count_triangles(nodes),
            'square': self.count_squares(nodes)
        }
        
        if self.directed:
            counts['feedforward'] = self.count_feedforward_loops(nodes)
            counts['bifan'] = self.count_bifans(nodes)
        
        return counts
    
    def _generate_random_subgraph(self, n_nodes: int, n_edges: int) -> nx.Graph:
        """Generate random graph with same size for null model."""
        if self.directed:
            return nx.gnm_random_graph(n_nodes, n_edges, directed=True)
        else:
            return nx.gnm_random_graph(n_nodes, n_edges, directed=False)
    
    def compute_expected_motifs(
        self,
        n_nodes: int,
        n_edges: int
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute expected motif counts under null model.
        
        Args:
            n_nodes: Number of nodes
            n_edges: Number of edges
            
        Returns:
            Dictionary of motif type -> (mean, std)
        """
        if n_nodes < 3 or n_edges < 3:
            return {m: (0.0, 0.0) for m in self.MOTIF_TYPES}
        
        # Sample random graphs
        samples = {m: [] for m in self.MOTIF_TYPES}
        
        for _ in range(self.n_random):
            G_random = self._generate_random_subgraph(n_nodes, n_edges)
            
            # Temporarily swap for counting
            G_orig = self.G
            self.G = G_random
            
            counts = self.count_all_motifs()
            
            self.G = G_orig
            
            for motif, count in counts.items():
                samples[motif].append(count)
        
        # Compute statistics
        results = {}
        for motif, values in samples.items():
            if values:
                results[motif] = (np.mean(values), np.std(values) + 1e-10)
            else:
                results[motif] = (0.0, 1e-10)
        
        return results
    
    def analyze_community(
        self,
        community: List[int],
        community_id: int = 0
    ) -> MotifResult:
        """
        Analyze motif enrichment in a single community.
        
        Args:
            community: List of node IDs
            community_id: Identifier for this community
            
        Returns:
            MotifResult with enrichment analysis
        """
        nodes = set(community)
        
        # Count motifs in community
        observed = self.count_all_motifs(nodes)
        
        # Get expected counts
        subgraph = self.G.subgraph(nodes)
        n_nodes = subgraph.number_of_nodes()
        n_edges = subgraph.number_of_edges()
        
        expected_stats = self.compute_expected_motifs(n_nodes, n_edges)
        
        expected = {m: stats[0] for m, stats in expected_stats.items()}
        
        # Compute enrichment scores (observed / expected)
        enrichment = {}
        z_scores = {}
        
        for motif in observed:
            exp_mean, exp_std = expected_stats.get(motif, (0.0, 1e-10))
            
            if exp_mean > 0:
                enrichment[motif] = observed[motif] / exp_mean
            else:
                enrichment[motif] = float('inf') if observed[motif] > 0 else 1.0
            
            # Z-score
            z_scores[motif] = (observed[motif] - exp_mean) / exp_std
        
        return MotifResult(
            community_id=community_id,
            motif_counts=observed,
            expected_counts=expected,
            enrichment_scores=enrichment,
            z_scores=z_scores
        )
    
    def analyze_all_communities(
        self,
        communities: List[List[int]]
    ) -> Tuple[List[MotifResult], Dict[str, float]]:
        """
        Analyze motif enrichment for all communities.
        
        Args:
            communities: List of communities
            
        Returns:
            results: List of MotifResult per community
            summary: Aggregated statistics
        """
        results = []
        
        for i, community in enumerate(communities):
            if len(community) >= 3:  # Need at least 3 nodes for motifs
                result = self.analyze_community(community, i)
                results.append(result)
        
        # Aggregate statistics
        summary = {}
        
        for motif in self.MOTIF_TYPES:
            z_scores = [r.z_scores.get(motif, 0) for r in results if motif in r.z_scores]
            if z_scores:
                summary[f'{motif}_mean_z'] = np.mean(z_scores)
                summary[f'{motif}_frac_enriched'] = np.mean([z > 1.96 for z in z_scores])
        
        # Overall score (average of mean z-scores)
        mean_zs = [v for k, v in summary.items() if 'mean_z' in k]
        summary['overall_enrichment'] = np.mean(mean_zs) if mean_zs else 0.0
        
        return results, summary


def compute_motif_enrichment(
    G: nx.Graph,
    communities: List[List[int]],
    directed: bool = False,
    n_random: int = 100,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute motif enrichment score for a community partition.
    
    Higher scores indicate communities that preserve
    functionally meaningful substructures.
    
    Args:
        G: NetworkX graph
        communities: List of communities
        directed: Whether graph is directed
        n_random: Number of random samples for null model
        seed: Random seed
        
    Returns:
        Dictionary of enrichment metrics
    """
    analyzer = MotifAnalyzer(G, directed=directed, n_random=n_random, seed=seed)
    _, summary = analyzer.analyze_all_communities(communities)
    return summary


def biological_benchmark_score(
    G: nx.Graph,
    communities: List[List[int]],
    ground_truth: Optional[List[List[int]]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive biological benchmark score.
    
    Combines:
    - Motif enrichment
    - Modularity
    - (Optionally) NMI with ground truth
    
    Args:
        G: NetworkX graph
        communities: Predicted communities
        ground_truth: Optional ground truth communities
        
    Returns:
        Dictionary of scores
    """
    scores = {}
    
    # Motif enrichment
    motif_scores = compute_motif_enrichment(G, communities, n_random=50)
    scores['motif_enrichment'] = motif_scores.get('overall_enrichment', 0.0)
    
    # Modularity
    try:
        # Convert to format expected by modularity function
        from networkx.algorithms.community import modularity
        comm_sets = [set(c) for c in communities]
        scores['modularity'] = modularity(G, comm_sets)
    except:
        scores['modularity'] = 0.0
    
    # NMI with ground truth if available
    if ground_truth is not None:
        try:
            from sklearn.metrics import normalized_mutual_info_score
            
            # Convert to labels
            n = G.number_of_nodes()
            pred_labels = np.zeros(n, dtype=int)
            true_labels = np.zeros(n, dtype=int)
            
            for i, comm in enumerate(communities):
                for node in comm:
                    pred_labels[node] = i
            
            for i, comm in enumerate(ground_truth):
                for node in comm:
                    true_labels[node] = i
            
            scores['nmi'] = normalized_mutual_info_score(true_labels, pred_labels)
        except:
            pass
    
    # Combined score
    weights = {'motif_enrichment': 0.4, 'modularity': 0.4, 'nmi': 0.2}
    available = [(k, v) for k, v in scores.items() if k in weights]
    if available:
        total_weight = sum(weights[k] for k, _ in available)
        scores['combined'] = sum(weights[k] * v / total_weight for k, v in available)
    
    return scores

