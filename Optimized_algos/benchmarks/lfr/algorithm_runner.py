"""
Algorithm Runner for LFR Benchmarks
===================================

Unified interface to run all community detection algorithms on LFR networks.
Includes classical algorithms from ClusterNet and GNN models.
"""

import os
import sys
import time
import traceback
import networkx as nx
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path

# Add ClusterNet to path
CLUSTERNET_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CLUSTERNET_PATH))
import clusternet

@dataclass
class AlgorithmResult:
    """Result from running an algorithm."""
    name: str
    communities: List[List[int]]
    runtime: float
    success: bool
    error_message: str = ""
    metadata: Dict = None


def _auto_weighted(G, kwargs_weighted):
    """Auto-detect if graph has weights, respecting explicit kwargs override."""
    if kwargs_weighted is not None:
        return kwargs_weighted
    return nx.is_weighted(G)


def _ensure_weights(G):
    """Ensure graph has weight attributes (some algorithms require this)."""
    if not nx.is_weighted(G):
        for u, v in G.edges():
            G[u][v]['weight'] = 1.0
    return G


class AlgorithmRunner:
    """
    Run community detection algorithms with unified interface.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._algorithms = {}
        self._load_algorithms()
    
    def _load_algorithms(self):
        """Load all available algorithms."""
        
        # ============= CLASSICAL ALGORITHMS =============
        
        # Louvain - use ClusterNet wrapper
        def run_louvain(G, **kwargs):
            wrapper = clusternet.algorithms.louvain_wrapper.LouvainWrapper(
                G, resolution=kwargs.get('resolution', 1.0)
            )
            return wrapper.run()
        
        self._algorithms['louvain'] = run_louvain
        
        # Leiden
        def run_leiden(G, **kwargs):
            wrapper = clusternet.algorithms.LeidenWrapper(G, resolution=kwargs.get('resolution', 1.0))
            return wrapper.run()
        
        self._algorithms['leiden'] = run_leiden
        
        # Label Propagation
        def run_label_propagation(G, **kwargs):
            wrapper = clusternet.algorithms.LabelPropagationWrapper(
                G, weighted=kwargs.get('weighted', True), directed=kwargs.get('directed', False)
            )
            return wrapper.run()
        
        self._algorithms['label_propagation'] = run_label_propagation
        
        # Walktrap
        def run_walktrap(G, **kwargs):
            wrapper = clusternet.algorithms.WalktrapWrapper(
                G, 
                steps=kwargs.get('steps', 10),  # Match test script
                weighted=kwargs.get('weighted', True),
                directed=kwargs.get('directed', False)
            )
            return wrapper.run()
        
        self._algorithms['walktrap'] = run_walktrap
        
        # FastGreedy
        def run_fastgreedy(G, **kwargs):
            wrapper = clusternet.algorithms.FastGreedyWrapper(
                G, weighted=kwargs.get('weighted', True), 
                directed=kwargs.get('directed', False), 
                n_clusters=kwargs.get('n_clusters', None)
            )
            return wrapper.run()
        self._algorithms['fastgreedy'] = run_fastgreedy
        
        # Spectral Clustering
        def run_spectral(G, **kwargs):
            n = G.number_of_nodes()
            estimated_k = max(2, int(np.sqrt(n / 2)))
            wrapper = clusternet.algorithms.SpectralWrapper(
                G, n_clusters=kwargs.get('n_clusters', estimated_k), 
                weighted=kwargs.get('weighted', True), 
                directed=kwargs.get('directed', False)
            )
            return wrapper.run()
        
        self._algorithms['spectral'] = run_spectral
        
        # Infomap (via TeamCS)
        def run_infomap(G, **kwargs):
            wrapper = clusternet.algorithms.TeamCSWrapper(
                G, weighted=kwargs.get('weighted', True), 
                directed=kwargs.get('directed', False), recursive=False
            )
            return wrapper.run()
        
        self._algorithms['infomap'] = run_infomap
        
        # Spinglass
        def run_spinglass(G, **kwargs):
            wrapper = clusternet.algorithms.SpinGlassWrapper(
                G, spins=kwargs.get('spins', 25), 
                weighted=kwargs.get('weighted', True), 
                directed=kwargs.get('directed', False)
            )
            return wrapper.run()
        
        self._algorithms['spinglass'] = run_spinglass

        # Leading Eigenvector
        def run_leading_eigen(G, **kwargs):
            wrapper = clusternet.algorithms.LeadingEigenWrapper(
                G, weighted=kwargs.get('weighted', True), 
                directed=kwargs.get('directed', False)
            )
            return wrapper.run()
        
        self._algorithms['leading_eigen'] = run_leading_eigen
        
        # ============= STATISTICAL INFERENCE =============
        
        # EM (Expectation-Maximization)
        def run_em(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.statistical_wrappers.EMWrapper(
                G, k=kwargs.get('k'), auto_k=kwargs.get('auto_k', True)
            )
            return wrapper.run()
        
        self._algorithms['em'] = run_em
        
        # SBM
        def run_sbm(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.statistical_wrappers.SBMWrapper(G)
            return wrapper.run()
        
        self._algorithms['sbm'] = run_sbm

        # Nested SBM
        def run_nested_sbm(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.statistical_wrappers.NestedSBMWrapper(G)
            return wrapper.run()
        
        self._algorithms['nested_sbm'] = run_nested_sbm
        
        # ============= CDLIB ALGORITHMS =============
        
        # CPM - use ClusterNet wrapper directly (works when graph has weights)
        def run_cpm(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.physics_wrappers.CPMWrapper(
                G, resolution_parameter=kwargs.get('resolution_parameter', 0.1)
            )
            return wrapper.run()
        
        self._algorithms['cpm'] = run_cpm
        
        # RB_POTS - use ClusterNet wrapper directly
        def run_rb_pots(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.physics_wrappers.RBPotsWrapper(
                G, resolution_parameter=kwargs.get('resolution_parameter', 1.0)
            )
            return wrapper.run()
        
        self._algorithms['rb_pots'] = run_rb_pots

        # RBER_POTS - use ClusterNet wrapper directly
        def run_rber_pots(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.physics_wrappers.RBERPotsWrapper(
                G, resolution_parameter=kwargs.get('resolution_parameter', 1.0)
            )
            return wrapper.run()
        
        self._algorithms['rber_pots'] = run_rber_pots
        
        # SCAN - use ClusterNet wrapper directly
        def run_scan(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.structural_wrappers.SCANWrapper(
                G, epsilon=kwargs.get('epsilon', 0.5), mu=kwargs.get('mu', 3)
            )
            return wrapper.run()
        
        self._algorithms['scan'] = run_scan
        
        # AGDL
        def run_agdl(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.structural_wrappers.AGDLWrapper(
                G, auto_tune=kwargs.get('auto_tune', True)
            )
            return wrapper.run()
        
        self._algorithms['agdl'] = run_agdl
        
        # Async Fluid
        def run_async_fluid(G, **kwargs):
            n = G.number_of_nodes()
            estimated_k = max(2, int(np.sqrt(n / 2)))
            wrapper = clusternet.algorithms.cdlib_wrappers.diffusion_wrappers.AsyncFluidWrapper(
                G, k=kwargs.get('k', estimated_k)
            )
            return wrapper.run()
        
        self._algorithms['async_fluid'] = run_async_fluid
        
        # DER
        def run_der(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.diffusion_wrappers.DERWrapper(
                G, auto_tune=kwargs.get('auto_tune', True)
            )
            return wrapper.run()
        
        self._algorithms['der'] = run_der
        
        # ============= OVERLAPPING ALGORITHMS =============
        
        # Angel
        def run_angel(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.overlapping_wrappers.AngelWrapper(
                G, threshold=kwargs.get('threshold', 0.5), use_demon_fallback=True
            )
            return wrapper.run()
        
        self._algorithms['angel'] = run_angel
        
        # DEMON
        def run_demon(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.overlapping_wrappers.DEMONWrapper(
                G, epsilon=kwargs.get('epsilon', 0.25), min_com_size=kwargs.get('min_com_size', 3)
            )
            return wrapper.run()
        
        self._algorithms['demon'] = run_demon
        
        # k-Clique
        def run_kclique(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.overlapping_wrappers.KCliqueWrapper(
                G, k=kwargs.get('k', 3)
            )
            return wrapper.run()
        
        self._algorithms['kclique'] = run_kclique

        # Surprise Communities
        def run_surprise(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.physics_wrappers.SurpriseCommunitiesWrapper(G)
            return wrapper.run()
        
        self._algorithms['surprise'] = run_surprise

        # GDMP2
        def run_gdmp2(G, **kwargs):
            wrapper = clusternet.algorithms.cdlib_wrappers.structural_wrappers.GDMP2Wrapper(
                G, min_threshold=kwargs.get('min_threshold', 0.75)
            )
            return wrapper.run()
        
        self._algorithms['gdmp2'] = run_gdmp2
        
        # ============= CLUSTERNET CUSTOM =============
        
        # SCORE
        def run_score(G, **kwargs):
            try:
                from clusternet.algorithms.score_wrapper import SCOREWrapper
                n = G.number_of_nodes()
                # Estimate n_clusters if not provided
                estimated_k = kwargs.get('n_clusters', max(2, int(np.sqrt(n / 2))))
                wrapper = SCOREWrapper(G, n_clusters=estimated_k)
                return wrapper.run()
            except:
                return None
        
        self._algorithms['score'] = run_score
        
        # SVT
        def run_svt(G, **kwargs):
            try:
                from clusternet.algorithms.svt_wrapper import SVTWrapper
                n = G.number_of_nodes()
                # Scale parameters to graph size (SVT fails on small graphs with default params)
                svd_k = min(kwargs.get('svd_k', 50), max(3, n // 3))
                n_clusters = min(kwargs.get('n_clusters', 50), max(2, n // 5))
                n_neighbors = min(kwargs.get('n_neighbors', 100), max(2, n - 1))
                wrapper = SVTWrapper(G, svd_k=svd_k, n_clusters=n_clusters, n_neighbors=n_neighbors)
                return wrapper.run()
            except Exception as e:
                print(f"SVT error: {e}")
                return None
        
        self._algorithms['svt'] = run_svt
        
        # TeamCS
        def run_teamcs(G, **kwargs):
            try:
                from clusternet.algorithms.teamcs_wrapper import TeamCSWrapper
                wrapper = TeamCSWrapper(G)
                return wrapper.run()
            except:
                return None
        
        self._algorithms['teamcs'] = run_teamcs
        
        # SimNet
        def run_simnet(G, **kwargs):
            try:
                from clusternet.algorithms.simnet_wrapper import SimNetWrapper
                wrapper = SimNetWrapper(G)
                return wrapper.run()
            except:
                return None
        
        self._algorithms['simnet'] = run_simnet
        
        # Tusk
        def run_tusk(G, **kwargs):
            try:
                from clusternet.algorithms.tusk_wrapper import TuskWrapper
                wrapper = TuskWrapper(G)
                return wrapper.run()
            except:
                return None
        
        self._algorithms['tusk'] = run_tusk
        
        # BiGS2
        def run_bigs2(G, **kwargs):
            try:
                from clusternet.algorithms.bigs2_wrapper import BiGS2Wrapper
                wrapper = BiGS2Wrapper(G)
                return wrapper.run()
            except:
                return None
        
        self._algorithms['bigs2'] = run_bigs2
        
        # CSBIO-IITM2
        def run_csbio(G, **kwargs):
            try:
                from clusternet.algorithms.csbio_iitm2_wrapper import CSBIOWrapper
                wrapper = CSBIOWrapper(G)
                return wrapper.run()
            except:
                return None
        
        self._algorithms['csbio_iitm2'] = run_csbio
    
    def _partition_dict_to_list(self, partition: Dict) -> List[List[int]]:
        """Convert partition dict {node: community_id} to list of communities."""
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        return list(communities.values())
    
    def get_available_algorithms(self) -> List[str]:
        """Return list of available algorithm names."""
        return list(self._algorithms.keys())
    
    def run_algorithm(
        self,
        G: nx.Graph,
        algorithm_name: str,
        **kwargs
    ) -> AlgorithmResult:
        """
        Run a single algorithm on a graph.
        
        Args:
            G: NetworkX graph
            algorithm_name: Name of algorithm to run
            **kwargs: Algorithm-specific parameters
            
        Returns:
            AlgorithmResult with communities and metadata
        """
        if algorithm_name not in self._algorithms:
            return AlgorithmResult(
                name=algorithm_name,
                communities=[],
                runtime=0.0,
                success=False,
                error_message=f"Algorithm '{algorithm_name}' not found"
            )
        
        algo_func = self._algorithms[algorithm_name]
        
        # Ensure graph has weights (many algorithms require this)
        _ensure_weights(G)
        
        start_time = time.time()
        try:
            communities = algo_func(G, **kwargs)
            runtime = time.time() - start_time
            
            if communities is None:
                return AlgorithmResult(
                    name=algorithm_name,
                    communities=[],
                    runtime=runtime,
                    success=False,
                    error_message="Algorithm returned None"
                )
            
            # Ensure communities is list of lists of ints
            communities = [
                [int(n) for n in comm]
                for comm in communities
            ]
            
            return AlgorithmResult(
                name=algorithm_name,
                communities=communities,
                runtime=runtime,
                success=True,
                metadata={'num_communities': len(communities)}
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            if self.verbose:
                print(f"  {algorithm_name} failed: {str(e)}")
            return AlgorithmResult(
                name=algorithm_name,
                communities=[],
                runtime=runtime,
                success=False,
                error_message=str(e)
            )
    
    def run_all_algorithms(
        self,
        G: nx.Graph,
        algorithm_names: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, AlgorithmResult]:
        """
        Run multiple algorithms on a graph.
        
        Args:
            G: NetworkX graph
            algorithm_names: List of algorithms to run (None = all)
            **kwargs: Passed to all algorithms
            
        Returns:
            Dict mapping algorithm name to result
        """
        if algorithm_names is None:
            algorithm_names = list(self._algorithms.keys())
        
        results = {}
        for name in algorithm_names:
            if self.verbose:
                print(f"  Running {name}...", end=' ')
            
            result = self.run_algorithm(G, name, **kwargs)
            results[name] = result
            
            if self.verbose:
                if result.success:
                    print(f"✓ ({len(result.communities)} communities, {result.runtime:.2f}s)")
                else:
                    print(f"✗ ({result.error_message})")
        
        return results


# ============= GNN ALGORITHMS =============

class GNNRunner:
    """
    Run GNN-based community detection algorithms.
    """
    
    def __init__(self, device: str = 'cpu', verbose: bool = True):
        self.device = device
        self.verbose = verbose
        self._check_torch()
    
    def _check_torch(self):
        """Check if PyTorch and PyG are available."""
        try:
            import torch
            import torch_geometric
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            if self.verbose:
                print("Warning: PyTorch/PyG not available. GNN models will be skipped.")
    
    def _graph_to_pyg(self, G: nx.Graph, feature_dim: int = 32, use_spectral: bool = True):
        """
        Convert NetworkX graph to PyG Data object with spectral features.
        
        Uses Laplacian eigenvectors as node features - these encode community structure
        and give GNNs strong signal for clustering.
        """
        import torch
        from torch_geometric.data import Data
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import diags
        
        num_nodes = G.number_of_nodes()
        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        # Create edge index
        edges = list(G.edges())
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(
                [[node_to_idx[e[0]] for e in edges] + [node_to_idx[e[1]] for e in edges],
                 [node_to_idx[e[1]] for e in edges] + [node_to_idx[e[0]] for e in edges]],
                dtype=torch.long
            )
        
        if use_spectral and num_nodes > 2:
            # Compute spectral features (Laplacian eigenvectors)
            try:
                # Create adjacency matrix with proper node ordering
                G_relabeled = nx.relabel_nodes(G, node_to_idx)
                A = nx.adjacency_matrix(G_relabeled).astype(np.float64)
                
                # Compute normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
                degrees = np.array(A.sum(axis=1)).flatten()
                degrees[degrees == 0] = 1  # Avoid division by zero
                D_inv_sqrt = diags(1.0 / np.sqrt(degrees))
                L_norm = diags(np.ones(num_nodes)) - D_inv_sqrt @ A @ D_inv_sqrt
                
                # Number of eigenvectors to compute
                k = min(feature_dim, num_nodes - 2)
                
                # Compute smallest k+1 eigenvalues (skip the trivial one)
                eigenvalues, eigenvectors = eigsh(L_norm, k=k+1, which='SM', tol=1e-6)
                
                # Sort by eigenvalue and skip the first (trivial) eigenvector
                idx = np.argsort(eigenvalues)
                eigenvectors = eigenvectors[:, idx][:, 1:k+1]  # Skip first, take k
                
                # Normalize eigenvectors
                eigenvectors = eigenvectors / (np.linalg.norm(eigenvectors, axis=0, keepdims=True) + 1e-8)
                
                # Pad if needed
                if eigenvectors.shape[1] < feature_dim:
                    padding = np.zeros((num_nodes, feature_dim - eigenvectors.shape[1]))
                    eigenvectors = np.hstack([eigenvectors, padding])
                
                x = torch.FloatTensor(eigenvectors[:, :feature_dim])
                
                if self.verbose:
                    print(f"    Using spectral features: {x.shape[1]} Laplacian eigenvectors")
                    
            except Exception as e:
                if self.verbose:
                    print(f"    Spectral features failed ({e}), using structural features")
                x = self._compute_structural_features(G, nodes, node_to_idx, feature_dim)
        else:
            x = self._compute_structural_features(G, nodes, node_to_idx, feature_dim)
        
        return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    
    def _compute_structural_features(self, G, nodes, node_to_idx, feature_dim):
        """Compute structural node features as fallback."""
        import torch
        
        num_nodes = len(nodes)
        
        # Compute structural features
        degrees = np.array([G.degree(n) for n in nodes], dtype=np.float32)
        max_deg = max(degrees.max(), 1)
        
        clustering = np.array([nx.clustering(G, n) for n in nodes], dtype=np.float32)
        
        # Compute neighbor degree stats
        neighbor_deg_mean = np.zeros(num_nodes, dtype=np.float32)
        neighbor_deg_std = np.zeros(num_nodes, dtype=np.float32)
        for i, node in enumerate(nodes):
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 0:
                neighbor_degs = [G.degree(n) for n in neighbors]
                neighbor_deg_mean[i] = np.mean(neighbor_degs)
                neighbor_deg_std[i] = np.std(neighbor_degs) if len(neighbor_degs) > 1 else 0
        
        # Normalize
        degrees = degrees / max_deg
        neighbor_deg_mean = neighbor_deg_mean / max_deg
        neighbor_deg_std = neighbor_deg_std / max_deg
        
        # Stack features
        features = np.column_stack([
            degrees,
            clustering,
            neighbor_deg_mean,
            neighbor_deg_std
        ])
        
        # Pad to feature_dim if needed
        if features.shape[1] < feature_dim:
            padding = np.random.randn(num_nodes, feature_dim - features.shape[1]).astype(np.float32) * 0.01
            features = np.hstack([features, padding])
        
        return torch.FloatTensor(features[:, :feature_dim])
    
    def run_dmon(
        self,
        G: nx.Graph,
        num_communities: int,
        hidden_dim: int = 64,
        epochs: int = 200,
        lr: float = 0.01
    ) -> AlgorithmResult:
        """
        Run DMoN (Deep Modularity Network) with improved loss function.
        """
        if not self.torch_available:
            return AlgorithmResult(
                name='dmon',
                communities=[],
                runtime=0.0,
                success=False,
                error_message="PyTorch not available"
            )
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        
        class DMoN(nn.Module):
            def __init__(self, in_dim, hidden_dim, num_clusters):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.cluster = nn.Linear(hidden_dim, num_clusters)
                # Initialize cluster layer with larger weights
                nn.init.xavier_uniform_(self.cluster.weight, gain=2.0)
            
            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, p=0.3, training=self.training)
                x = F.relu(self.conv2(x, edge_index))
                # Use temperature scaling for sharper assignments
                s = F.softmax(self.cluster(x) * 2.0, dim=-1)
                return s
        
        start_time = time.time()
        
        try:
            data = self._graph_to_pyg(G, feature_dim=32)
            data = data.to(self.device)
            
            model = DMoN(32, hidden_dim, num_communities).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            
            # Compute adjacency for modularity loss
            adj = torch.zeros((data.num_nodes, data.num_nodes), device=self.device)
            adj[data.edge_index[0], data.edge_index[1]] = 1
            
            # Precompute degree matrix
            d = adj.sum(dim=1, keepdim=True)
            m = adj.sum() / 2  # Each edge counted twice
            
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                s = model(data.x, data.edge_index)
                
                # Modularity loss (maximize modularity = minimize negative)
                if m > 0:
                    B = adj - torch.mm(d, d.t()) / (2 * m)
                    modularity_loss = -torch.trace(torch.mm(torch.mm(s.t(), B), s)) / (2 * m)
                else:
                    modularity_loss = torch.tensor(0.0, device=self.device)
                
                # Collapse regularization: penalize uneven cluster sizes
                # Use entropy-based regularization
                cluster_sizes = s.sum(dim=0) / data.num_nodes
                entropy_loss = -torch.sum(cluster_sizes * torch.log(cluster_sizes + 1e-10))
                # We want to MAXIMIZE entropy (uniform sizes), so minimize negative
                collapse_loss = -entropy_loss / np.log(num_communities)  # Normalize
                
                # Orthogonality regularization: clusters should be distinct
                ss = torch.mm(s.t(), s) / data.num_nodes
                eye = torch.eye(num_communities, device=self.device)
                ortho_loss = torch.norm(ss - eye * ss.diag().unsqueeze(1))
                
                loss = modularity_loss + 1.0 * collapse_loss + 0.5 * ortho_loss
                loss.backward()
                optimizer.step()
            
            # Get assignments
            model.eval()
            with torch.no_grad():
                s = model(data.x, data.edge_index)
                assignments = s.argmax(dim=1).cpu().numpy()
            
            # Convert to communities
            communities = {}
            for node, comm in enumerate(assignments):
                if comm not in communities:
                    communities[comm] = []
                communities[comm].append(node)
            
            # Filter out empty communities
            communities = [c for c in communities.values() if len(c) > 0]
            
            runtime = time.time() - start_time
            
            return AlgorithmResult(
                name='dmon',
                communities=communities,
                runtime=runtime,
                success=True,
                metadata={'epochs': epochs}
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            return AlgorithmResult(
                name='dmon',
                communities=[],
                runtime=runtime,
                success=False,
                error_message=str(e)
            )
    
    def run_mincut(
        self,
        G: nx.Graph,
        num_communities: int,
        hidden_dim: int = 64,
        epochs: int = 200,
        lr: float = 0.01
    ) -> AlgorithmResult:
        """
        Run MinCut pooling for community detection with improved loss.
        """
        if not self.torch_available:
            return AlgorithmResult(
                name='mincut',
                communities=[],
                runtime=0.0,
                success=False,
                error_message="PyTorch not available"
            )
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        
        class MinCutPool(nn.Module):
            def __init__(self, in_dim, hidden_dim, num_clusters):
                super().__init__()
                self.conv1 = GCNConv(in_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.cluster = nn.Linear(hidden_dim, num_clusters)
                nn.init.xavier_uniform_(self.cluster.weight, gain=2.0)
            
            def forward(self, x, edge_index):
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, p=0.3, training=self.training)
                x = F.relu(self.conv2(x, edge_index))
                s = F.softmax(self.cluster(x) * 2.0, dim=-1)
                return s
        
        start_time = time.time()
        
        try:
            data = self._graph_to_pyg(G, feature_dim=32)
            data = data.to(self.device)
            
            model = MinCutPool(32, hidden_dim, num_communities).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            
            # Adjacency matrix
            adj = torch.zeros((data.num_nodes, data.num_nodes), device=self.device)
            adj[data.edge_index[0], data.edge_index[1]] = 1
            
            # Degree matrix for normalized cut
            d = adj.sum(dim=1)
            d_sqrt = torch.sqrt(d + 1e-10)
            
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                s = model(data.x, data.edge_index)
                
                # Normalized MinCut loss
                # Numerator: edges cut
                cut_edges = torch.trace(torch.mm(torch.mm(s.t(), adj), s))
                
                # Denominator: volume of clusters
                d_s = torch.mm(d.unsqueeze(0), s).squeeze()  # Degree sum per cluster
                cut_vol = (d_s * d_s).sum() + 1e-10
                
                # Minimize ratio cut
                mincut_loss = -cut_edges / cut_vol
                
                # Orthogonality: soft cluster assignments should be orthogonal
                ss = torch.mm(s.t(), s) / data.num_nodes
                eye = torch.eye(num_communities, device=self.device) / num_communities
                ortho_loss = torch.norm(ss - torch.diag(ss.diag()))
                
                # Balance loss: encourage similar sized clusters
                cluster_sizes = s.sum(dim=0) / data.num_nodes
                balance_loss = torch.var(cluster_sizes) * num_communities
                
                loss = mincut_loss + 0.5 * ortho_loss + 0.5 * balance_loss
                loss.backward()
                optimizer.step()
            
            # Get assignments
            model.eval()
            with torch.no_grad():
                s = model(data.x, data.edge_index)
                assignments = s.argmax(dim=1).cpu().numpy()
            
            communities = {}
            for node, comm in enumerate(assignments):
                if comm not in communities:
                    communities[comm] = []
                communities[comm].append(node)
            
            # Filter empty
            communities = [c for c in communities.values() if len(c) > 0]
            
            runtime = time.time() - start_time
            
            return AlgorithmResult(
                name='mincut',
                communities=communities,
                runtime=runtime,
                success=True,
                metadata={'epochs': epochs}
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            return AlgorithmResult(
                name='mincut',
                communities=[],
                runtime=runtime,
                success=False,
                error_message=str(e)
            )
    
    def run_gnn_semisupervised(
        self,
        G: nx.Graph,
        num_communities: int,
        ground_truth: list = None,
        labeled_ratio: float = 0.1,
        hidden_dim: int = 64,
        epochs: int = 200,
        lr: float = 0.01
    ) -> AlgorithmResult:
        """
        Run semi-supervised GNN with learnable embeddings.
        
        This method achieves excellent results when some labels are provided.
        Uses learnable node embeddings + GCN message passing.
        
        Args:
            G: NetworkX graph
            num_communities: Number of communities
            ground_truth: List of communities (list of node lists) for semi-supervision
            labeled_ratio: Fraction of nodes to use as labeled (default 10%)
            hidden_dim: Hidden dimension
            epochs: Training epochs
            lr: Learning rate
        """
        if not self.torch_available:
            return AlgorithmResult(
                name='gnn_semisupervised',
                communities=[],
                runtime=0.0,
                success=False,
                error_message="PyTorch not available"
            )
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        
        class SemiSupervisedGNN(nn.Module):
            def __init__(self, num_nodes, hidden_dim, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(num_nodes, hidden_dim)
                nn.init.xavier_uniform_(self.embedding.weight)
                self.conv1 = GCNConv(hidden_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.classifier = nn.Linear(hidden_dim, num_classes)
            
            def forward(self, edge_index):
                x = self.embedding.weight
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, p=0.5, training=self.training)
                x = self.conv2(x, edge_index)
                return self.classifier(x)
        
        start_time = time.time()
        
        try:
            num_nodes = G.number_of_nodes()
            nodes = list(G.nodes())
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            
            # Create edge index
            edges = list(G.edges())
            edge_index = torch.tensor([
                [node_to_idx[e[0]] for e in edges] + [node_to_idx[e[1]] for e in edges],
                [node_to_idx[e[1]] for e in edges] + [node_to_idx[e[0]] for e in edges]
            ], dtype=torch.long, device=self.device)
            
            # Create labels from ground truth
            if ground_truth is None:
                return AlgorithmResult(
                    name='gnn_semisupervised',
                    communities=[],
                    runtime=time.time() - start_time,
                    success=False,
                    error_message="Ground truth required for semi-supervised training"
                )
            
            # Convert ground truth to node labels
            gt_labels = np.zeros(num_nodes, dtype=np.int64)
            for i, comm in enumerate(ground_truth):
                for node in comm:
                    if node in node_to_idx:
                        gt_labels[node_to_idx[node]] = i
            
            y = torch.tensor(gt_labels, dtype=torch.long, device=self.device)
            
            # Create labeled mask (sample labeled_ratio from each community)
            labeled_mask = np.zeros(num_nodes, dtype=bool)
            for c in range(num_communities):
                c_nodes = np.where(gt_labels == c)[0]
                if len(c_nodes) > 0:
                    n_label = max(1, int(len(c_nodes) * labeled_ratio))
                    labeled_mask[np.random.choice(c_nodes, n_label, replace=False)] = True
            
            train_mask = torch.tensor(labeled_mask, device=self.device)
            
            if self.verbose:
                print(f"    Semi-supervised: {labeled_mask.sum()} labeled nodes ({100*labeled_mask.mean():.1f}%)")
            
            # Train
            model = SemiSupervisedGNN(num_nodes, hidden_dim, num_communities).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                out = model(edge_index)
                loss = F.cross_entropy(out[train_mask], y[train_mask])
                loss.backward()
                optimizer.step()
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                out = model(edge_index)
                assignments = out.argmax(dim=1).cpu().numpy()
            
            # Convert to communities
            communities = {}
            for idx, comm in enumerate(assignments):
                if comm not in communities:
                    communities[comm] = []
                communities[comm].append(nodes[idx])  # Use original node ID
            
            communities = [c for c in communities.values() if len(c) > 0]
            
            runtime = time.time() - start_time
            
            return AlgorithmResult(
                name='gnn_semisupervised',
                communities=communities,
                runtime=runtime,
                success=True,
                metadata={'epochs': epochs, 'labeled_ratio': labeled_ratio, 'n_labeled': int(labeled_mask.sum())}
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            return AlgorithmResult(
                name='gnn_semisupervised',
                communities=[],
                runtime=runtime,
                success=False,
                error_message=str(e)
            )
    
    def run_gnn_unsupervised_learnable(
        self,
        G: nx.Graph,
        num_communities: int,
        hidden_dim: int = 64,
        epochs: int = 300,
        lr: float = 0.01
    ) -> AlgorithmResult:
        """
        Run unsupervised GNN with learnable embeddings and modularity loss.
        
        Uses learnable node embeddings instead of spectral features.
        Optimizes modularity + entropy regularization.
        """
        if not self.torch_available:
            return AlgorithmResult(
                name='gnn_unsupervised',
                communities=[],
                runtime=0.0,
                success=False,
                error_message="PyTorch not available"
            )
        
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv
        
        class UnsupervisedGNN(nn.Module):
            def __init__(self, num_nodes, hidden_dim, num_clusters):
                super().__init__()
                self.embedding = nn.Embedding(num_nodes, hidden_dim)
                nn.init.xavier_uniform_(self.embedding.weight)
                self.conv1 = GCNConv(hidden_dim, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, hidden_dim)
                self.cluster = nn.Linear(hidden_dim, num_clusters)
            
            def forward(self, edge_index):
                x = self.embedding.weight
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, p=0.3, training=self.training)
                x = self.conv2(x, edge_index)
                s = F.softmax(self.cluster(x) * 2.0, dim=-1)
                return s
        
        start_time = time.time()
        
        try:
            num_nodes = G.number_of_nodes()
            nodes = list(G.nodes())
            node_to_idx = {n: i for i, n in enumerate(nodes)}
            
            # Create edge index
            edges = list(G.edges())
            edge_index = torch.tensor([
                [node_to_idx[e[0]] for e in edges] + [node_to_idx[e[1]] for e in edges],
                [node_to_idx[e[1]] for e in edges] + [node_to_idx[e[0]] for e in edges]
            ], dtype=torch.long, device=self.device)
            
            # Build adjacency for modularity
            adj = torch.zeros(num_nodes, num_nodes, device=self.device)
            adj[edge_index[0], edge_index[1]] = 1
            
            # Modularity matrix
            d = adj.sum(dim=1, keepdim=True)
            m = adj.sum() / 2
            B = adj - torch.mm(d, d.t()) / (2 * m)
            
            # Train
            model = UnsupervisedGNN(num_nodes, hidden_dim, num_communities).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                s = model(edge_index)
                
                # Modularity loss
                mod_loss = -torch.trace(torch.mm(torch.mm(s.t(), B), s)) / (2 * m)
                
                # Entropy regularization
                cluster_sizes = s.sum(dim=0) / num_nodes
                entropy = -torch.sum(cluster_sizes * torch.log(cluster_sizes + 1e-10))
                collapse_loss = -entropy / np.log(num_communities)
                
                loss = mod_loss + 0.5 * collapse_loss
                loss.backward()
                optimizer.step()
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                s = model(edge_index)
                assignments = s.argmax(dim=1).cpu().numpy()
            
            # Convert to communities
            communities = {}
            for idx, comm in enumerate(assignments):
                if comm not in communities:
                    communities[comm] = []
                communities[comm].append(nodes[idx])
            
            communities = [c for c in communities.values() if len(c) > 0]
            
            runtime = time.time() - start_time
            
            if self.verbose:
                print(f"    Unsupervised GNN: {len(communities)} communities found")
            
            return AlgorithmResult(
                name='gnn_unsupervised',
                communities=communities,
                runtime=runtime,
                success=True,
                metadata={'epochs': epochs}
            )
            
        except Exception as e:
            runtime = time.time() - start_time
            return AlgorithmResult(
                name='gnn_unsupervised',
                communities=[],
                runtime=runtime,
                success=False,
                error_message=str(e)
            )


if __name__ == '__main__':
    # Test the runner
    print("Testing Algorithm Runner...")
    
    # Create test graph
    G = nx.karate_club_graph()
    print(f"\nTest graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test classical algorithms
    runner = AlgorithmRunner(verbose=True)
    print(f"\nAvailable algorithms: {runner.get_available_algorithms()}")
    
    print("\n--- Testing Classical Algorithms ---")
    test_algos = ['louvain', 'leiden', 'label_propagation', 'spectral', 'fastgreedy']
    results = runner.run_all_algorithms(G, test_algos)
    
    print("\n--- Summary ---")
    for name, result in results.items():
        if result.success:
            print(f"{name}: {len(result.communities)} communities, {result.runtime:.3f}s")
        else:
            print(f"{name}: FAILED - {result.error_message}")
    
    # Test GNN (if available)
    print("\n--- Testing GNN Algorithms ---")
    gnn_runner = GNNRunner(device='cpu', verbose=True)
    if gnn_runner.torch_available:
        result = gnn_runner.run_dmon(G, num_communities=2, epochs=100)
        if result.success:
            print(f"DMoN: {len(result.communities)} communities, {result.runtime:.3f}s")
        else:
            print(f"DMoN: FAILED - {result.error_message}")

