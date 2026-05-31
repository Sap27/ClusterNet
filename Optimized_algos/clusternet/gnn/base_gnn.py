"""
Base class for GNN-based community detection algorithms.

All GNN clustering methods inherit from this class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from abc import abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_dense_adj

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from clusternet.base import BaseAlgorithm


class BaseGNNClustering(BaseAlgorithm):
    """
    Abstract base class for GNN-based community detection.
    
    Extends BaseAlgorithm with GNN-specific functionality:
    - PyTorch model management
    - Training loops
    - Feature handling
    - GPU support
    
    Attributes:
        model: PyTorch GNN model
        device: torch.device (cuda/cpu)
        features: Node feature matrix
        epochs: Number of training epochs
        lr: Learning rate
    """
    
    SUPPORTS_DIRECTED = False  # Most GNNs work on undirected
    SUPPORTS_WEIGHTED = True
    REQUIRES_FEATURES = True   # GNNs need node features
    IS_TRAINABLE = True        # GNNs are trainable
    
    def __init__(
        self,
        G: nx.Graph,
        num_clusters: Optional[int] = None,
        features: Optional[np.ndarray] = None,
        hidden_channels: int = 32,
        epochs: int = 200,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        dropout: float = 0.5,
        device: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize GNN clustering algorithm.
        
        Args:
            G: NetworkX graph
            num_clusters: Number of clusters (auto-detected if None)
            features: Node feature matrix (generated if None)
            hidden_channels: Hidden layer dimension
            epochs: Training epochs
            lr: Learning rate
            weight_decay: L2 regularization
            dropout: Dropout rate
            device: 'cuda' or 'cpu' (auto-detected if None)
            verbose: Print training progress
            **kwargs: Additional parameters
        """
        super().__init__(G, **kwargs)
        
        # GNN-specific parameters
        self.num_clusters = num_clusters or self._estimate_clusters()
        self.hidden_channels = hidden_channels
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.verbose = verbose
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Convert graph to PyG format
        self.data = self._prepare_data(G, features)
        
        # Model will be initialized in subclass
        self.model = None
        self.optimizer = None
        
        # Store params
        self.params.update({
            'num_clusters': self.num_clusters,
            'hidden_channels': hidden_channels,
            'epochs': epochs,
            'lr': lr,
            'device': str(self.device)
        })
    
    def _estimate_clusters(self) -> int:
        """Estimate number of clusters using spectral gap."""
        try:
            import scipy.sparse as sp
            from scipy.sparse.linalg import eigsh
            
            L = nx.laplacian_matrix(self.G).astype(float)
            k = min(20, self.G.number_of_nodes() - 2)
            eigenvalues, _ = eigsh(L, k=k, which='SM')
            eigenvalues = np.sort(eigenvalues)
            
            # Find largest gap
            gaps = np.diff(eigenvalues[1:])  # Skip first (should be 0)
            num_clusters = np.argmax(gaps) + 2
            return max(2, min(num_clusters, int(np.sqrt(self.G.number_of_nodes()))))
        except:
            return int(np.sqrt(self.G.number_of_nodes() / 10))
    
    def _prepare_data(self, G: nx.Graph, features: Optional[np.ndarray] = None) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data.
        
        Args:
            G: NetworkX graph
            features: Optional node features
            
        Returns:
            PyG Data object
        """
        # Handle directed graphs
        if G.is_directed():
            G = G.to_undirected()
        
        # Get edge index
        pyg_data = from_networkx(G)
        
        # Handle features
        if features is not None:
            x = torch.tensor(features, dtype=torch.float)
        elif hasattr(pyg_data, 'x') and pyg_data.x is not None:
            x = pyg_data.x.float()
        else:
            # Generate features based on structural properties
            x = self._generate_structural_features(G)
        
        # Ensure correct shape
        if x.dim() == 1:
            x = x.unsqueeze(1)
        
        data = Data(
            x=x,
            edge_index=pyg_data.edge_index,
            num_nodes=G.number_of_nodes()
        )
        
        # Add edge weights if present
        if self.weighted:
            weights = []
            for u, v in G.edges():
                w = G[u][v].get('weight', 1.0)
                weights.append(w)
            if weights:
                # Make bidirectional
                weights = weights + weights
                data.edge_attr = torch.tensor(weights, dtype=torch.float)
        
        return data.to(self.device)
    
    def _generate_structural_features(self, G: nx.Graph, dim: int = 32) -> torch.Tensor:
        """
        Generate node features based on structural properties.
        
        Uses:
        - Degree
        - Clustering coefficient
        - PageRank
        - Random features per component
        
        Args:
            G: NetworkX graph
            dim: Feature dimension
            
        Returns:
            Feature tensor [num_nodes, dim]
        """
        n = G.number_of_nodes()
        features = np.zeros((n, dim))
        
        nodes = list(G.nodes())
        node_idx = {node: i for i, node in enumerate(nodes)}
        
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
        
        # Component-based random features (similar nodes get similar features)
        components = list(nx.connected_components(G))
        for i, comp in enumerate(components):
            # Each component gets a random base vector
            base = np.random.randn(dim - 3)
            for node in comp:
                # Add small noise to base
                features[node_idx[node], 3:] = base + 0.1 * np.random.randn(dim - 3)
        
        return torch.tensor(features, dtype=torch.float)
    
    @abstractmethod
    def _build_model(self) -> nn.Module:
        """
        Build the GNN model. Must be implemented by subclasses.
        
        Returns:
            PyTorch model
        """
        pass
    
    @abstractmethod
    def _train_step(self, data: Data) -> float:
        """
        Single training step. Must be implemented by subclasses.
        
        Args:
            data: PyG Data object
            
        Returns:
            Loss value
        """
        pass
    
    @abstractmethod
    def _get_cluster_assignments(self, data: Data) -> np.ndarray:
        """
        Get cluster assignments from trained model.
        
        Args:
            data: PyG Data object
            
        Returns:
            Array of cluster labels
        """
        pass
    
    def train(self) -> List[float]:
        """
        Train the GNN model.
        
        Returns:
            List of loss values per epoch
        """
        if self.model is None:
            self.model = self._build_model().to(self.device)
        
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        
        losses = []
        self.model.train()
        
        for epoch in range(self.epochs):
            loss = self._train_step(self.data)
            losses.append(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'    epoch {epoch + 1:03d}/{self.epochs}, loss: {loss:.4f}', flush=True)
        
        return losses
    
    def run(self) -> List[List]:
        """
        Run the GNN clustering algorithm.
        
        Returns:
            List of communities (list of node lists)
        """
        # Train model
        self.train()
        
        # Get cluster assignments
        self.model.eval()
        with torch.no_grad():
            labels = self._get_cluster_assignments(self.data)
        
        # Convert to community format
        communities = self._labels_to_communities(labels)
        
        return communities
    
    def _labels_to_communities(self, labels: np.ndarray) -> List[List]:
        """
        Convert cluster labels to list of communities.
        
        Args:
            labels: Array of cluster labels
            
        Returns:
            List of communities
        """
        nodes = list(self.G.nodes())
        communities = {}
        
        for i, label in enumerate(labels):
            label = int(label)
            if label not in communities:
                communities[label] = []
            communities[label].append(nodes[i])
        
        return list(communities.values())
    
    def get_embeddings(self) -> np.ndarray:
        """
        Get node embeddings from the trained model.
        
        Returns:
            Node embedding matrix [num_nodes, embedding_dim]
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call run() first.")
        
        self.model.eval()
        with torch.no_grad():
            # Get embeddings (implementation depends on model)
            embeddings = self._get_embeddings(self.data)
        
        return embeddings.cpu().numpy()
    
    def _get_embeddings(self, data: Data) -> torch.Tensor:
        """
        Get embeddings from model. Override in subclass if needed.
        """
        raise NotImplementedError("Subclass must implement _get_embeddings()")


def modularity_loss(adj: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Compute modularity-based loss for soft cluster assignments.
    
    Args:
        adj: Adjacency matrix [N, N]
        s: Soft cluster assignments [N, K]
        
    Returns:
        Negative modularity (to minimize)
    """
    # Normalize assignments
    s = F.softmax(s, dim=-1)
    
    # Degree matrix
    d = adj.sum(dim=1)
    m = adj.sum() / 2
    
    # Expected edges under null model
    expected = torch.outer(d, d) / (2 * m)
    
    # Modularity matrix
    B = adj - expected
    
    # Modularity
    Q = torch.trace(s.T @ B @ s) / (2 * m)
    
    return -Q  # Negative because we minimize


def orthogonality_loss(s: torch.Tensor) -> torch.Tensor:
    """
    Encourage orthogonal cluster assignments.
    
    Args:
        s: Cluster assignments [N, K]
        
    Returns:
        Orthogonality loss
    """
    s_norm = F.normalize(s, p=2, dim=0)
    identity = torch.eye(s.shape[1], device=s.device)
    return torch.norm(s_norm.T @ s_norm - identity)


def cluster_size_loss(s: torch.Tensor, min_size: int = 5) -> torch.Tensor:
    """
    Penalize clusters that are too small.
    
    Args:
        s: Cluster assignments [N, K]
        min_size: Minimum cluster size
        
    Returns:
        Size penalty loss
    """
    cluster_sizes = F.softmax(s, dim=-1).sum(dim=0)
    penalty = F.relu(min_size - cluster_sizes).sum()
    return penalty









