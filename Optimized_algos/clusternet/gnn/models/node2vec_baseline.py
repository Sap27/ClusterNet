"""
Node2Vec + KMeans Baseline for Community Detection

Topology-only baseline: learns node embeddings from biased random walks
using skip-gram with negative sampling, then clusters with KMeans.
No node features are used — this tests whether graph structure alone is
sufficient for community detection.

Original paper: Grover & Leskovec "node2vec: Scalable Feature Learning
for Networks" (KDD 2016)

Loss: Skip-gram + negative sampling on random walk co-occurrences.
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.nn import Node2Vec as PyGNode2Vec
from torch_geometric.utils import from_networkx
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('node2vec', aliases=['node2vec_clustering', 'node2vec_kmeans'])
class Node2VecCluster(BaseAlgorithm):
    """
    Node2Vec-based community detection.

    Learns topology-only node embeddings via biased random walks and
    skip-gram, then clusters them with KMeans. Node features are ignored.

    Parameters:
        num_clusters: Number of clusters (auto-detected if None)
        embedding_dim: Embedding dimension (default: 64)
        walk_length: Random walk length (default: 20)
        context_size: Skip-gram window size (default: 10)
        walks_per_node: Number of walks starting from each node (default: 10)
        p: Return parameter (default: 1.0)
        q: In-out parameter (default: 1.0)
        epochs: Training epochs (default: 100)
        lr: Learning rate (default: 0.01)
        batch_size: Training batch size (default: 128)
        seed: Random seed (default: 42)

    Reference:
        Grover & Leskovec (2016). "node2vec: Scalable Feature Learning
        for Networks" (KDD)
    """

    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = True
    REQUIRES_FEATURES = False
    IS_TRAINABLE = True

    def __init__(
        self,
        G: nx.Graph,
        num_clusters: Optional[int] = None,
        embedding_dim: int = 64,
        walk_length: int = 20,
        context_size: int = 10,
        walks_per_node: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 128,
        seed: int = 42,
        device: Optional[str] = None,
        verbose: bool = False,
        **kwargs
    ):
        super().__init__(G, **kwargs)
        self.num_clusters = num_clusters or self._estimate_clusters()
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        if G.is_directed():
            G = G.to_undirected()
        pyg_data = from_networkx(G)
        self.edge_index = pyg_data.edge_index.to(self.device)
        self.num_nodes = G.number_of_nodes()

    def _estimate_clusters(self) -> int:
        try:
            from scipy.sparse.linalg import eigsh
            L = nx.laplacian_matrix(self.G).astype(float)
            k = min(20, self.G.number_of_nodes() - 2)
            eigenvalues, _ = eigsh(L, k=k, which='SM')
            eigenvalues = np.sort(eigenvalues)
            gaps = np.diff(eigenvalues[1:])
            num_clusters = np.argmax(gaps) + 2
            return max(2, min(num_clusters, int(np.sqrt(self.G.number_of_nodes()))))
        except Exception:
            return int(np.sqrt(self.G.number_of_nodes() / 10))

    def run(self) -> List[List]:
        torch.manual_seed(self.seed)

        model = PyGNode2Vec(
            self.edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            p=self.p,
            q=self.q,
            num_nodes=self.num_nodes,
        ).to(self.device)

        loader = model.loader(batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f'    epoch {epoch + 1:03d}/{self.epochs}, loss: {total_loss / len(loader):.4f}', flush=True)

        model.eval()
        with torch.no_grad():
            embeddings = model().cpu().numpy()

        kmeans = KMeans(n_clusters=self.num_clusters, n_init=10, random_state=self.seed)
        labels = kmeans.fit_predict(embeddings)

        nodes = list(self.G.nodes())
        communities = {}
        for i, label in enumerate(labels):
            label = int(label)
            if label not in communities:
                communities[label] = []
            communities[label].append(nodes[i])
        return list(communities.values())


def node2vec_clustering(
    G,
    num_clusters: Optional[int] = None,
    embedding_dim: int = 64,
    epochs: int = 100,
    lr: float = 0.01,
    **kwargs
) -> Tuple[List[List], float]:
    """
    Run Node2Vec clustering.

    Returns:
        communities: List of communities
        runtime: Training time in seconds
    """
    import time

    t1 = time.time()
    algo = Node2VecCluster(
        G,
        num_clusters=num_clusters,
        embedding_dim=embedding_dim,
        epochs=epochs,
        lr=lr,
        **kwargs
    )
    communities = algo.run()
    t2 = time.time()
    return communities, t2 - t1
