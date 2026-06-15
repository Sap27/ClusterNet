"""
MLP + KMeans Baseline for Community Detection

Features-only baseline: clusters raw node features with KMeans, ignoring
all graph structure. If a GNN performs comparably to this baseline, the
GNN has learned nothing from the graph topology.

No neural network training — this is a pure feature clustering baseline.

Reference:
    Tanis et al. (2024) demonstrate that MLP baselines are critical for
    interpreting GNN results, especially on low-homophily graphs.
"""

import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm


@register_algorithm('mlp_kmeans', aliases=['mlp_baseline', 'features_only'])
class MLPKMeansBaseline(BaseAlgorithm):
    """
    Features-only community detection baseline.

    Applies KMeans directly to node features (optionally scaled), completely
    ignoring graph structure. This establishes a lower bound: any GNN that
    fails to beat this baseline is not learning from edges.

    Parameters:
        num_clusters: Number of clusters (required, or auto-estimated)
        features: Node feature matrix [N, F] (required)
        scale_features: Whether to standardize features before clustering
        n_init: Number of KMeans random restarts (default: 10)
        seed: Random seed for reproducibility
    """

    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    REQUIRES_FEATURES = True
    IS_TRAINABLE = False

    def __init__(
        self,
        G: nx.Graph,
        num_clusters: Optional[int] = None,
        features: Optional[np.ndarray] = None,
        scale_features: bool = True,
        n_init: int = 10,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(G, **kwargs)
        self.num_clusters = num_clusters or self._estimate_clusters()
        self.scale_features = scale_features
        self.n_init = n_init
        self.seed = seed

        if features is None:
            raise ValueError(
                "MLPKMeansBaseline requires node features. "
                "Pass features= to the constructor."
            )
        self.features = np.asarray(features, dtype=np.float64)

    def _estimate_clusters(self) -> int:
        """Estimate cluster count via spectral gap (same as BaseGNNClustering)."""
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
        X = self.features.copy()
        if self.scale_features:
            X = StandardScaler().fit_transform(X)

        kmeans = KMeans(
            n_clusters=self.num_clusters,
            n_init=self.n_init,
            random_state=self.seed,
        )
        labels = kmeans.fit_predict(X)

        nodes = list(self.G.nodes())
        communities = {}
        for i, label in enumerate(labels):
            label = int(label)
            if label not in communities:
                communities[label] = []
            communities[label].append(nodes[i])
        return list(communities.values())


def mlp_kmeans_clustering(
    G,
    num_clusters: Optional[int] = None,
    features: Optional[np.ndarray] = None,
    scale_features: bool = True,
    seed: int = 42,
    **kwargs
) -> Tuple[List[List], float]:
    """
    Run MLP+KMeans baseline clustering.

    Returns:
        communities: List of communities
        runtime: Clustering time in seconds
    """
    import time

    t1 = time.time()
    algo = MLPKMeansBaseline(
        G,
        num_clusters=num_clusters,
        features=features,
        scale_features=scale_features,
        seed=seed,
        **kwargs
    )
    communities = algo.run()
    t2 = time.time()
    return communities, t2 - t1
