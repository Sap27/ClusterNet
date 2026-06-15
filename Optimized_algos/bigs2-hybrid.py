"""
BiGS2-Hybrid: Multi-stage SCORE algorithm with optional GPU acceleration.

CPU is used by default. GPU acceleration is optional and requires:
  - cupy: pip install cupy-cuda11x (or appropriate CUDA version)
  - cuml: conda install -c rapidsai cuml
  - faiss-gpu (optional, for faster KMeans): pip install faiss-gpu

Usage:
    # CPU only (default)
    communities = bigs2_clustering(G)
    
    # With GPU acceleration
    communities = bigs2_clustering(G, use_gpu=True, gpu_threshold=1000)
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh as cpu_eigsh
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import time

# Optional GPU imports
GPU_AVAILABLE = False
FAISS_AVAILABLE = False

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
    from cuml.cluster import KMeans as cuKMeans
    from cuml.preprocessing import SimpleImputer as cuSimpleImputer
    GPU_AVAILABLE = True
except ImportError:
    pass

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    pass


def check_gpu_available():
    """Check if GPU libraries are available."""
    return GPU_AVAILABLE


# =============================================================================
# CPU Implementation (Default)
# =============================================================================

def score_cpu(adj, K):
    """
    CPU implementation of SCORE algorithm.
    
    Parameters:
    -----------
    adj : scipy sparse matrix
        Adjacency matrix
    K : int
        Number of clusters
        
    Returns:
    --------
    labels : numpy array
        Cluster labels for each node
    """
    m = adj.shape[0]
    
    if m == 0:
        return np.array([], dtype=np.int32)
    if m == 1 or K == 1:
        return np.zeros(m, dtype=np.int32)
    
    k_eigsh = min(K, m - 1)
    if k_eigsh < 1:
        return np.zeros(m, dtype=np.int32)
    
    try:
        values, vectors = cpu_eigsh(adj.astype(np.float64), k=k_eigsh, which='LM')
    except:
        return np.arange(m) % max(1, K)
    
    if vectors.shape[1] < 2:
        return np.zeros(m, dtype=np.int32)
    
    K_adjusted = vectors.shape[1]
    if K_adjusted < 2:
        return np.zeros(m, dtype=np.int32)
    
    # Compute ratio features
    ratio = np.zeros((m, K_adjusted - 1))
    epsilon = 1e-10
    denominator = np.where(np.abs(vectors[:, 0]) < epsilon, epsilon, vectors[:, 0])
    
    for j in range(1, K_adjusted):
        ratio[:, j - 1] = vectors[:, j] / denominator
    
    # Threshold to prevent extreme values
    thresh = np.log(m) if m > 1 else 1
    ratio = np.clip(ratio, -thresh, thresh)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    ratio_imputed = imputer.fit_transform(ratio)
    
    if ratio_imputed.shape[1] == 0:
        return np.zeros(m, dtype=np.int32)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=K_adjusted, n_init=10, init='k-means++', 
                    random_state=0, max_iter=300)
    labels = kmeans.fit_predict(ratio_imputed)
    
    return labels


# =============================================================================
# GPU Implementation (Optional)
# =============================================================================

def score_gpu(adj, K, verbose=False):
    """
    GPU-accelerated SCORE algorithm using CuPy and cuML.
    
    Parameters:
    -----------
    adj : cupyx sparse matrix
        Adjacency matrix on GPU
    K : int
        Number of clusters
    verbose : bool
        Print timing information
        
    Returns:
    --------
    labels : cupy array
        Cluster labels for each node
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU libraries not available. Install cupy and cuml.")
    
    m = adj.shape[0]

    if m == 0:
        return cp.array([], dtype=cp.int32)
    if m == 1 or K == 1:
        return cp.zeros(m, dtype=cp.int32)

    k_eigsh = min(K, m - 1)
    if k_eigsh < 1:
        return cp.zeros(m, dtype=cp.int32)

    # Compute eigenvalues/vectors
    t0 = time.time()
    values, vectors = cp_eigsh(adj, k=k_eigsh, which='LM')
    t_eigsh = time.time() - t0

    if vectors.shape[1] < 2:
        return cp.zeros(m, dtype=cp.int32)

    K_adjusted = vectors.shape[1]
    if K_adjusted < 2:
        return cp.zeros(m, dtype=cp.int32)

    # Compute ratio features
    ratio = cp.zeros((m, K_adjusted - 1))
    epsilon = 1e-10
    denominator = cp.where(cp.abs(vectors[:, 0]) < epsilon, epsilon, vectors[:, 0])

    for j in range(1, K_adjusted):
        ratio[:, j - 1] = vectors[:, j] / denominator

    thresh = cp.log(m) if m > 1 else 1
    ratio = cp.clip(ratio, -thresh, thresh)

    # Impute missing values
    t0 = time.time()
    imputer = cuSimpleImputer(strategy='mean')
    ratio_imputed = imputer.fit_transform(ratio)
    t_impute = time.time() - t0

    if ratio_imputed.shape[1] == 0:
        return cp.zeros(m, dtype=cp.int32)

    # K-means clustering
    t0 = time.time()
    
    if FAISS_AVAILABLE:
        labels = _faiss_kmeans_gpu(ratio_imputed, K_adjusted)
        labels = cp.asarray(labels)
    else:
        kmeans = cuKMeans(n_clusters=K_adjusted, n_init=10, init='k-means++', 
                         random_state=0, max_iter=300)
        labels = kmeans.fit_predict(ratio_imputed)
        if not isinstance(labels, cp.ndarray):
            labels = cp.asarray(labels)
    
    t_kmeans = time.time() - t0
    
    if verbose:
        backend = "FAISS" if FAISS_AVAILABLE else "cuML"
        print(f"    [GPU] eigsh: {t_eigsh:.3f}s, impute: {t_impute:.3f}s, "
              f"kmeans ({backend}): {t_kmeans:.3f}s")

    return labels


def _faiss_kmeans_gpu(data, n_clusters, n_iter=100):
    """FAISS GPU KMeans (faster than cuML)."""
    if isinstance(data, cp.ndarray):
        data_np = cp.asnumpy(data)
    else:
        data_np = data
    
    data_np = data_np.astype(np.float32)
    n_features = data_np.shape[1]
    
    kmeans = faiss.Kmeans(d=n_features, k=n_clusters, niter=n_iter, 
                          verbose=False, gpu=True, seed=0)
    kmeans.train(data_np)
    _, labels = kmeans.index.search(data_np, 1)
    
    return labels.flatten()


# =============================================================================
# Main Algorithm
# =============================================================================

def bigs2_clustering(G, n_clusters=None, max_size=100, min_size=3, max_iter=15, 
                     m_subclusters=5, use_gpu=False, gpu_threshold=1000, 
                     verbose=False):
    """
    Multi-stage SCORE algorithm for community detection.
    
    Recursively splits large communities until all communities are within
    the size constraints.
    
    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    n_clusters : int, optional
        Initial number of clusters. If None, estimated from graph size.
    max_size : int
        Maximum community size (default: 100)
    min_size : int
        Minimum community size (default: 3)
    max_iter : int
        Maximum refinement iterations (default: 15)
    m_subclusters : int
        Number of subclusters when splitting (default: 5)
    use_gpu : bool
        Use GPU acceleration if available (default: False)
    gpu_threshold : int
        Minimum matrix size to use GPU (default: 1000)
    verbose : bool
        Print timing information (default: False)
        
    Returns:
    --------
    communities : list of lists
        List of communities, each containing node IDs
    """
    t_start = time.time()
    n_nodes = G.number_of_nodes()
    
    if n_nodes == 0:
        return []
    
    # Estimate initial clusters if not provided
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(n_nodes / 10)))
    
    # Check GPU availability
    if use_gpu and not GPU_AVAILABLE:
        print("Warning: GPU requested but not available. Using CPU.")
        use_gpu = False
    
    if verbose:
        mode = "GPU" if use_gpu else "CPU"
        print(f"BiGS2 clustering: {n_nodes} nodes, mode={mode}")
    
    # Create node mappings for sequential integer labels
    original_nodes = list(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(original_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    # Relabel graph with sequential integers
    G_relabeled = nx.relabel_nodes(G, node_to_idx)
    
    # Get adjacency matrix
    adj = nx.adjacency_matrix(G_relabeled).astype(np.float64)
    node_indices = np.array(list(G_relabeled.nodes()))
    
    # Initial clustering
    if use_gpu and adj.shape[0] >= gpu_threshold:
        adj_gpu = cpsp.csr_matrix(adj)
        labels = cp.asnumpy(score_gpu(adj_gpu, n_clusters, verbose=verbose))
        del adj_gpu
    else:
        labels = score_cpu(adj, n_clusters)
    
    # Initialize block structures
    size_community = np.array([np.sum(labels == i) for i in range(n_clusters)])
    
    block_matrix = []
    block_names = []
    for i in range(n_clusters):
        mask = labels == i
        idx = np.where(mask)[0]
        block_mat = adj.tocsr()[idx, :][:, idx]
        block_matrix.append(block_mat)
        block_names.append(idx)
    
    # Iteratively split large communities
    iter_count = 0
    while np.max(size_community) > max_size and iter_count < max_iter:
        iter_count += 1
        new_block_matrix = []
        new_block_names = []
        new_size_community = []
        
        for i in range(len(block_names)):
            if size_community[i] > max_size:
                # Split this community
                n_sub = min(m_subclusters, int(np.ceil(size_community[i] / max_size)))
                
                if use_gpu and block_matrix[i].shape[0] >= gpu_threshold:
                    block_gpu = cpsp.csr_matrix(block_matrix[i])
                    sub_labels = cp.asnumpy(score_gpu(block_gpu, n_sub, verbose=verbose))
                    del block_gpu
                else:
                    sub_labels = score_cpu(block_matrix[i], n_sub)
                
                # Create sub-blocks
                for j in range(n_sub):
                    mask = sub_labels == j
                    sub_idx = np.where(mask)[0]
                    if len(sub_idx) > 0:
                        sub_mat = block_matrix[i].tocsr()[sub_idx, :][:, sub_idx]
                        sub_names = block_names[i][sub_idx]
                        new_block_matrix.append(sub_mat)
                        new_block_names.append(sub_names)
                        new_size_community.append(len(sub_idx))
            else:
                # Keep this community as-is
                new_block_matrix.append(block_matrix[i])
                new_block_names.append(block_names[i])
                new_size_community.append(size_community[i])
        
        size_community = np.array(new_size_community)
        block_matrix = new_block_matrix
        block_names = new_block_names
        
        if verbose:
            print(f"  Iteration {iter_count}: {len(block_names)} blocks, "
                  f"max_size={np.max(size_community)}")
    
    # Build final communities with original node IDs
    communities = []
    for block_idx in block_names:
        if min_size <= len(block_idx) <= max_size:
            community = [idx_to_node[idx] for idx in block_idx]
            communities.append(community)
    
    if verbose:
        print(f"BiGS2 complete: {len(communities)} communities, "
              f"time={time.time()-t_start:.2f}s")
    
    # Clean up GPU memory if used
    if use_gpu and GPU_AVAILABLE:
        cp.get_default_memory_pool().free_all_blocks()
    
    return communities


# =============================================================================
# Wrapper for file-based input (backward compatibility)
# =============================================================================

def multi_stage_score_local(N, in_file, out_file, Max_size, Min_size, Max_iter, M,
                            G=None, directed=False, weighted=True, 
                            use_gpu=False, gpu_threshold=1000):
    """
    File-based interface for backward compatibility.
    
    Parameters:
    -----------
    N : int
        Initial number of clusters
    in_file : str
        Path to input network file (tab-separated: node1, node2, [weight])
    out_file : str
        Path to output file for communities
    Max_size : int
        Maximum community size
    Min_size : int
        Minimum community size
    Max_iter : int
        Maximum iterations
    M : int
        Number of subclusters for splitting
    G : networkx.Graph, optional
        Pre-loaded graph (if None, load from in_file)
    directed : bool
        Whether graph is directed (default: False)
    weighted : bool
        Whether graph is weighted (default: True)
    use_gpu : bool
        Use GPU acceleration (default: False)
    gpu_threshold : int
        Minimum matrix size for GPU (default: 1000)
        
    Returns:
    --------
    communities : list of lists
        List of communities
    """
    # Load graph if not provided
    if G is None:
        G = nx.DiGraph() if directed else nx.Graph()
        with open(in_file, 'r') as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and weighted:
                    G.add_edge(parts[0], parts[1], weight=float(parts[2]))
                elif len(parts) >= 2:
                    G.add_edge(parts[0], parts[1], weight=1.0)
    
    # Run clustering
    communities = bigs2_clustering(
        G,
        n_clusters=N,
        max_size=Max_size,
        min_size=Min_size,
        max_iter=Max_iter,
        m_subclusters=M,
        use_gpu=use_gpu,
        gpu_threshold=gpu_threshold,
        verbose=True
    )
    
    # Write output file
    if out_file:
        with open(out_file, 'w') as f:
            for j, comm in enumerate(communities):
                f.write(f"{j} 0.5 {' '.join(map(str, comm))}\n")
    
    return communities


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BiGS2-Hybrid: Multi-stage SCORE Community Detection")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")
    if GPU_AVAILABLE:
        print(f"FAISS Available: {FAISS_AVAILABLE}")
    print()
    
    # Test with a sample graph
    print("Testing with Karate Club graph...")
    G = nx.karate_club_graph()
    
    # CPU mode (default)
    print("\n1. CPU Mode (default):")
    communities = bigs2_clustering(G, verbose=True)
    print(f"   Found {len(communities)} communities")
    
    # GPU mode (if available)
    if GPU_AVAILABLE:
        print("\n2. GPU Mode:")
        communities = bigs2_clustering(G, use_gpu=True, gpu_threshold=10, verbose=True)
        print(f"   Found {len(communities)} communities")
    else:
        print("\n2. GPU Mode: Skipped (GPU not available)")
    
    print("\nDone!")
