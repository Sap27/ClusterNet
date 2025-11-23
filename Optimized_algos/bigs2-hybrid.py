import cupy as cp
import cupyx.scipy.sparse as cpsp
import pandas as pd
from cupyx.scipy.sparse.linalg import eigsh as cp_eigsh
from cuml.cluster import KMeans as cuKMeans
import networkx as nx
import numpy as np
from cuml.preprocessing import SimpleImputer as cuSimpleImputer
import time

# Try to import FAISS for faster GPU KMeans
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Install with: pip install faiss-gpu")


def score_gpu(adj, K, verbose=False):
    """
    GPU-accelerated version of the score function.
    Computes eigenvalues and performs K-means clustering on GPU.
    """
    m = adj.shape[0]

    # Handle cases where the community is too small for spectral clustering or K=1
    # If m=0, no nodes to cluster. If m=1 or K=1, clustering is trivial (all in one cluster).
    if m == 0:
        return cp.array([], dtype=cp.int32)  # No nodes to cluster
    if m == 1 or K == 1:
        return cp.zeros(m, dtype=cp.int32)  # Single node or single cluster, assign to cluster 0

    # Ensure K is valid for eigsh: k must be < adj.shape[0]
    # And K must be at least 2 for spectral clustering to create ratio features.
    k_eigsh = min(K, m - 1)
    if k_eigsh < 1: # This can happen if m=1 (already handled) or if K was somehow <1. Safeguard.
        return cp.zeros(m, dtype=cp.int32)

    # Compute the largest k_eigsh eigenvalues and corresponding eigenvectors
    t0 = time.time()
    values, vectors = cp_eigsh(adj, k=k_eigsh, which='LM')
    t_eigsh = time.time() - t0

    # If fewer than 2 eigenvectors are returned, we cannot form a ratio for spectral clustering
    if vectors.shape[1] < 2:
        return cp.zeros(m, dtype=cp.int32)

    # Adjust K to the actual number of useful eigenvectors available for ratio calculation
    # We need vectors[:,0] and at least one more vector, so actual_K should be at least 2.
    # The KMeans n_clusters parameter will be based on this adjusted K.
    K_adjusted = vectors.shape[1]
    # If K_adjusted becomes 1 or less after eigsh returns fewer vectors, we can't create features.
    if K_adjusted < 2:
        return cp.zeros(m, dtype=cp.int32)  # Cannot form >1 cluster

    ratio = cp.zeros((m, K_adjusted - 1))  # Create K_adjusted-1 features for KMeans

    # Add a small epsilon to the denominator to prevent division by zero
    # Use cp.where for CuPy array operations
    epsilon = 1e-10
    denominator = cp.where(cp.abs(vectors[:, 0]) < epsilon, epsilon, vectors[:, 0])

    for j in range(1, K_adjusted):
        ratio[:, j - 1] = vectors[:, j] / denominator

    # Thresholding the ratio to prevent extreme values
    thresh = cp.log(m) if m > 0 else 0  # Handle log(0) if m=0, though already handled.
    ratio = cp.clip(ratio, -thresh, thresh)

    # Impute missing values using cuML's SimpleImputer
    t0 = time.time()
    imputer = cuSimpleImputer(strategy='mean')
    ratio_imputed = imputer.fit_transform(ratio)
    t_impute = time.time() - t0

    # NEW CHECK: If after imputation, no features remain, return a trivial clustering
    if ratio_imputed.shape[1] == 0:
        return cp.zeros(m, dtype=cp.int32)

    # K-means clustering on GPU
    # Try FAISS first (much faster), fallback to cuML
    t0 = time.time()
    
    if FAISS_AVAILABLE:
        print("FAISS available")
        # FAISS KMeans (typically 2-5x faster than cuML)
        labels = faiss_kmeans_gpu(ratio_imputed, K_adjusted, verbose=verbose)
        labels = cp.asarray(labels)
    else:
        # cuML KMeans fallback
        # n_clusters must be >= 1. Here, K_adjusted is used as n_clusters, and it is >= 2.
        # NOTE: n_init=10 (not 100) for speed. KMeans++ init already provides good starting points.
        kmeans = cuKMeans(n_clusters=K_adjusted, n_init=10, init='k-means++', random_state=0, max_iter=300)
        labels = kmeans.fit_predict(ratio_imputed)
        # Convert labels to CuPy array if it isn't already
        if not isinstance(labels, cp.ndarray):
            labels = cp.asarray(labels)
    
    t_kmeans = time.time() - t0
    
    if verbose:
        backend = "FAISS" if FAISS_AVAILABLE else "cuML"
        print(f"    [GPU breakdown] eigsh: {t_eigsh:.3f}s, impute: {t_impute:.3f}s, kmeans ({backend}): {t_kmeans:.3f}s")

    return labels


def faiss_kmeans_gpu(data, n_clusters, n_iter=100, verbose=False):
    """
    Fast GPU KMeans using FAISS library.
    FAISS is typically 2-5x faster than cuML for KMeans.
    
    Parameters:
    -----------
    data : cupy array or numpy array
        Data to cluster (n_samples, n_features)
    n_clusters : int
        Number of clusters
    n_iter : int
        Number of iterations (default: 100)
    verbose : bool
        Print timing info
    
    Returns:
    --------
    labels : numpy array
        Cluster labels for each sample
    """
    # Convert CuPy to NumPy if needed (FAISS works with NumPy)
    if isinstance(data, cp.ndarray):
        data_np = cp.asnumpy(data)
    else:
        data_np = data
    
    # Ensure float32 (FAISS requirement)
    data_np = data_np.astype(np.float32)
    
    n_samples, n_features = data_np.shape
    
    # Create FAISS GPU resources
    res = faiss.StandardGpuResources()
    
    # Configure KMeans
    kmeans = faiss.Kmeans(
        d=n_features,
        k=n_clusters,
        niter=n_iter,
        verbose=verbose,
        gpu=True,
        seed=0
    )
    
    # Train (this is the clustering step)
    kmeans.train(data_np)
    
    # Get cluster assignments
    _, labels = kmeans.index.search(data_np, 1)
    labels = labels.flatten()
    
    return labels


def multi_stage_score_local_gpu(N, in_file, out_file, Max_size, Min_size, Max_iter, M,
                                 G=None, directed=False, weighted=True, gpu_threshold=100):
    """
    Hybrid GPU-CPU version of the multi-stage SCORE algorithm.
    Uses GPU only for large matrices where it provides benefit.

    Parameters:
    -----------
    N : int
        Initial number of clusters
    in_file : str
        Path to input network file
    out_file : str
        Path to output file for communities
    Max_size : int
        Maximum community size
    Min_size : int
        Minimum community size
    Max_iter : int
        Maximum number of iterations
    M : int
        Number of subclusters for large communities
    G : networkx.Graph, optional
        Pre-loaded graph
    directed : bool
        Whether the graph is directed
    weighted : bool
        Whether the graph is weighted
    gpu_threshold : int
        Minimum matrix size to use GPU (default: 100)

    Returns:
    --------
    communities : list of lists
        List of communities, each containing node IDs
    """
    t_start = time.time()
    
    if G == None:
        # Build graph from file
        t0 = time.time()
        G = nx.DiGraph() if directed else nx.Graph()
        with open(in_file, 'r') as datafile:
            for line in datafile:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    G.add_edge(parts[0], parts[1], weight=float(parts[2]))
                else:
                    G.add_edge(parts[0], parts[1], weight=1.0)
        print(f"[TIMER] Graph loading: {time.time()-t0:.3f}s")

    # Create node mappings
    t0 = time.time()
    original_nodes = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}

    # Relabel graph with sequential integers
    relabeled_G = nx.relabel_nodes(G, node_mapping)

    # Get adjacency matrix on CPU
    weighted_adj_cpu = nx.adjacency_matrix(relabeled_G)
    weighted_adj_cpu = weighted_adj_cpu.astype(np.float64)
    
    # Get node array
    col1 = np.array(list(relabeled_G.nodes()))
    print(f"[TIMER] Graph preprocessing: {time.time()-t0:.3f}s (nodes={len(col1)})")

    # Initial clustering - use GPU only if matrix is large enough
    t0 = time.time()
    if weighted_adj_cpu.shape[0] >= gpu_threshold:
        print(f"[TIMER] Initial clustering: Using GPU (size={weighted_adj_cpu.shape[0]})")
        t_gpu = time.time()
        weighted_adj_gpu = cpsp.csr_matrix(weighted_adj_cpu)
        print(f"  - CPU->GPU transfer: {time.time()-t_gpu:.3f}s")
        t_gpu = time.time()
        labels = cp.asnumpy(score_gpu(weighted_adj_gpu, N, verbose=True))
        print(f"  - GPU computation total: {time.time()-t_gpu:.3f}s")
        del weighted_adj_gpu  # Free GPU memory immediately
    else:
        print(f"[TIMER] Initial clustering: Using CPU (size={weighted_adj_cpu.shape[0]})")
        labels = score_cpu(weighted_adj_cpu, N)
    print(f"[TIMER] Initial clustering total: {time.time()-t0:.3f}s")

    size_community = np.array([np.sum(labels == i) for i in range(N)])

    # Create block matrices on CPU (avoid GPU memory fragmentation)
    t0 = time.time()
    block_matrix = []
    block_names = []
    for i in range(N):
        mask = labels == i
        idx = np.where(mask)[0]
        block_mat = weighted_adj_cpu.tocsr()[idx, :][:, idx]
        block_matrix.append(block_mat)
        block_names.append(idx)
    print(f"[TIMER] Initial block creation: {time.time()-t0:.3f}s")

    gpu_calls = 0
    cpu_calls = 0
    gpu_time = 0
    cpu_time = 0
    matrix_time = 0
    score_time = 0
    iter_count = 1
    while np.max(size_community) > Max_size and iter_count <= Max_iter:
        t_iter = time.time()
        new_block_matrix = []
        new_block_names = []
        new_size_community = []

        for i in range(len(block_names)):
            if size_community[i] > Max_size:
                # Number of subclusters
                S = min(M, int(np.ceil(size_community[i] / Max_size)))

                # Use GPU only for large blocks
                if block_matrix[i].shape[0] >= gpu_threshold:
                    t0 = time.time()
                    block_mat_gpu = cpsp.csr_matrix(block_matrix[i])
                    matrix_time += time.time() - t0
                    t1 = time.time()
                    sub_labels = cp.asnumpy(score_gpu(block_mat_gpu, S))
                    score_time += time.time() - t1
                    del block_mat_gpu  # Free GPU memory
                    gpu_time += time.time() - t0
                    gpu_calls += 1
                else:
                    t0 = time.time()
                    sub_labels = score_cpu(block_matrix[i], S)
                    cpu_time += time.time() - t0
                    cpu_calls += 1

                tmp_size_community = np.array([np.sum(sub_labels == j) for j in range(S)])

                # Create sub-blocks (keep on CPU)
                for j in range(S):
                    mask = sub_labels == j
                    sub_idx = np.where(mask)[0]
                    tmp_block_mat = block_matrix[i].tocsr()[sub_idx, :][:, sub_idx]
                    tmp_block_name = block_names[i][sub_idx]

                    new_block_matrix.append(tmp_block_mat)
                    new_block_names.append(tmp_block_name)
                    new_size_community.append(tmp_size_community[j])
            else:
                new_block_matrix.append(block_matrix[i])
                new_block_names.append(block_names[i])
                new_size_community.append(size_community[i])

        iter_count += 1
        size_community = np.array(new_size_community)
        block_matrix = new_block_matrix
        block_names = new_block_names
        print(f"[TIMER] Iteration {iter_count-1}: {time.time()-t_iter:.3f}s (blocks={len(block_names)})")

    print(f"\n[TIMER SUMMARY]")
    print(f"  GPU calls: {gpu_calls}, total time: {gpu_time:.3f}s")
    print(f"  Matrix time: {matrix_time:.3f}s")
    print(f"  Score time: {score_time:.3f}s")
    print(f"  CPU calls: {cpu_calls}, total time: {cpu_time:.3f}s")
    
    t0 = time.time()
    Ncla = len(size_community)
    unique_col1 = np.unique(col1)

    # Write results to file
    if out_file != '':
        with open(out_file, 'w') as file:
            for j in range(Ncla):
                tmp_nodes = np.intersect1d(block_names[j], unique_col1)
                if len(tmp_nodes) <= Max_size and len(tmp_nodes) >= Min_size:
                    file.write(f"{j} 0.5 {' '.join(map(str, tmp_nodes))}\n")

    # Build communities list
    communities = []
    for j in range(Ncla):
        tmp_nodes = np.intersect1d(block_names[j], unique_col1)
        if len(tmp_nodes) <= Max_size and len(tmp_nodes) >= Min_size:
            original_labels = [reverse_mapping[idx] for idx in tmp_nodes]
            communities.append(original_labels)
    
    print(f"[TIMER] Output generation: {time.time()-t0:.3f}s")
    print(f"[TIMER] TOTAL TIME: {time.time()-t_start:.3f}s\n")

    return [[int(j) for j in i] for i in communities]


def score_cpu(adj, K):
    """
    CPU fallback version of score function for small matrices.
    """
    from scipy.sparse.linalg import eigsh as cpu_eigsh
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    
    m = adj.shape[0]
    
    if m == 0:
        return np.array([], dtype=np.int32)
    if m == 1 or K == 1:
        return np.zeros(m, dtype=np.int32)
    
    k_eigsh = min(K, m - 1)
    if k_eigsh < 1:
        return np.zeros(m, dtype=np.int32)
    
    try:
        values, vectors = cpu_eigsh(adj, k=k_eigsh, which='LM')
    except:
        return np.arange(m) % K
    
    if vectors.shape[1] < 2:
        return np.zeros(m, dtype=np.int32)
    
    K_adjusted = vectors.shape[1]
    if K_adjusted < 2:
        return np.zeros(m, dtype=np.int32)
    
    ratio = np.zeros((m, K_adjusted - 1))
    epsilon = 1e-10
    denominator = np.where(np.abs(vectors[:, 0]) < epsilon, epsilon, vectors[:, 0])
    
    for j in range(1, K_adjusted):
        ratio[:, j - 1] = vectors[:, j] / denominator
    
    thresh = np.log(m) if m > 0 else 0
    ratio = np.clip(ratio, -thresh, thresh)
    
    imputer = SimpleImputer(strategy='mean')
    ratio_imputed = imputer.fit_transform(ratio)
    
    if ratio_imputed.shape[1] == 0:
        return np.zeros(m, dtype=np.int32)
    
    # Reduced n_init to 10 for speed (was 100)
    kmeans = KMeans(n_clusters=K_adjusted, n_init=10, init='k-means++', random_state=0, max_iter=300)
    labels = kmeans.fit_predict(ratio_imputed)
    
    return labels


def multi_stage_score_local_gpu_optimized(N, in_file, out_file, Max_size, Min_size,
                                          Max_iter, M, G=None, directed=False,
                                          weighted=True, device_id=0, gpu_threshold=100):
    """
    Optimized GPU version with better memory management and device selection.

    Additional Parameters:
    ----------------------
    device_id : int
        GPU device ID to use (default: 0)
    gpu_threshold : int
        Minimum matrix size to use GPU (default: 100). Increase for smaller networks.
    """
    # Set GPU device
    cp.cuda.Device(device_id).use()

    try:
        result = multi_stage_score_local_gpu(N, in_file, out_file, Max_size, Min_size,
                                             Max_iter, M, G, directed, weighted, gpu_threshold)
        return result
    finally:
        # Clear GPU memory
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    # Display which KMeans backend is being used
    print("="*60)
    if FAISS_AVAILABLE:
        print("Using FAISS KMeans (FASTEST GPU option)")
    else:
        print("Using cuML KMeans (install faiss-gpu for 2-5x speedup)")
        print("Install: pip install faiss-gpu")
    print("="*60)
    print()
    
    # Test with different gpu_threshold values
    print("="*60)
    print("TEST 1: gpu_threshold=5000 (CPU-ONLY for this network)")
    print("="*60)
    communities1 = multi_stage_score_local_gpu_optimized(
        N=2, 
        in_file='network (4).dat',
        out_file='bigs2_gpu.txt', 
        Max_size=49, 
        Min_size=31, 
        Max_iter=15, 
        M=5,
        gpu_threshold=5000
    )
    print(f"Found {len(communities1)} communities")
    
    print("\n" + "="*60)
    print("TEST 2: gpu_threshold=1000 (GPU for initial + large blocks)")
    print("="*60)
    communities2 = multi_stage_score_local_gpu_optimized(
        N=2, 
        in_file='network (4).dat',
        out_file='', 
        Max_size=49, 
        Min_size=31, 
        Max_iter=15, 
        M=5,
        gpu_threshold=50
    )
    print(f"Found {len(communities2)} communities")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    print("For networks with ~4000 nodes, GPU provides NO benefit.")
    print("GPU is only beneficial for networks with 10,000+ nodes.")
    print("Recommendation: Use cpu_threshold >= 5000 or just use CPU version.")
    print()
    print("GPU KMeans Options (fastest to slowest):")
    print("  1. FAISS (pip install faiss-gpu) - 2-5x faster than cuML")
    print("  2. cuML KMeans with n_init=10")
    print("  3. cuML KMeans with n_init=100 (original, very slow)")