"""
GPU-accelerated SimNet using CuPy for NVIDIA GPUs
Requires: pip install cupy-cuda11x (or cupy-cuda12x depending on your CUDA version)
"""

import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import svds as cp_svd
    GPU_AVAILABLE = True
    print("✓ GPU (CuPy) detected and loaded")
except ImportError as e:
    print(e)
    print("✗ CuPy not found. Install with: pip install cupy-cuda11x")
    print("  Falling back to CPU (NumPy)")
    cp = np
    GPU_AVAILABLE = False

def denoise_network_gpu(W, C, lambda_, beta, max_iter=1000, tol=1e-5, use_gpu=True):
    """
    GPU-accelerated denoising using CuPy
    
    Args:
        W: Adjacency matrix (numpy array)
        C: Number of components
        lambda_: Regularization parameter
        beta: Sparsity parameter
        max_iter: Maximum iterations
        tol: Convergence tolerance
        use_gpu: Whether to use GPU (if available)
    
    Returns:
        Denoised matrix (numpy array)
    """
    #print("hi")
    n = W.shape[0]
    
    # Decide whether to use GPU
    use_gpu = use_gpu and GPU_AVAILABLE
    xp = cp if use_gpu else np
    svd_func = cp_svd if use_gpu else np.linalg.svd
    
    if use_gpu:
        # Transfer to GPU
        W_gpu = cp.asarray(W)
        S = cp.random.rand(n, n)
    else:
        W_gpu = W
        S = np.random.rand(n, n)
    
    # Initialize S as stochastic matrix
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums = xp.where(row_sums == 0, 1, row_sums)
    S = S / row_sums
    
    # Limit C to valid range
    C = min(C, n - 1)
    
    # Create identity matrix
    I = xp.eye(n)
    
    for iteration in range(max_iter):
        S_old = S.copy()
        
        # Update F (using SVD on GPU)
        try:
            U, _, _ = svd_func(S - I)
            F = U[:, :C]
        except:
            # If SVD fails, break
            break
        
        # Update S with vectorized operations (on GPU)
        FtF = xp.dot(F, F.T)
        gradient = -W_gpu + lambda_ * FtF + beta * S
        S = S - 0.01 * gradient
        
        # Ensure S is stochastic
        S = xp.maximum(S, 0)
        row_sums = S.sum(axis=1, keepdims=True)
        row_sums = xp.where(row_sums == 0, 1, row_sums)
        S = S / row_sums
        
        # Check convergence
        if xp.linalg.norm(S - S_old, 'fro') < tol:
            break
    
    # Symmetrize S
    S = (S + S.T) / 2
    
    # Transfer back to CPU
    if use_gpu:
        S = cp.asnumpy(S)
    
    return S


def run_simnet_pipeline_gpu(G, initial_clusters=28, cluster_size_threshold=50, 
                            denoise_lambda=1.0, denoise_beta=0.1, 
                            use_gpu=True, verbose=True):
    """
    GPU-accelerated SimNet pipeline
    
    Note: Spectral clustering and agglomerative clustering still run on CPU
    (sklearn doesn't have GPU support), but the expensive denoising step runs on GPU
    """
    # Convert NetworkX graph to adjacency matrix
    original_nodes = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    if verbose:
        gpu_status = "GPU" if (use_gpu and GPU_AVAILABLE) else "CPU"
        print(f"Processing network with {len(original_nodes)} nodes using {gpu_status}...")
    
    # Relabel nodes
    relabeled_G = nx.relabel_nodes(G, node_mapping)
    W = nx.to_numpy_array(relabeled_G)
    
    if verbose:
        print(f"Adjacency matrix shape: {W.shape}")
    
    # Step 1: Spectral Clustering (runs on CPU - sklearn limitation)
    if verbose:
        print(f"Running spectral clustering with {initial_clusters} clusters (CPU)...")
    
    spectral = SpectralClustering(n_clusters=initial_clusters, affinity='precomputed',
                                   assign_labels='discretize', random_state=42)
    initial_labels = spectral.fit_predict(W)
    
    final_clusters = []
    unique_labels = np.unique(initial_labels)
    
    if verbose:
        print(f"Found {len(unique_labels)} initial clusters")
    
    # Step 2: Process each cluster
    large_clusters = 0
    for i, cluster_id in enumerate(unique_labels):
        cluster_indices = np.where(initial_labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        
        if verbose and i % 5 == 0:
            gpu_status = "GPU" if (use_gpu and GPU_AVAILABLE) else "CPU"
            print(f"Processing cluster {i+1}/{len(unique_labels)} (size: {cluster_size}) on {gpu_status}...")
        
        if cluster_size <= cluster_size_threshold:
            final_clusters.append(cluster_indices.tolist())
        else:
            large_clusters += 1
            # Step 3: Extract subgraph and apply GPU denoising
            subgraph = W[np.ix_(cluster_indices, cluster_indices)]
            
            C_adjusted = min(initial_clusters, subgraph.shape[0] - 1)
            denoised_subgraph = denoise_network_gpu(subgraph, C=C_adjusted,
                                                   lambda_=denoise_lambda, 
                                                   beta=denoise_beta,
                                                   use_gpu=use_gpu)
            
            # Step 4: Agglomerative Clustering (CPU - sklearn limitation)
            agglomerative = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5,
                                                    metric='euclidean', linkage='ward')
            agglomerative_labels = agglomerative.fit_predict(denoised_subgraph)
            
            for subcluster_id in np.unique(agglomerative_labels):
                subcluster_mask = agglomerative_labels == subcluster_id
                subcluster_indices = cluster_indices[subcluster_mask]
                subcluster_size = len(subcluster_indices)
                
                if 3 < subcluster_size < cluster_size_threshold:
                    final_clusters.append(subcluster_indices.tolist())
    
    if verbose:
        print(f"Final result: {len(final_clusters)} clusters")
        print(f"GPU-accelerated denoising performed on {large_clusters} large clusters")
    
    return [[int(reverse_mapping[j]) for j in cluster] for cluster in final_clusters]


def process_network_file(network_file, weighted=True, directed=False):
    """Load network from file"""
    if weighted:
        if directed:
            G = nx.read_edgelist(network_file, data=(('weight', float),), 
                               nodetype=int, create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(network_file, data=(('weight', float),), 
                               nodetype=int)
    else:
        if directed:
            G = nx.read_edgelist(network_file, nodetype=int, 
                               create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(network_file, nodetype=int)
    return G


if __name__ == "__main__":
    import time
    
    print("="*60)
    print("GPU-ACCELERATED SIMNET")
    print("="*60)
    
    # Check GPU availability
    # if GPU_AVAILABLE:
    #     print(f"GPU: {cp.cuda.Device().name}")
    #     print(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")
    #     print(f"GPU Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB")
    # else:
    #     print("Running on CPU (install CuPy for GPU acceleration)")
    
    print("="*60)
    
    # Load network
    print("\nLoading network file...")
    G = process_network_file('network.dat')
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Run GPU version
    print("\n" + "="*60)
    print("RUNNING GPU VERSION")
    print("="*60)
    start_time = time.time()
    clusters_gpu = run_simnet_pipeline_gpu(G, initial_clusters=28, 
                                          cluster_size_threshold=50,
                                          denoise_lambda=1.0, 
                                          denoise_beta=0.1,
                                          use_gpu=True, 
                                          verbose=True)
    gpu_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total clusters found: {len(clusters_gpu)}")
    print(f"GPU Time: {gpu_time:.2f} seconds")
    print("="*60)
    
    # Optional: Compare with CPU
    compare_cpu = True  # Set to True to benchmark CPU vs GPU
    if compare_cpu:
        print("\n" + "="*60)
        print("RUNNING CPU VERSION FOR COMPARISON")
        print("="*60)
        start_time = time.time()
        clusters_cpu = run_simnet_pipeline_gpu(G, initial_clusters=28,
                                              cluster_size_threshold=50,
                                              denoise_lambda=1.0,
                                              denoise_beta=0.1,
                                              use_gpu=False,
                                              verbose=True)
        cpu_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("CPU vs GPU COMPARISON")
        print("="*60)
        print(f"CPU Time: {cpu_time:.2f} seconds")
        print(f"GPU Time: {gpu_time:.2f} seconds")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        print(f"Same results: {len(clusters_cpu) == len(clusters_gpu)}")
        print("="*60)



