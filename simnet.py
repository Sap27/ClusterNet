import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings

def denoise_network(W, C, lambda_, beta, max_iter=1000, tol=1e-5):
    """
    Optimized denoising with:
    - Original iterations (1000) and tolerance (1e-5) for mathematical equivalence
    - Efficient SVD (full_matrices=False) - no math difference
    - Safe division handling
    """
    n = W.shape[0]
    
    # Initialize S as a stochastic matrix (original random initialization)
    S = np.random.rand(n, n)
    row_sums = S.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    S = S / row_sums
    
    # Limit C to valid range
    C = min(C, n - 1)
    
    for iteration in range(max_iter):
        S_old = S.copy()
        
        # Update F (using efficient SVD)
        try:
            U, _, _ = svd(S - np.eye(n),full_matrices=False)
            F = U[:, :C]
        except:
            # If SVD fails, skip this iteration
            break
        
        # Update S with vectorized operations
        FtF = np.dot(F, F.T)
        gradient = -W + lambda_ * FtF + beta * S
        S = S - 0.01 * gradient
        
        # Ensure S is stochastic (non-negative and row-normalized)
        S = np.maximum(S, 0)
        row_sums = S.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        S = S / row_sums
        
        # Check for convergence
        if np.linalg.norm(S - S_old, 'fro') < tol:
            break
    
    # Symmetrize S
    S = (S + S.T) / 2
    return S

def run_simnet_pipeline(G, initial_clusters=28, cluster_size_threshold=50, denoise_lambda=1.0, denoise_beta=0.1, verbose=True):
    """
    Optimized pipeline with:
    - Progress tracking
    - Better subgraph extraction
    - Vectorized operations
    """
    # Convert NetworkX graph to an adjacency matrix
    original_nodes = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    if verbose:
        print(f"Processing network with {len(original_nodes)} nodes...")
    
    # Relabel the graph nodes with new sequential integers
    relabeled_G = nx.relabel_nodes(G, node_mapping)
    W = nx.to_numpy_array(relabeled_G)
    
    if verbose:
        print(f"Adjacency matrix shape: {W.shape}")
    
    # Step 1: Spectral Clustering for Initial Clusters
    if verbose:
        print(f"Running spectral clustering with {initial_clusters} clusters...")
    
    spectral = SpectralClustering(n_clusters=initial_clusters, affinity='precomputed', 
                                   assign_labels='discretize', random_state=42)
    initial_labels = spectral.fit_predict(W)
    
    final_clusters = []
    unique_labels = np.unique(initial_labels)
    
    if verbose:
        print(f"Found {len(unique_labels)} initial clusters")
    
    # Step 2: Process each cluster
    for i, cluster_id in enumerate(unique_labels):
        cluster_indices = np.where(initial_labels == cluster_id)[0]
        cluster_size = len(cluster_indices)
        
        if verbose and i % 5 == 0:
            print(f"Processing cluster {i+1}/{len(unique_labels)} (size: {cluster_size})...")
        
        if cluster_size <= cluster_size_threshold:
            # Save as a valid module
            final_clusters.append(cluster_indices.tolist())
        else:
            # Step 3: Extract subgraph and apply denoising
            subgraph = W[np.ix_(cluster_indices, cluster_indices)]
            
            # Adjust C parameter based on subgraph size
            C_adjusted = min(initial_clusters, subgraph.shape[0] - 1)
            denoised_subgraph = denoise_network(subgraph, C=C_adjusted, 
                                               lambda_=denoise_lambda, beta=denoise_beta)
            
            # Step 4: Apply Agglomerative Clustering
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
    
    # Vectorized mapping back to original node IDs
    return [[int(reverse_mapping[j]) for j in cluster] for cluster in final_clusters]

def read_edge_list(file_path):

    G = nx.read_edgelist(file_path, nodetype=int, data=(('weight', float),))
    return G

# Example usage:

# Option 1: Directly use a NetworkX graph
def process_network_file(network_file,weighted=True,directed=False):
    if weighted:
        if directed:
            G=nx.read_edgelist(network_file, data=(('weight', float),), nodetype=int,create_using=nx.DiGraph())
        else:
            G=nx.read_edgelist(network_file, data=(('weight', float),), nodetype=int)
    else:
        if directed:
            G=nx.read_edgelist(network_file, nodetype=int,create_using=nx.DiGraph())
        else:
            G=nx.read_edgelist(network_file, nodetype=int)
    #print(G.nodes)
    return G

if __name__ == "__main__":
    import time
    t1=[]
    for i in range(100):
        start_time = time.time()
        print("Loading network file...")
        G = process_network_file('network.dat')
        print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        print("\nRunning SimNet pipeline...")
        clusters = run_simnet_pipeline(G, initial_clusters=28, cluster_size_threshold=50, 
                                       denoise_lambda=1.0, denoise_beta=0.1, verbose=False)
        elapsed = time.time() - start_time
        t1.append(elapsed)
    print(f"Average time: {sum(t1)/len(t1):.2f} seconds")
    print(f"Standard deviation: {np.std(t1):.2f} seconds")
    print(f"Minimum time: {min(t1):.2f} seconds")
    print(f"Maximum time: {max(t1):.2f} seconds")
    print(f"Median time: {np.median(t1):.2f} seconds")
    
    # start_time = time.time()
    # print("Loading network file...")
    # G = process_network_file('network.dat')
    # print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # print("\nRunning SimNet pipeline...")
    # clusters = run_simnet_pipeline(G, initial_clusters=28, cluster_size_threshold=50, 
    #                                denoise_lambda=1.0, denoise_beta=0.1, verbose=False)
    
    # elapsed = time.time() - start_time
    # print(f"\n{'='*50}")
    # print(f"Total clusters found: {len(clusters)}")
    # print(f"Time elapsed: {elapsed:.2f} seconds")
    # print(f"{'='*50}")
# rc=realcommunities=rc_.groundtruth('C:/Users/hp/Mu_0.60/community.dat')
# print(score_benchmarks.nmi_score(clusters,rc))"""

# Option 2: Read from an edge list file
# G = read_edge_list("path_to_edge_list.txt")
# clusters = community_detection_pipeline(G)
# print(clusters)