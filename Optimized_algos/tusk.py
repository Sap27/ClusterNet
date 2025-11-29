import networkx as nx
import numpy as np
import igraph as ig
from sklearn.cluster import SpectralClustering, KMeans
from scipy.spatial.distance import pdist, squareform
import warnings

# ==========================================
# PART 1: DSD CORE LOGIC
# Replaces dsd_gen.py and capDSD.dsdcore
# ==========================================

def compute_dsd_matrix(adj_matrix, steps=7):
    """
    Calculates the Diffusion State Distance (DSD) matrix.
    It computes the L2 distance between the probability distributions
    of random walks of length 'steps'.
    """
    n = adj_matrix.shape[0]
    
    # 1. Degree Matrix and Transition Matrix
    degrees = np.sum(adj_matrix, axis=1)
    
    # Avoid division by zero for isolated nodes
    with np.errstate(divide='ignore'):
        inv_degree = 1.0 / degrees
    inv_degree[degrees == 0] = 0
    
    D_inv = np.diag(inv_degree)
    P = np.dot(D_inv, adj_matrix) # Transition matrix P = D^-1 * A
    
    # 2. Compute the Accumulation Matrix (Heuristic for DSD)
    # We sum the probabilities of being at node j starting from i over k steps.
    current_P = np.eye(n)
    accumulation = np.zeros((n, n))
    
    for _ in range(steps):
        current_P = np.dot(current_P, P)
        accumulation += current_P
        
    # 3. Pairwise Euclidean Distance between rows (DSD metric)
    dsd_matrix = squareform(pdist(accumulation, metric='euclidean'))
    
    return dsd_matrix

# ==========================================
# PART 2: CLUSTERING ALGORITHMS
# Replaces clustering_algs_ig and generate_clusters.py
# ==========================================

def run_spectral_clustering(dsd_matrix, n_clusters):
    """
    Runs spectral clustering on the DSD matrix.
    Since DSD is a distance matrix, we convert it to an affinity matrix first.
    """
    if n_clusters < 2:
        return [list(range(dsd_matrix.shape[0]))]

    # Convert Distance -> Affinity
    # Standard RBF kernel: exp(-gamma * d^2)
    delta = np.std(dsd_matrix)
    if delta == 0: delta = 1.0
    affinity_matrix = np.exp(- (dsd_matrix ** 2) / (2. * delta ** 2))
    
    try:
        sc = SpectralClustering(n_clusters=n_clusters, 
                                affinity='precomputed', 
                                assign_labels='discretize',
                                random_state=42)
        labels = sc.fit_predict(affinity_matrix)
    except Exception:
        # Fallback to K-Means if Spectral fails due to matrix issues
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(dsd_matrix)

    # Convert labels to list of lists (clusters)
    clusters = {}
    for node_idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node_idx)
        
    return list(clusters.values())

# ==========================================
# PART 3: SPLITTING LOGIC
# Replaces split_clusters.py
# ==========================================

def recursive_split(dsd_matrix, cluster_indices, max_size=100):
    """
    Recursively splits clusters larger than max_size using sub-clustering.
    """
    final_clusters = []
    stack = [cluster_indices]
    
    while stack:
        current_cluster = stack.pop(0)
        
        if len(current_cluster) <= max_size:
            final_clusters.append(current_cluster)
            continue
            
        # Determine number of sub-clusters
        cluster_size = len(current_cluster)
        k_sub = int(cluster_size / 100.0) if cluster_size > 200 else 2
        if k_sub < 2: k_sub = 2
        
        # Extract sub-matrix for DSD
        sub_dsd = dsd_matrix[np.ix_(current_cluster, current_cluster)]
        
        # Run clustering on this subset
        sub_results = run_spectral_clustering(sub_dsd, k_sub)
        
        # Map local indices back to global indices
        for sub_c in sub_results:
            global_indices = [current_cluster[i] for i in sub_c]
            if len(global_indices) > max_size:
                stack.append(global_indices)
            else:
                final_clusters.append(global_indices)
                
    return final_clusters

# ==========================================
# PART 4: MAIN WRAPPER (Entry Point)
# ==========================================

def tusk_clustering(input_data, k_components=20, is_directed=False):
    """
    The main entry point for the Team Tusk Clustering Algorithm.
    
    Args:
        input_data: Either a NetworkX Graph object OR a path to an edgelist file.
        k_components: The target number of clusters (approximate).
        is_directed: Boolean, true if graph is directed.
        
    Returns:
        list of list: [['node1', 'node2'], ['node3', ...]]
    """
    
    # --- 1. INPUT HANDLING ---
    if isinstance(input_data, str):
        # Determine correct graph type
        create_using = nx.DiGraph() if is_directed else nx.Graph()
        
        try:
            # Attempt 1: Try reading as Weighted (3 columns)
            G = nx.read_edgelist(input_data, create_using=create_using, nodetype=str, data=(('weight', float),))
        except TypeError:
            # Attempt 2: Fallback to Unweighted (2 columns)
            # We must re-instantiate create_using to ensure clean state
            create_using = nx.DiGraph() if is_directed else nx.Graph()
            G = nx.read_edgelist(input_data, create_using=create_using, nodetype=str)
            
    elif isinstance(input_data, (nx.Graph, nx.DiGraph)):
        G = input_data
    else:
        raise ValueError("Input must be a file path string or a NetworkX Graph object.")

    # --- 2. PRE-PROCESSING ---
    node_list = list(G.nodes())
    
    # Map node names to integers 0..N
    node_map = {node: i for i, node in enumerate(node_list)}
    inv_map = {i: node for i, node in enumerate(node_list)}
    
    adj_matrix = nx.to_numpy_array(G, nodelist=node_list)
    
    # Handle unconnected graphs by processing components separately
    # (We use igraph here just for efficient component separation logic)
    mode = ig.ADJ_DIRECTED if is_directed else ig.ADJ_UNDIRECTED
    ig_G = ig.Graph.Adjacency(adj_matrix.tolist(), mode=mode)
    components = ig_G.components(mode="weak")
        
    final_communities = []
    
    # --- 3. CORE LOOP ---
    for component in components:
        # Get indices of nodes in this component
        comp_indices = list(component)
        
        # If component is too small, just add it as a community
        if len(comp_indices) < 3:
            final_communities.append([inv_map[i] for i in comp_indices])
            continue
            
        # Extract Sub-Matrix for this component
        comp_adj = adj_matrix[np.ix_(comp_indices, comp_indices)]
        
        # A. Calculate DSD
        dsd_mat = compute_dsd_matrix(comp_adj)
        
        # B. Initial Clustering
        # Dynamic k: if component is small, use fewer clusters
        local_k = max(2, int(k_components * (len(comp_indices) / len(node_list))))
        if local_k > len(comp_indices): local_k = len(comp_indices) // 2
        
        initial_clusters = run_spectral_clustering(dsd_mat, local_k)
        
        # C. Recursive Split
        for cluster_idx_list in initial_clusters:
            # Extract sub-DSD for the specific cluster
            sub_dsd_indices = cluster_idx_list 
            sub_dsd = dsd_mat[np.ix_(sub_dsd_indices, sub_dsd_indices)]
            
            # Run the recursive splitter
            # Note: We pass indices relative to the sub_dsd (0 to len-1)
            sub_splits = recursive_split(sub_dsd, list(range(len(sub_dsd_indices))))
            
            # D. Output Formatting (Map back to Strings)
            for sub in sub_splits:
                real_nodes = []
                for s_idx in sub:
                    # s_idx -> index in cluster_idx_list
                    # cluster_idx_list -> index in comp_indices
                    # comp_indices -> index in original node_list (inv_map)
                    
                    original_comp_idx = cluster_idx_list[s_idx]
                    global_idx = comp_indices[original_comp_idx]
                    real_nodes.append(inv_map[global_idx])
                
                final_communities.append(real_nodes)

    return final_communities
