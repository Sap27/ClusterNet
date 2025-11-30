import networkx as nx
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from infomap import Infomap  # Infomap for community detection
from joblib import Parallel, delayed
import os # For file operations

# %%
# --- Original Helper Functions (Slightly Modified) ---

def load_network(file_path):
    """Loads the network from an edgelist file."""
    print(f"Loading network from {file_path}...")
    G = nx.read_edgelist(file_path, data=(('weight', float),), nodetype=int)
    print(f"Network loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G

def save_clusters(clusters, file_path):
    """Saves the final clusters to a text file."""
    print(f"Saving {len(clusters)} clusters to {file_path}...")
    with open(file_path, 'w') as f:
        for i, cluster in enumerate(clusters, 1):
            f.write(f"Cluster {i}: {cluster}\n")
    print("Save complete.")

def infomap_clustering(in_file='', weighted=True, directed=False, out_file='', G=None):
    """
    Performs Infomap clustering on a NetworkX graph or an input file.
    
    NOTE: The buggy 'out_file' logic from the original has been removed,
    as it referenced undefined variables.
    """
    if G is None:
        if not in_file:
            raise ValueError("Must provide either 'G' or 'in_file'")
        G = nx.DiGraph() if directed else nx.Graph()
        with open(in_file, 'r') as datafile:
            for line in datafile:
                g = line.strip().split() # Use split() for any whitespace
                if len(g) < 2: continue
                # Assuming format: source target [weight]
                source, target = int(g[0]), int(g[1])
                weight = float(g[2]) if len(g) > 2 and weighted else 1.0
                G.add_edge(source, target, weight=weight)
    
    cmd = "--two-level"
    if directed:
        cmd += " --directed"
    
    im = Infomap(cmd)
    
    if weighted:
        for source, target, data in G.edges(data=True):
            weight_val = data.get('weight', 1.0)
            im.add_link(int(source), int(target), float(weight_val))
    else:
        for source, target in G.edges():
            im.add_link(int(source), int(target)) 
    
    im.run()
    
    module_dict = defaultdict(list)
    for node in im.tree:
        if node.is_leaf:
            module_id = node.module_id
            module_dict[module_id].append(node.node_id)
            
    communities = list(module_dict.values())
    
    return communities

# %%
# --- New Optimized Functions (Parallel & Sparse) ---

def inverse_log_weighted_similarity_parallel(subgraph, nodes, n_jobs=-1):
    """
    Calculates the inverse log-weighted similarity matrix in parallel,
    returning a sparse CSR matrix.
    
    NOTE: This replicates the original logic, where the "strength" of a
    common neighbor is calculated based *only* on its connections
    *within the subgraph*.
    """
    n = len(nodes)
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # Pre-calculate subgraph strengths (log(weighted_degree)) once
    # This is critical to avoid re-calculating in the loop
    subgraph_log_strengths = {}
    for node in nodes:
        strength = sum(data.get('weight', 1.0) for _, _, data in subgraph.edges(node, data=True))
        if strength <= 1.0:
            subgraph_log_strengths[node] = np.inf  # Will result in 0 similarity
        else:
            subgraph_log_strengths[node] = np.log(strength)
    
    # Pre-fetch neighbor sets
    subgraph_neighbors = {node: set(subgraph.neighbors(node)) for node in nodes}

    # This is the parallel worker function
    def _calculate_row(i):
        row_indices = []
        col_indices = []
        data = []
        node1 = nodes[i]
        neighbors1 = subgraph_neighbors[node1]

        for j in range(i + 1, n):
            node2 = nodes[j]
            neighbors2 = subgraph_neighbors[node2]
            common_neighbors = neighbors1 & neighbors2
            
            similarity = 0.0
            if common_neighbors:
                for nbr in common_neighbors:
                    log_strength = subgraph_log_strengths.get(nbr, np.inf)
                    if log_strength > 0 and log_strength != np.inf:
                        similarity += 1.0 / log_strength
            
            if similarity > 0:
                row_indices.append(i)
                col_indices.append(j)
                data.append(similarity)
        return row_indices, col_indices, data

    # Run the row calculations in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_row)(i) for i in range(n)
    )

    # Combine results from all parallel workers
    all_rows = []
    all_cols = []
    all_data = []
    for res in results:
        if res: # Check if result is not None
            all_rows.extend(res[0])
            all_cols.extend(res[1])
            all_data.extend(res[2])
    
    if not all_data: # Handle empty matrix
        return sp.csr_matrix((n, n))

    # Create COO matrix (fast for building)
    sim_matrix = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(n, n))
    
    # Make it symmetric and convert to CSR (fast for math)
    sim_matrix = sim_matrix + sim_matrix.T
    return sim_matrix.tocsr()

def sparsify_matrix_sparse(similarity_matrix, percentile):
    """
    Sparsifies a Scipy sparse matrix based on a percentile threshold
    of its non-zero (and > 0) values.
    """
    if similarity_matrix.nnz == 0:
        return similarity_matrix # Nothing to do
    
    # Get only the non-zero data
    data = similarity_matrix.data
    
    # Find the threshold, replicating original logic [similarity_matrix > 0]
    positive_data = data[data > 0]
    if positive_data.size == 0:
        return similarity_matrix # No positive values
        
    threshold = np.percentile(positive_data, percentile)
    
    # Create a copy to modify
    new_matrix = similarity_matrix.copy()
    
    # Set values below threshold to 0
    new_matrix.data[new_matrix.data < threshold] = 0.0
    
    # Remove the new zeros from the sparse structure
    new_matrix.eliminate_zeros()
    return new_matrix

# This is the "worker" function that processes one large cluster
def _process_cluster(cluster, G, size_threshold, min_size, percentile, directed, weighted, n_jobs_inner=1):
    """
    This function handles the full recursive logic for a single cluster.
    It will be called in parallel by the main recursive function.
    """
    if len(cluster) < min_size:
        return []  # Discard clusters smaller than min_size
    
    if len(cluster) <= size_threshold:
        return [cluster]  # Keep clusters below threshold
    
    # --- This cluster is large and needs processing ---
    # print(f"Processing cluster with {len(cluster)} nodes...") # Debug print
    
    # 1. Get subgraph (NetworkX)
    subgraph = G.subgraph(cluster)
    nodes = list(subgraph.nodes()) # Get a fixed order
    
    if subgraph.number_of_edges() == 0:
        return [cluster] # Isolated nodes, cannot cluster further
        
    # 2. Calculate sparse similarity matrix (Parallel)
    similarity_matrix = inverse_log_weighted_similarity_parallel(subgraph, nodes, n_jobs=n_jobs_inner)
    
    if similarity_matrix.nnz == 0:
        return [cluster] # No similarity, cannot cluster further

    # 3. Sparsify (Sparse)
    sparsified_matrix = sparsify_matrix_sparse(similarity_matrix, percentile)
    
    if sparsified_matrix.nnz == 0:
        return [cluster] # Sparsification removed all links

    # 4. Perform Infomap clustering on the *new* sparse graph
    # We feed the sparse matrix directly into Infomap
    cmd = "--two-level" + (" --directed" if directed else "")
    im = Infomap(cmd)
    
    # Map sparse matrix indices (0..n-1) back to original node IDs
    sparsified_matrix_coo = sparsified_matrix.tocoo()
    for r, c, w in zip(sparsified_matrix_coo.row, sparsified_matrix_coo.col, sparsified_matrix_coo.data):
        if r < c: # Add links only once for undirected graph
            node1_id = nodes[r]
            node2_id = nodes[c]
            im.add_link(int(node1_id), int(node2_id), float(w))
    
    im.run()
    
    if im.num_top_modules <= 1:
        # Infomap didn't find subgroups
        return [cluster]

    # 5. Extract new clusters
    module_dict = defaultdict(list)
    for node in im.tree:
        if node.is_leaf:
            module_dict[node.module_id].append(node.node_id)
    new_clusters = list(module_dict.values())

    # 6. Recurse: Process the new sub-clusters
    final_sub_clusters = []
    for new_cluster in new_clusters:
        # Recursion happens here, within the worker
        final_sub_clusters.extend(
            _process_cluster(new_cluster, G, size_threshold, min_size, percentile, directed, weighted, n_jobs_inner)
        )
    
    return final_sub_clusters

def recursive_sparsify_and_cluster_parallel(G, clusters, size_threshold=100, min_size=3, percentile=40, directed=False, weighted=True, n_jobs_outer=-1, n_jobs_inner=1):
    """
    Entry point for parallel recursive clustering.
    
    - n_jobs_outer: How many clusters to process in parallel.
    - n_jobs_inner: How many cores to use *inside* each cluster's
                    similarity calculation.
    """
    final_clusters = []

    print(f"Recursively processing {len(clusters)} clusters...")
    print(f"Using n_jobs_outer={n_jobs_outer} (parallel clusters), n_jobs_inner={n_jobs_inner} (parallel similarity calc)")

    # Use Joblib to parallelize the processing of the *initial* large clusters
    results = Parallel(n_jobs=n_jobs_outer)(
        delayed(_process_cluster)(
            cluster, G, size_threshold, min_size, percentile, directed, weighted, n_jobs_inner
        )
        for cluster in clusters
    )

    # Flatten the list of lists returned by the parallel workers
    for cluster_list in results:
        final_clusters.extend(cluster_list)
        
    return final_clusters

# %%
# --- Main Execution ---

def main(network_file='', output_file=None, G=None, directed=False, weighted=True, recursive=True, 
         size_threshold=100, min_size=3, percentile=40,
         n_jobs_outer=-1, n_jobs_inner=1):
    
    if network_file:
        G = load_network(network_file)
    elif G is None:
        raise ValueError("Must provide either a network_file or a NetworkX Graph object 'G'")

    # 1. Perform initial clustering
    print("Performing initial Infomap clustering...")
    cluster_assignment = infomap_clustering(G=G, directed=directed, weighted=weighted)
    print(f"Found {len(cluster_assignment)} initial clusters.")
    
    if recursive:
        # 2. Call the new parallel recursive function
        final_clusters = recursive_sparsify_and_cluster_parallel(
            G, 
            cluster_assignment,
            size_threshold=size_threshold,
            min_size=min_size,
            percentile=percentile,
            directed=directed, 
            weighted=weighted,
            n_jobs_outer=n_jobs_outer,
            n_jobs_inner=n_jobs_inner
        )
        print(f"Refined to {len(final_clusters)} final clusters.")
    else:
        final_clusters = cluster_assignment

    # 3. Save the final clusters
    if output_file:
        save_clusters(final_clusters, output_file)
    
    return final_clusters

if __name__ == "__main__":
    
    # --- Create a dummy network file for testing ---
    network_file = "network.dat"
    # print(f"Creating dummy file: {network_file}")
    # with open(network_file, 'w') as f:
    #     # A large cluster (0-199)
    #     for i in range(200):
    #         for j in range(i + 1, 200):
    #             if np.random.rand() < 0.05: # Sparse connections
    #                 f.write(f"{i} {j} {np.random.rand() + 1.0}\n")
    #     # A second cluster (200-300)
    #     for i in range(200, 301):
    #         for j in range(i + 1, 301):
    #             if np.random.rand() < 0.2: # Denser connections
    #                 f.write(f"{i} {j} {np.random.rand() + 1.0}\n")
    #     # Bridge nodes
    #     f.write(f"50 250 {np.random.rand()}\n")
    #     f.write(f"51 251 {np.random.rand()}\n")
    # # --- End of dummy file creation ---


    output_file = "modules_optimized.txt"

    # --- How to control parallelism ---
    #
    # Option 1 (Default): Best for many large clusters.
    # Use all available cores to process *different clusters* in parallel.
    # Each cluster's internal calculation runs on a single core.
    JOBS_OUTER = -1
    JOBS_INNER = 1
    
    # Option 2: Best for a few *very large* clusters.
    # Process one cluster at a time, but use all cores for its
    # internal similarity calculation.
    # JOBS_OUTER = 1
    # JOBS_INNER = -1
    
    # Option 3: Balanced approach (e.g., 4 cores)
    # Process 2 clusters at once, giving each 2 cores for internal calc.
    # JOBS_OUTER = 2
    # JOBS_INNER = 2 

    main(
        network_file=network_file,
        output_file=output_file,
        recursive=True,
        size_threshold=100, # Clusters > 100 nodes will be split
        min_size=3,
        percentile=40,
        n_jobs_outer=JOBS_OUTER,
        n_jobs_inner=JOBS_INNER
    )

    # Clean up dummy file
    # if os.path.exists(network_file):
    #     os.remove(network_file)
    if os.path.exists(output_file):
        print(f"\n--- Output file '{output_file}' ---")
        with open(output_file, 'r') as f:
            for _ in range(5): # Print first 5 lines
                try:
                    print(next(f), end='')
                except StopIteration:
                    break
        print("...")
        # os.remove(output_file) # Uncomment to auto-delete output
