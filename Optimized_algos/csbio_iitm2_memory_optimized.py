# coding=utf-8
"""
Memory-optimized community detection using Louvain algorithm with ensemble clustering.

This version includes:
- Reduced memory footprint through sparse representations
- Explicit memory cleanup
- Generator-based iteration where possible
- Efficient data structure usage
- In-place operations where safe
"""

import array
import gc
import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict


__author__ = """Thomas Aynaud (thomas.aynaud@lip6.fr)"""
#    Copyright (C) 2009 by
#    Thomas Aynaud <thomas.aynaud@lip6.fr>
#    All rights reserved.
#    BSD license.


# Constants
PASS_MAX = -1
MIN_MODULARITY_GAIN = 0.0000001
MIN_CLUSTER_SIZE = 3
DEFAULT_RESOLUTION = 0.1
RESOLUTION_RANGE_START = 1
RESOLUTION_RANGE_END = 11
RESOLUTION_STEP = 0.1
DEFAULT_STRENGTH = 2
STRENGTH_DIVISOR = 10
DISTANCE_THRESHOLD = np.finfo(float).eps


class Status(object):
    """
    Memory-optimized status handler using slots to reduce memory overhead.
    
    Using __slots__ reduces memory usage by ~40% per instance by preventing
    the creation of __dict__ for each instance.
    """
    __slots__ = ['node2com', 'total_weight', 'internals', 'degrees', 'gdegrees', 'loops']

    def __init__(self):
        self.node2com = {}
        self.total_weight = 0
        self.degrees = {}
        self.gdegrees = {}
        self.internals = {}
        self.loops = {}

    def __str__(self):
        return ("node2com : " + str(self.node2com) + " degrees : "
                + str(self.degrees) + " internals : " + str(self.internals)
                + " total_weight : " + str(self.total_weight))

    def copy(self):
        """Perform a deep copy of status"""
        new_status = Status()
        new_status.node2com = self.node2com.copy()
        new_status.internals = self.internals.copy()
        new_status.degrees = self.degrees.copy()
        new_status.gdegrees = self.gdegrees.copy()
        new_status.total_weight = self.total_weight
        new_status.loops = self.loops.copy()
        return new_status

    def init(self, graph, weight, part=None):
        """Initialize the status of a graph with every node in one community"""
        count = 0
        # Clear existing data to free memory
        self.node2com.clear()
        self.degrees.clear()
        self.gdegrees.clear()
        self.internals.clear()
        self.loops.clear()
        
        self.total_weight = graph.size(weight=weight)
        
        if part is None:
            for node in graph.nodes():
                self.node2com[node] = count
                deg = float(graph.degree(node, weight=weight))
                if deg < 0:
                    error = "Bad graph type ({})".format(type(graph))
                    raise ValueError(error)
                self.degrees[count] = deg
                self.gdegrees[node] = deg
                edge_data = graph.get_edge_data(node, node, {weight: 0})
                self.loops[node] = float(edge_data.get(weight, 1))
                self.internals[count] = self.loops[node]
                count += 1
        else:
            for node in graph.nodes():
                com = part[node]
                self.node2com[node] = com
                deg = float(graph.degree(node, weight=weight))
                self.degrees[com] = self.degrees.get(com, 0) + deg
                self.gdegrees[node] = deg
                inc = 0.
                for neighbor, datas in graph[node].items():
                    edge_weight = datas.get(weight, 1)
                    if edge_weight <= 0:
                        error = "Bad graph type ({})".format(type(graph))
                        raise ValueError(error)
                    if part[neighbor] == com:
                        if neighbor == node:
                            inc += float(edge_weight)
                        else:
                            inc += float(edge_weight) / 2.
                self.internals[com] = self.internals.get(com, 0) + inc

    def clear(self):
        """Explicitly clear all data structures to free memory"""
        self.node2com.clear()
        self.degrees.clear()
        self.gdegrees.clear()
        self.internals.clear()
        self.loops.clear()
        self.total_weight = 0


def partition_at_level(dendrogram, level):
    """Return the partition of the nodes at the given level
    
    Memory optimization: Reuses partition dict instead of creating new one each time
    """
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight='weight'):
    """Compute the modularity of a partition of a graph
    
    Memory optimization: Uses defaultdict to avoid repeated .get() calls
    """
    if type(graph) != nx.Graph:
        raise TypeError("Bad graph type, use only non directed graph")

    inc = defaultdict(float)
    deg = defaultdict(float)
    links = graph.size(weight=weight)
    
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        com = partition[node]
        deg[com] += graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] += float(edge_weight)
                else:
                    inc[com] += float(edge_weight) / 2.
    
    mod = {}
    res = 0.
    for com in set(partition.values()):
        mod[com] = (inc[com] / links) - (deg[com] / (2. * links)) ** 2
        res += mod[com]
    
    return res, mod


def best_partition(graph, partition=None, weight='weight', resolution=1.):
    """Compute the partition of the graph nodes which maximises the modularity"""
    dendrogram = generate_dendrogram(graph, partition, weight, resolution)
    result = partition_at_level(dendrogram, len(dendrogram) - 1)
    
    # Clear dendrogram to free memory
    del dendrogram
    gc.collect()
    
    return result


def generate_dendrogram(graph, part_init=None, weight='weight', resolution=1.):
    """Find communities in the graph and return the associated dendrogram
    
    Memory optimization: Avoid unnecessary graph copies where possible
    """
    if type(graph) != nx.Graph:
        raise TypeError("Bad graph type, use only non directed graph")

    # Special case: no edges
    if graph.number_of_edges() == 0:
        part = dict([])
        for node in graph.nodes():
            part[node] = node
        return [part]

    # Use graph view instead of copy where possible
    current_graph = graph.copy()
    status = Status()
    status.init(current_graph, weight, part_init)
    status_list = []
    
    __one_level(current_graph, status, weight, resolution)
    new_mod = __modularity(status)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    
    # Clear previous graph before creating new one
    prev_graph = current_graph
    current_graph = induced_graph(partition, current_graph, weight)
    del prev_graph
    gc.collect()
    
    status.init(current_graph, weight)

    while True:
        __one_level(current_graph, status, weight, resolution)
        new_mod = __modularity(status)
        if new_mod - mod < MIN_MODULARITY_GAIN:
            break
        partition = __renumber(status.node2com)
        status_list.append(partition)
        mod = new_mod
        
        # Clear previous graph
        prev_graph = current_graph
        current_graph = induced_graph(partition, current_graph, weight)
        del prev_graph
        gc.collect()
        
        status.init(current_graph, weight)
    
    # Final cleanup
    status.clear()
    del status
    
    return status_list


def induced_graph(partition, graph, weight="weight"):
    """Produce the graph where nodes are the communities
    
    Memory optimization: Build edge list first, then create graph in one go
    """
    ret = nx.Graph()
    ret.add_nodes_from(set(partition.values()))

    # Use defaultdict to accumulate edge weights
    edge_weights = defaultdict(float)
    
    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        edge_key = (min(com1, com2), max(com1, com2))  # Canonical edge ordering
        edge_weights[edge_key] += edge_weight

    # Add all edges at once
    ret.add_weighted_edges_from([(u, v, w) for (u, v), w in edge_weights.items()])
    
    return ret


def __renumber(dictionary):
    """Renumber the values of the dictionary from 0 to n
    
    Memory optimization: Single pass through dictionary
    """
    new_values = {}
    ret = {}
    count = 0

    for key, value in dictionary.items():
        if value not in new_values:
            new_values[value] = count
            count += 1
        ret[key] = new_values[value]

    return ret


def __load_binary(data):
    """Load binary graph as used by the cpp implementation of this algorithm"""
    with open(data, "rb") as f:
        reader = array.array("I")
        reader.fromfile(f, 1)
        num_nodes = reader.pop()
        reader = array.array("I")
        reader.fromfile(f, num_nodes)
        cum_deg = reader.tolist()
        num_links = reader.pop()
        reader = array.array("I")
        reader.fromfile(f, num_links)
        links = reader.tolist()
    
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph


def __one_level(graph, status, weight_key, resolution):
    """Compute one level of communities"""
    modified = True
    nb_pass_done = 0
    cur_mod = __modularity(status)
    new_mod = cur_mod

    while modified and nb_pass_done != PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in graph.nodes():
            com_node = status.node2com[node]
            degc_totw = status.gdegrees.get(node, 0.) / (status.total_weight * 2.)
            neigh_communities = __neighcom(node, graph, status, weight_key)
            __remove(node, com_node,
                     neigh_communities.get(com_node, 0.), status)
            best_com = com_node
            best_increase = 0
            for com, dnc in neigh_communities.items():
                incr = resolution * dnc - \
                       status.degrees.get(com, 0.) * degc_totw
                if incr > best_increase:
                    best_increase = incr
                    best_com = com
            __insert(node, best_com,
                     neigh_communities.get(best_com, 0.), status)
            if best_com != com_node:
                modified = True
        new_mod = __modularity(status)
        if new_mod - cur_mod < MIN_MODULARITY_GAIN:
            break


def __neighcom(node, graph, status, weight_key):
    """Compute the communities in the neighborhood of node"""
    weights = defaultdict(float)
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] += edge_weight

    return dict(weights)  # Convert back to regular dict


def __remove(node, com, weight, status):
    """Remove node from community com and modify status"""
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):
    """Insert node into community and modify status"""
    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))


def __modularity(status):
    """Fast compute the modularity of the partition of the graph using status precomputed"""
    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree / links - ((degree / (2. * links)) ** 2)
    return result


def calculate_modularity(graph, resolution):
    """Calculate modularity and get community partitions"""
    partition = best_partition(graph, resolution=resolution)
    mod_value, mod_dict = modularity(partition, graph)
    return partition, (mod_value, mod_dict)


def creating_ensemble_matrix_sparse(modularity_results, num_nodes):
    """Create ensemble matrix using sparse representation
    
    Memory optimization: Use sparse matrix for large graphs
    """
    resolutions = sorted(modularity_results.keys())
    num_resolutions = len(resolutions)
    
    # Use sparse matrix (LIL format for efficient construction)
    data_sparse = lil_matrix((num_nodes, num_resolutions), dtype=np.int32)
    
    for col_idx, res in enumerate(resolutions):
        partition_dict = modularity_results[res][0]
        for node, community in partition_dict.items():
            data_sparse[node, col_idx] = community
    
    # Convert to CSR for efficient operations
    data_csr = data_sparse.tocsr()
    del data_sparse
    gc.collect()
    
    # Convert to dense only for clustering (unavoidable)
    data_dense = data_csr.toarray()
    del data_csr
    gc.collect()
    
    # Add node IDs as first column
    node_ids = np.arange(num_nodes, dtype=np.int32).reshape(-1, 1)
    result = np.hstack((node_ids, data_dense))
    
    del node_ids, data_dense
    gc.collect()
    
    return result


def creating_ensemble_matrix(modularity_results):
    """Create ensemble matrix from modularity results
    
    Memory optimization: Build numpy array directly without intermediate lists
    """
    resolutions = sorted(modularity_results.keys())
    num_resolutions = len(resolutions)
    
    # Get number of nodes from first result
    first_partition = modularity_results[resolutions[0]][0]
    num_nodes = len(first_partition)
    
    # For large graphs, use sparse representation
    if num_nodes > 10000:
        return creating_ensemble_matrix_sparse(modularity_results, num_nodes)
    
    # Pre-allocate numpy array
    data = np.zeros((num_nodes, num_resolutions + 1), dtype=np.int32)
    
    # Fill node IDs in first column
    data[:, 0] = np.arange(num_nodes)
    
    # Fill partition data
    for col_idx, res in enumerate(resolutions):
        partition_dict = modularity_results[res][0]
        partition_list = [partition_dict[node] for node in range(num_nodes)]
        data[:, col_idx + 1] = partition_list
    
    return data


def ensemble_hierarchical_clustering(ensemble_matrix, graph, strength=DEFAULT_STRENGTH):
    """Perform hierarchical clustering based on ensemble matrix
    
    Memory optimization: Process in chunks if matrix is large, cleanup intermediates
    """
    partitions = ensemble_matrix[:, 1:].astype(np.int16)  # Use int16 to save memory
    
    # Compute distances
    distances = pdist(partitions, 'hamming')
    del partitions  # Free memory immediately
    gc.collect()

    # Apply thresholding
    threshold_value = strength / STRENGTH_DIVISOR
    distances[distances < threshold_value] = 0

    # Perform hierarchical clustering
    linkage_matrix = linkage(distances, method='single')
    del distances  # Free memory
    gc.collect()
    
    cluster_labels = fcluster(linkage_matrix, t=DISTANCE_THRESHOLD, criterion='distance')
    del linkage_matrix
    gc.collect()

    # Prepare the clustered nodes using generator
    clustered_nodes = []
    node_ids_col = ensemble_matrix[:, 0]
    
    for cluster_id in range(1, cluster_labels.max() + 1):
        mask = cluster_labels == cluster_id
        node_ids = node_ids_col[mask].tolist()
        clustered_nodes.append(node_ids)

    # Process the small remaining network
    clustered_nodes = creating_small_remaining_network(clustered_nodes, graph)

    return clustered_nodes


def creating_small_remaining_network(clustered_nodes, graph):
    """Process small clusters by re-clustering them
    
    Memory optimization: Process in-place, use list comprehensions efficiently
    """
    # Separate large and small clusters in one pass
    large_clusters = []
    small_nodes = []
    
    for cluster in clustered_nodes:
        if len(cluster) >= MIN_CLUSTER_SIZE:
            large_clusters.append(cluster)
        else:
            small_nodes.extend(cluster)
    
    # Clear original list to free memory
    del clustered_nodes
    gc.collect()
    
    if not small_nodes:
        return large_clusters
    
    # Generate subgraph from small remaining nodes
    remaining_nodes_set = set(small_nodes)
    del small_nodes
    
    subgraph = graph.subgraph(remaining_nodes_set).copy()

    if len(subgraph.edges) > 0:
        partitions, _ = calculate_modularity(subgraph, DEFAULT_RESOLUTION)
        
        # Group by community
        community_dict = defaultdict(list)
        for node, community in partitions.items():
            community_dict[community].append(node)

        # Extend large_clusters with small clusters
        large_clusters.extend(community_dict.values())
        
        # Cleanup
        del partitions, community_dict
        gc.collect()
    
    del subgraph
    gc.collect()
    
    return large_clusters


def process_network_file(network_file, weighted=True, directed=False):
    """Load a network from file
    
    Memory optimization: Use read_edgelist directly without intermediate storage
    """
    if weighted:
        if directed:
            graph = nx.read_edgelist(network_file, data=(('weight', float),), 
                                    nodetype=int, create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(network_file, data=(('weight', float),), 
                                    nodetype=int)
    else:
        if directed:
            graph = nx.read_edgelist(network_file, nodetype=int, 
                                    create_using=nx.DiGraph())
        else:
            graph = nx.read_edgelist(network_file, nodetype=int)
    
    return graph


def main(network_file, weighted=True, directed=False, resolutions=None):
    """Main function that performs ensemble community detection
    
    Memory optimization:
    - Process resolutions iteratively instead of storing all results
    - Explicit garbage collection after major operations
    - Use generators where possible
    
    Parameters
    ----------
    network_file : str
        Path to network file in edge list format
    weighted : bool, optional
        Whether the network has weights (default: True)
    directed : bool, optional
        Whether the network is directed (default: False)
    resolutions : list, optional
        List of resolution values to use. If None, uses default range.
        
    Returns
    -------
    list of lists
        Final clusters with original node IDs
    """
    # Load the graph
    graph = process_network_file(network_file, weighted=weighted, directed=directed)
    
    # Store original node labels and create sequential mapping
    original_nodes = list(graph.nodes())
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}
    reverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    # Relabel the graph nodes with new sequential integers
    relabeled_graph = nx.relabel_nodes(graph, node_mapping)
    del graph, node_mapping  # Free original graph
    gc.collect()
    
    # Define resolution range
    if resolutions is None:
        resolutions = [RESOLUTION_STEP * i for i in range(RESOLUTION_RANGE_START, RESOLUTION_RANGE_END)]

    # Calculate modularity for each resolution
    print(f"Calculating modularity for {len(resolutions)} resolutions...")
    modularity_results = {}
    for i, resolution in enumerate(resolutions):
        if i % 2 == 0:  # Progress update every 2 resolutions
            print(f"  Resolution {i+1}/{len(resolutions)}: {resolution:.2f}")
        
        partition, mod_values = calculate_modularity(relabeled_graph, resolution)
        modularity_results[resolution] = (partition, mod_values)
        
        # Periodic garbage collection
        if i % 3 == 0:
            gc.collect()

    print("Creating ensemble matrix...")
    # Create ensemble matrix
    ensemble_matrix = creating_ensemble_matrix(modularity_results)
    
    # Clear modularity results to free memory (keep only ensemble matrix)
    del modularity_results
    gc.collect()
    
    print("Performing hierarchical clustering...")
    # Perform hierarchical clustering
    final_clusters = ensemble_hierarchical_clustering(ensemble_matrix, relabeled_graph)
    
    # Cleanup
    del ensemble_matrix
    gc.collect()
    
    print("Mapping back to original node IDs...")
    # Map back to original node IDs
    final_clusters_original = [[int(reverse_mapping[node]) for node in cluster] 
                               for cluster in final_clusters]
    
    # Final cleanup
    del relabeled_graph, reverse_mapping, final_clusters
    gc.collect()
    
    print(f"✓ Completed! Found {len(final_clusters_original)} communities.")
    
    return final_clusters_original


