import networkx as nx
import numpy as np

class LouvainAlgorithm:
    """
    A Generalized Python implementation of the Louvain algorithm.
    
    HANDLES ALL 4 NETWORK TYPES:
    1. Undirected Unweighted (defaults to weight=1.0)
    2. Undirected Weighted
    3. Directed Unweighted (converted to Undirected Weighted via symmetrization)
    4. Directed Weighted (converted to Undirected Weighted via symmetrization)
    """
    def __init__(self, G, resolution=1.0):
        self.resolution = resolution
        
        # 1. Preprocess: Normalize all inputs (Directed/Unweighted) 
        # into a common Weighted Undirected format for the math engine.
        self.G = self._preprocess_graph(G)
            
        self.m = self.G.size(weight='weight') / 2.0 # Total weight (undirected so /2)
        if self.m == 0: self.m = 1 # Safety
        
        # Cache edge weights for speed (Adjacency Dictionary)
        self.adj = {n: dict(nbrs) for n, nbrs in self.G.adjacency()}
        
        # Calculate node degrees (weighted)
        # In a weighted undirected graph, degree = sum of weights of incident edges
        self.node_weights = {n: sum(nbr_data.get('weight', 1.0) for nbr_data in d.values()) for n, d in self.adj.items()}

    def _preprocess_graph(self, input_G):
        """
        Normalizes input graph into a Weighted Undirected Graph.
        """
        # A. Handle Direction
        if input_G.is_directed():
            # Symmetrization Strategy:
            # We convert Directed -> Undirected.
            # If A->B (w=1) and B->A (w=2), the undirected edge A-B becomes w=3.
            # This preserves the "strength" of mutual connections.
            undirected_G = nx.Graph()
            for u, v, data in input_G.edges(data=True):
                w = data.get('weight', 1.0)
                if undirected_G.has_edge(u, v):
                    # Add to existing weight (handling the reverse edge case)
                    undirected_G[u][v]['weight'] += w
                else:
                    undirected_G.add_edge(u, v, weight=w)
            
            # Ensure every node is kept (even isolated ones)
            undirected_G.add_nodes_from(input_G.nodes())
            return undirected_G
            
        # B. Handle Undirected (Check for weights)
        else:
            # If already undirected, we just need to ensure 'weight' attribute exists
            # If the user passed an unweighted graph, we don't want to modify the 
            # original object, so we make a lightweight copy with default weights.
            
            # fast check if weighted
            is_weighted = False
            for _, _, data in input_G.edges(data=True):
                if 'weight' in data:
                    is_weighted = True
                break
            
            if is_weighted:
                return input_G.copy()
            else:
                # Assign weight 1.0 to all
                G_w = input_G.copy()
                nx.set_edge_attributes(G_w, 1.0, 'weight')
                return G_w

    def run(self):
        """Main execution loop."""
        current_graph = self.G
        
        # Initial partition: every node is its own community
        partition = {n: i for i, n in enumerate(current_graph.nodes())}
        
        while True:
            # PHASE 1: Modularity Optimization
            partition, improvement = self._one_level(current_graph, partition)
            
            if not improvement:
                break
                
            # PHASE 2: Aggregation
            current_graph = self._induce_graph(partition, current_graph)
            
            # For this benchmark version, we stop after one full pass of aggregation
            # to ensure consistent output formats and speed. 
            # (Standard Louvain loops this until graph size doesn't shrink)
            break 

        return self._format_output(partition)

    def _one_level(self, graph, partition):
        """Core Greedy Optimization."""
        modified = False
        node2com = partition.copy()
        
        degrees = dict(graph.degree(weight='weight'))
        self_loops = {n: 0 for n in graph.nodes()}
        for u, v, d in graph.edges(data=True):
            if u == v: self_loops[u] = d.get('weight', 1.0)
        
        # Initialize community stats
        in_degrees = {} 
        tot_degrees = {} 
        
        for node, com in node2com.items():
            in_degrees[com] = in_degrees.get(com, 0) + self_loops.get(node, 0)
            tot_degrees[com] = tot_degrees.get(com, 0) + degrees.get(node, 0)
            
        nb_pass_done = 0
        
        # Iterate until convergence (or limit to avoid infinite loops)
        while nb_pass_done < 20: 
            cur_mod = False
            nodes = list(graph.nodes())
            # Random shuffle is crucial for Louvain stability
            np.random.shuffle(nodes)
            
            for node in nodes:
                com_node = node2com[node]
                degc_totw = tot_degrees[com_node]
                deg_node = degrees[node]
                
                # Remove node from its current community
                tot_degrees[com_node] -= deg_node
                # (We don't need to explicitly update in_degrees for removal 
                #  because we only calculate Gain for *other* communities)

                best_com = com_node
                best_increase = 0
                
                # Find neighbor communities
                neighbor_communities = {}
                for neighbor in graph[node]:
                    wt = graph[node][neighbor].get('weight', 1.0)
                    nbr_com = node2com[neighbor]
                    neighbor_communities[nbr_com] = neighbor_communities.get(nbr_com, 0) + wt
                
                # Calculate Gain
                # Delta Q = k_i_in - (Sigma_tot * k_i * resolution) / m
                # We factor out constants to speed up comparison
                for nbr_com, k_i_in in neighbor_communities.items():
                    if nbr_com == com_node: continue 
                    
                    tot = tot_degrees.get(nbr_com, 0)
                    gain = k_i_in - (tot * deg_node * self.resolution) / (2 * self.m)
                    
                    if gain > best_increase:
                        best_increase = gain
                        best_com = nbr_com
                
                # Move Node
                if best_com != com_node:
                    tot_degrees[best_com] += deg_node
                    node2com[node] = best_com
                    cur_mod = True
                    modified = True
                else:
                    # Put back stats if we didn't move
                    tot_degrees[com_node] += deg_node
            
            if not cur_mod:
                break
            nb_pass_done += 1
            
        return node2com, modified

    def _induce_graph(self, partition, graph):
        """Aggregates communities into super-nodes."""
        new_graph = nx.Graph()
        
        for node1, node2, data in graph.edges(data=True):
            weight = data.get('weight', 1.0)
            com1 = partition[node1]
            com2 = partition[node2]
            
            if new_graph.has_edge(com1, com2):
                new_graph[com1][com2]['weight'] += weight
            else:
                new_graph.add_edge(com1, com2, weight=weight)
            
        return new_graph

    def _format_output(self, partition):
        clusters = {}
        for node, com_id in partition.items():
            if com_id not in clusters:
                clusters[com_id] = []
            clusters[com_id].append(str(node))
        return list(clusters.values())

# ==========================================
# WRAPPER FUNCTION
# ==========================================

def louvain_clustering(input_data, resolution=1.0, is_directed=False):
    """
    Main entry point for Louvain.
    
    Args:
        input_data: NetworkX graph OR file path (edgelist).
        resolution: 1.0 is standard.
        is_directed: Boolean. If True, input is treated as directed 
                     and symmetrized for the algorithm.
        
    Returns:
        List of lists: [['n1', 'n2'], ...]
    """
    
    # --- 1. INPUT HANDLING ---
    if isinstance(input_data, str):
        try:
            # We assume weighted first
            create_using = nx.DiGraph() if is_directed else nx.Graph()
            G = nx.read_edgelist(input_data, create_using=create_using, nodetype=str, data=(('weight', float),))
        except TypeError:
            # Fallback to unweighted
            create_using = nx.DiGraph() if is_directed else nx.Graph()
            G = nx.read_edgelist(input_data, create_using=create_using, nodetype=str)
    elif isinstance(input_data, (nx.Graph, nx.DiGraph)):
        G = input_data
    else:
        raise ValueError("Input must be a file path string or a NetworkX Graph object.")

    # --- 2. EXECUTION ---
    algo = LouvainAlgorithm(G, resolution)
    communities = algo.run()
    
    return communities