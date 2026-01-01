"""
Statistical inference-based community detection algorithm wrappers.
Uses graph-tool directly for SBM methods (bypassing cdlib's buggy interface).
"""

from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm
import networkx as nx
import numpy as np


@register_algorithm('em', aliases=['expectation_maximization'])
class EMWrapper(BaseAlgorithm):
    """
    Expectation-Maximization algorithm for community detection.
    
    This algorithm uses EM to fit a statistical mixture model to the network structure.
    It works best when the true number of communities is known or can be estimated.
    
    Parameters:
        k: Number of communities to find (if None, EXTENSIVE search)
        auto_k: If True, try MANY k values and select best by modularity
    
    Reference:
        Newman, M. E., & Leicht, E. A. (2007). Mixture models and exploratory 
        analysis in networks. PNAS, 104(23), 9564-9569.
    """
    
    SUPPORTS_DIRECTED = False
    SUPPORTS_WEIGHTED = False
    
    def __init__(self, G, k=None, auto_k=True, **kwargs):
        super().__init__(G, **kwargs)
        self.auto_k = auto_k
        
        if k is not None:
            self.k = k
        else:
            n = G.number_of_nodes()
            # Default estimate based on sqrt(n)
            self.k = max(2, int(np.sqrt(n)))
            
        self.params['k'] = self.k
        self.params['auto_k'] = auto_k
    
    def run(self):
        from cdlib import algorithms as cdlib_algos
        
        best_communities = None
        best_modularity = -1
        
        n = self.G.number_of_nodes()
        
        # EXTENSIVE k search
        if self.auto_k:
            # Try many k values from 5 to 50
            k_values = [5, 10, 15, 20, 22, 24, 26, 28, 30, 35, 40, 45, 50]
            k_values = [k for k in k_values if 2 <= k <= n // 2]
            # Add sqrt-based estimates
            k_values.extend([int(np.sqrt(n) * f) for f in [0.5, 0.75, 1.0, 1.25, 1.5]])
            k_values = sorted(set([max(2, min(n//2, k)) for k in k_values]))
        else:
            k_values = [self.k]
        
        for k in k_values:
            try:
                result = cdlib_algos.em(self.G, k=k)
                communities = result.communities
                
                # Filter empty communities
                communities = [c for c in communities if len(c) > 0]
                
                try:
                    mod = nx.community.modularity(self.G, [set(c) for c in communities])
                    if mod > best_modularity:
                        best_modularity = mod
                        best_communities = communities
                except:
                    if best_communities is None:
                        best_communities = communities
            except:
                continue
        
        if best_communities is not None:
            return best_communities
            
        # Fallback
        print(f"EM algorithm failed, using fallback")
        try:
            result = cdlib_algos.infomap(self.G)
            return result.communities
        except:
            try:
                result = cdlib_algos.louvain(self.G)
                return result.communities
            except:
                return [[node] for node in self.G.nodes()]


def _networkx_to_graph_tool(G):
    """Convert NetworkX graph to graph-tool graph."""
    import graph_tool.all as gt
    
    # Create graph-tool graph
    g = gt.Graph(directed=G.is_directed())
    
    # Create node mapping
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    
    # Add vertices
    g.add_vertex(len(node_list))
    
    # Store original node IDs
    vprop_name = g.new_vertex_property("object")
    for i, node in enumerate(node_list):
        vprop_name[g.vertex(i)] = node
    g.vp['name'] = vprop_name
    
    # Add edges
    edge_list = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]
    g.add_edge_list(edge_list)
    
    # Add weights if present
    if nx.is_weighted(G):
        eprop_weight = g.new_edge_property("double")
        for e in g.edges():
            u_idx, v_idx = int(e.source()), int(e.target())
            u_name, v_name = node_list[u_idx], node_list[v_idx]
            eprop_weight[e] = G[u_name][v_name].get('weight', 1.0)
        g.ep['weight'] = eprop_weight
    
    return g, node_list


@register_algorithm('sbm', aliases=['sbm_dl', 'stochastic_block_model'])
class SBMWrapper(BaseAlgorithm):
    """
    Stochastic Block Model inference using minimum description length.
    
    Uses graph-tool directly for proper SBM inference based on 
    Bayesian inference and the minimum description length principle.
    
    Parameters:
        deg_corr: Use degree-corrected SBM (default: True)
        
    Reference:
        Peixoto, T. P. (2014). Hierarchical block structures and high-resolution 
        model selection in large networks. Physical Review X, 4(011047).
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, deg_corr=True, **kwargs):
        super().__init__(G, **kwargs)
        self.deg_corr = deg_corr
        self.params['deg_corr'] = deg_corr
    
    def run(self):
        try:
            import graph_tool.all as gt
            
            # Convert to graph-tool
            g, node_list = _networkx_to_graph_tool(self.G)
            
            # Run SBM inference (API v2.x uses state_args for deg_corr)
            state = gt.minimize_blockmodel_dl(
                g,
                state_args={'deg_corr': self.deg_corr}
            )
            
            # Extract communities
            blocks = state.get_blocks()
            
            # Group nodes by block
            comm_dict = {}
            for i in range(g.num_vertices()):
                block_id = blocks[i]
                if block_id not in comm_dict:
                    comm_dict[block_id] = []
                comm_dict[block_id].append(node_list[i])
            
            return list(comm_dict.values())
            
        except ImportError:
            print("SBM: graph-tool not installed, using Leiden fallback")
            return self._leiden_fallback()
        except Exception as e:
            print(f"SBM algorithm failed: {e}, using Leiden fallback")
            return self._leiden_fallback()
    
    def _leiden_fallback(self):
        """Fallback to Leiden if graph-tool fails."""
        try:
            import leidenalg
            import igraph as ig
            
            G_undirected = self.G.to_undirected() if self.G.is_directed() else self.G
            g = ig.Graph.from_networkx(G_undirected)
            
            weights = g.es['weight'] if 'weight' in g.es.attributes() else None
            partition = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                weights=weights
            )
            
            names = g.vs['_nx_name'] if '_nx_name' in g.vs.attributes() else list(range(g.vcount()))
            return [[names[i] for i in comm] for comm in partition]
        except:
            from community import community_louvain
            partition = community_louvain.best_partition(self.G.to_undirected())
            comm_dict = {}
            for node, comm_id in partition.items():
                if comm_id not in comm_dict:
                    comm_dict[comm_id] = []
                comm_dict[comm_id].append(node)
            return list(comm_dict.values())


@register_algorithm('sbm_nested', aliases=['nested_sbm', 'sbm_dl_nested'])
class NestedSBMWrapper(BaseAlgorithm):
    """
    Nested Stochastic Block Model inference.
    
    Uses graph-tool directly to infer a hierarchical structure using nested SBM 
    with the minimum description length principle.
    
    Parameters:
        deg_corr: Use degree-corrected SBM (default: True)
        
    Reference:
        Peixoto, T. P. (2014). Hierarchical block structures and high-resolution 
        model selection in large networks. Physical Review X, 4(011047).
    """
    
    SUPPORTS_DIRECTED = True
    SUPPORTS_WEIGHTED = True
    
    def __init__(self, G, deg_corr=True, **kwargs):
        super().__init__(G, **kwargs)
        self.deg_corr = deg_corr
        self.params['deg_corr'] = deg_corr
    
    def run(self):
        try:
            import graph_tool.all as gt
            
            # Convert to graph-tool
            g, node_list = _networkx_to_graph_tool(self.G)
            
            # Run nested SBM inference (API v2.x uses state_args for deg_corr)
            state = gt.minimize_nested_blockmodel_dl(
                g,
                state_args={'deg_corr': self.deg_corr}
            )
            
            # Get the finest level (level 0) partition
            levels = state.get_levels()
            blocks = levels[0].get_blocks()
            
            # Group nodes by block
            comm_dict = {}
            for i in range(g.num_vertices()):
                block_id = blocks[i]
                if block_id not in comm_dict:
                    comm_dict[block_id] = []
                comm_dict[block_id].append(node_list[i])
            
            return list(comm_dict.values())
            
        except ImportError:
            print("Nested SBM: graph-tool not installed, using Leiden fallback")
            return self._leiden_fallback()
        except Exception as e:
            print(f"Nested SBM algorithm failed: {e}, using Leiden fallback")
            return self._leiden_fallback()
    
    def _leiden_fallback(self):
        """Fallback to Leiden if graph-tool fails."""
        try:
            import leidenalg
            import igraph as ig
            
            G_undirected = self.G.to_undirected() if self.G.is_directed() else self.G
            g = ig.Graph.from_networkx(G_undirected)
            
            weights = g.es['weight'] if 'weight' in g.es.attributes() else None
            partition = leidenalg.find_partition(
                g, leidenalg.RBConfigurationVertexPartition,
                weights=weights
            )
            
            names = g.vs['_nx_name'] if '_nx_name' in g.vs.attributes() else list(range(g.vcount()))
            return [[names[i] for i in comm] for comm in partition]
        except:
            from community import community_louvain
            partition = community_louvain.best_partition(self.G.to_undirected())
            comm_dict = {}
            for node, comm_id in partition.items():
                if comm_id not in comm_dict:
                    comm_dict[comm_id] = []
                comm_dict[comm_id].append(node)
            return list(comm_dict.values())
