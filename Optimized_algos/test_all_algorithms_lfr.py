#!/usr/bin/env python3
"""
Test all non-GNN algorithms on LFR benchmark with 1000 nodes.
Uses ClusterNet wrappers (not raw igraph/cdlib).
Prints AMI (Adjusted Mutual Information) for each algorithm.
"""

import sys
import os
import time
import warnings
import traceback
import numpy as np
import networkx as nx
from sklearn.metrics import adjusted_mutual_info_score

warnings.filterwarnings('ignore')

# Add paths for ClusterNet
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'clusternet'))

# ============================================================
# LFR Benchmark Generation
# ============================================================

def generate_lfr_benchmark(n=1000, mu=0.3, seed=42):
    """
    Generate LFR benchmark graph with integer node IDs and weights.
    """
    print(f"\n{'='*60}")
    print(f"Generating LFR Benchmark (n={n}, μ={mu})")
    print(f"{'='*60}")
    
    np.random.seed(seed)
    
    # LFR parameters
    tau1 = 2.5
    tau2 = 1.5
    average_degree = 20
    max_degree = int(0.1 * n)
    min_community = max(20, int(0.02 * n))
    max_community = int(0.1 * n)
    
    try:
        G = nx.generators.community.LFR_benchmark_graph(
            n=n,
            tau1=tau1,
            tau2=tau2,
            mu=mu,
            average_degree=average_degree,
            max_degree=max_degree,
            min_community=min_community,
            max_community=max_community,
            seed=seed
        )
    except Exception as e:
        print(f"Failed with first params, trying alternative: {e}")
        G = nx.generators.community.LFR_benchmark_graph(
            n=n,
            tau1=3.0,
            tau2=2.0,
            mu=mu,
            average_degree=15,
            max_degree=50,
            min_community=20,
            max_community=100,
            seed=seed
        )
    
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Add weights (required by some algorithms)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    
    # Extract ground truth communities
    community_dict = nx.get_node_attributes(G, 'community')
    
    # Map nodes to community labels
    unique_comms = {}
    for node, comm_set in community_dict.items():
        comm_id = tuple(sorted(comm_set))
        if comm_id not in unique_comms:
            unique_comms[comm_id] = len(unique_comms)
    
    ground_truth_labels = np.zeros(n, dtype=np.int64)
    comm_members = {}
    
    for node, comm_set in community_dict.items():
        comm_id = tuple(sorted(comm_set))
        label = unique_comms[comm_id]
        ground_truth_labels[node] = label
        
        if label not in comm_members:
            comm_members[label] = []
        comm_members[label].append(node)
    
    ground_truth_communities = [comm_members[i] for i in sorted(comm_members.keys())]
    
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Ground truth communities: {len(ground_truth_communities)}")
    sizes = sorted([len(c) for c in ground_truth_communities], reverse=True)
    print(f"  Community sizes: {sizes[:10]}..." if len(sizes) > 10 else f"  Community sizes: {sizes}")
    
    return G, ground_truth_labels, ground_truth_communities


def communities_to_labels(communities, num_nodes, node_mapping=None):
    """Convert list of communities to label array, handling both int and str nodes."""
    labels = np.full(num_nodes, -1, dtype=int)
    
    for comm_id, comm in enumerate(communities):
        for node in comm:
            try:
                if node_mapping is not None:
                    idx = node_mapping.get(node, node_mapping.get(str(node), node_mapping.get(int(node) if isinstance(node, str) and node.isdigit() else node, -1)))
                else:
                    idx = int(node) if isinstance(node, str) else node
                
                if 0 <= idx < num_nodes:
                    labels[idx] = comm_id
            except (ValueError, TypeError):
                continue
    
    return labels


def compute_ami(detected_communities, ground_truth_labels, num_nodes, node_mapping=None):
    """Compute Adjusted Mutual Information."""
    detected_labels = communities_to_labels(detected_communities, num_nodes, node_mapping)
    
    # Filter out unclustered nodes
    mask = (detected_labels >= 0) & (ground_truth_labels >= 0)
    if mask.sum() == 0:
        return 0.0
    
    return adjusted_mutual_info_score(ground_truth_labels[mask], detected_labels[mask])


# ============================================================
# ClusterNet Algorithm Wrappers
# ============================================================

def get_clusternet_algorithms(G):
    """
    Get dictionary of all non-GNN algorithms from ClusterNet.
    Uses ClusterNet wrappers, NOT raw igraph/cdlib.
    """
    algorithms = {}
    num_nodes = G.number_of_nodes()
    estimated_k = max(2, int(np.sqrt(num_nodes / 2)))
    
    # =========================================
    # Classic Algorithms (ClusterNet wrappers)
    # =========================================
    
    # 1. Louvain
    try:
        from clusternet.algorithms.louvain_wrapper import LouvainWrapper
        def run_louvain(G):
            # resolution=1.0 gives best AMI on planted partition tests
            wrapper = LouvainWrapper(G, resolution=1.0)
            return wrapper.run()
        algorithms['Louvain'] = run_louvain
    except Exception as e:
        print(f"  [SKIP] Louvain: {e}")
    
    # 2. Leiden
    try:
        from clusternet.algorithms.leiden_wrapper import LeidenWrapper
        def run_leiden(G):
            # Use standard resolution=1.0 (now using proper leidenalg)
            wrapper = LeidenWrapper(G, resolution=1.0)
            return wrapper.run()
        algorithms['Leiden'] = run_leiden
    except Exception as e:
        print(f"  [SKIP] Leiden: {e}")
    
    # 3. Walktrap
    try:
        from clusternet.algorithms.walktrap_wrapper import WalktrapWrapper
        def run_walktrap(G):
            wrapper = WalktrapWrapper(G, steps=10, weighted=True, directed=False)
            return wrapper.run()
        algorithms['Walktrap'] = run_walktrap
    except Exception as e:
        print(f"  [SKIP] Walktrap: {e}")
    
    # 4. FastGreedy
    try:
        from clusternet.algorithms.fastgreedy_wrapper import FastGreedyWrapper
        def run_fastgreedy(G):
            # Use optimal modularity cut (n_clusters=None)
            # Forcing k hurts AMI because dendrogram doesn't align with ground truth
            wrapper = FastGreedyWrapper(G, weighted=True, directed=False, n_clusters=None)
            return wrapper.run()
        algorithms['FastGreedy'] = run_fastgreedy
    except Exception as e:
        print(f"  [SKIP] FastGreedy: {e}")
    
    # 5. SpinGlass
    try:
        from clusternet.algorithms.spinglass_wrapper import SpinGlassWrapper
        def run_spinglass(G):
            wrapper = SpinGlassWrapper(G, spins=25, weighted=True, directed=False)
            return wrapper.run()
        algorithms['SpinGlass'] = run_spinglass
    except Exception as e:
        print(f"  [SKIP] SpinGlass: {e}")
    
    # 6. Leading Eigenvector
    try:
        from clusternet.algorithms.leading_eigen_wrapper import LeadingEigenWrapper
        def run_leading_eigen(G):
            wrapper = LeadingEigenWrapper(G, weighted=True, directed=False)
            return wrapper.run()
        algorithms['LeadingEigen'] = run_leading_eigen
    except Exception as e:
        print(f"  [SKIP] LeadingEigen: {e}")
    
    # 7. Girvan-Newman (skip for large graphs - too slow)
    try:
        from clusternet.algorithms.girvan_newman_wrapper import GirvanNewmanWrapper
        def run_girvan_newman(G):
            if G.number_of_nodes() > 300:
                return None  # Too slow
            wrapper = GirvanNewmanWrapper(G, weighted=True, directed=False)
            return wrapper.run()
        algorithms['Girvan-Newman'] = run_girvan_newman
    except Exception as e:
        print(f"  [SKIP] Girvan-Newman: {e}")
    
    # 8. Label Propagation
    try:
        from clusternet.algorithms.label_propagation_wrapper import LabelPropagationWrapper
        def run_label_prop(G):
            wrapper = LabelPropagationWrapper(G, weighted=True, directed=False)
            return wrapper.run()
        algorithms['LabelPropagation'] = run_label_prop
    except Exception as e:
        print(f"  [SKIP] LabelPropagation: {e}")
    
    # 9. Spectral Clustering
    try:
        from clusternet.algorithms.spectral_wrapper import SpectralWrapper
        def run_spectral(G):
            wrapper = SpectralWrapper(G, n_clusters=estimated_k, weighted=True, directed=False)
            return wrapper.run()
        algorithms['Spectral'] = run_spectral
    except Exception as e:
        print(f"  [SKIP] Spectral: {e}")
    
    # =========================================
    # Custom Algorithms (ClusterNet)
    # =========================================
    
    # 10. SCORE
    try:
        from clusternet.algorithms.score_wrapper import SCOREWrapper
        def run_score(G):
            wrapper = SCOREWrapper(G, n_clusters=estimated_k)
            return wrapper.run()
        algorithms['SCORE'] = run_score
    except Exception as e:
        print(f"  [SKIP] SCORE: {e}")
    
    # 11. SVT/Tianle
    try:
        from clusternet.algorithms.svt_wrapper import SVTWrapper
        def run_svt(G):
            wrapper = SVTWrapper(G)
            return wrapper.run()
        algorithms['SVT'] = run_svt
    except Exception as e:
        print(f"  [SKIP] SVT: {e}")
    
    # 12. TeamCS
    try:
        from clusternet.algorithms.teamcs_wrapper import TeamCSWrapper
        def run_teamcs(G):
            wrapper = TeamCSWrapper(G)
            return wrapper.run()
        algorithms['TeamCS'] = run_teamcs
    except Exception as e:
        print(f"  [SKIP] TeamCS: {e}")
    
    # 13. SimNet
    try:
        from clusternet.algorithms.simnet_wrapper import SimNetWrapper
        def run_simnet(G):
            wrapper = SimNetWrapper(G)
            return wrapper.run()
        algorithms['SimNet'] = run_simnet
    except Exception as e:
        print(f"  [SKIP] SimNet: {e}")
    
    # 14. Tusk
    try:
        from clusternet.algorithms.tusk_wrapper import TuskWrapper
        def run_tusk(G):
            wrapper = TuskWrapper(G)
            return wrapper.run()
        algorithms['Tusk'] = run_tusk
    except Exception as e:
        print(f"  [SKIP] Tusk: {e}")
    
    # 15. BiGS2 (CPU fallback using SCORE)
    try:
        from SCORE import SCOREAlgorithm
        def run_bigs2(G):
            """BiGS2-style multi-stage SCORE on CPU."""
            n = G.number_of_nodes()
            # Initial clustering
            initial_k = max(2, int(np.sqrt(n / 10)))
            algo = SCOREAlgorithm(G, n_clusters=initial_k)
            communities = algo.run()
            
            # Refine large communities (BiGS2-style)
            max_size = 100
            refined = []
            for comm in communities:
                if len(comm) > max_size:
                    # Create subgraph and re-cluster
                    subgraph = G.subgraph([int(node) if isinstance(node, str) and node.isdigit() else node for node in comm])
                    sub_k = max(2, len(comm) // max_size + 1)
                    sub_algo = SCOREAlgorithm(subgraph, n_clusters=sub_k)
                    sub_comms = sub_algo.run()
                    refined.extend(sub_comms)
                else:
                    refined.append(comm)
            
            # Filter by min size
            return [c for c in refined if len(c) >= 3]
        algorithms['BiGS2'] = run_bigs2
    except Exception as e:
        print(f"  [SKIP] BiGS2: {e}")
    
    # 16. CSBIO-IITM2
    try:
        from clusternet.algorithms.csbio_iitm2_wrapper import CSBIOWrapper
        def run_csbio(G):
            wrapper = CSBIOWrapper(G)
            return wrapper.run()
        algorithms['CSBIO-IITM2'] = run_csbio
    except Exception as e:
        print(f"  [SKIP] CSBIO-IITM2: {e}")
    
    # =========================================
    # CDlib Wrappers (through ClusterNet)
    # =========================================
    
    # 17. EM (auto-tuned in wrapper)
    try:
        from clusternet.algorithms.cdlib_wrappers.statistical_wrappers import EMWrapper
        def run_em(G):
            wrapper = EMWrapper(G, auto_k=True)
            return wrapper.run()
        algorithms['EM'] = run_em
    except Exception as e:
        print(f"  [SKIP] EM: {e}")
    
    # 18. SBM
    try:
        from clusternet.algorithms.cdlib_wrappers.statistical_wrappers import SBMWrapper
        def run_sbm(G):
            wrapper = SBMWrapper(G)
            return wrapper.run()
        algorithms['SBM'] = run_sbm
    except Exception as e:
        print(f"  [SKIP] SBM: {e}")
    
    # 19. Nested SBM
    try:
        from clusternet.algorithms.cdlib_wrappers.statistical_wrappers import NestedSBMWrapper
        def run_nested_sbm(G):
            wrapper = NestedSBMWrapper(G)
            return wrapper.run()
        algorithms['NestedSBM'] = run_nested_sbm
    except Exception as e:
        print(f"  [SKIP] NestedSBM: {e}")
    
    # 20. CPM (tuned - lower resolution for fewer, larger communities)
    try:
        from clusternet.algorithms.cdlib_wrappers.physics_wrappers import CPMWrapper
        def run_cpm(G):
            # Lower resolution = fewer, larger communities
            wrapper = CPMWrapper(G, resolution_parameter=0.1)
            return wrapper.run()
        algorithms['CPM'] = run_cpm
    except Exception as e:
        print(f"  [SKIP] CPM: {e}")
    
    # 21. RB Potts
    try:
        from clusternet.algorithms.cdlib_wrappers.physics_wrappers import RBPotsWrapper
        def run_rb_pots(G):
            wrapper = RBPotsWrapper(G, resolution_parameter=1.0)
            return wrapper.run()
        algorithms['RB_Pots'] = run_rb_pots
    except Exception as e:
        print(f"  [SKIP] RB_Pots: {e}")
    
    # 22. RBER Potts
    try:
        from clusternet.algorithms.cdlib_wrappers.physics_wrappers import RBERPotsWrapper
        def run_rber_pots(G):
            wrapper = RBERPotsWrapper(G, resolution_parameter=1.0)
            return wrapper.run()
        algorithms['RBER_Pots'] = run_rber_pots
    except Exception as e:
        print(f"  [SKIP] RBER_Pots: {e}")
    
    # 23. DER (auto-tuned in wrapper)
    try:
        from clusternet.algorithms.cdlib_wrappers.diffusion_wrappers import DERWrapper
        def run_der(G):
            wrapper = DERWrapper(G, auto_tune=True)
            return wrapper.run()
        algorithms['DER'] = run_der
    except Exception as e:
        print(f"  [SKIP] DER: {e}")
    
    # 24. Async Fluid
    try:
        from clusternet.algorithms.cdlib_wrappers.diffusion_wrappers import AsyncFluidWrapper
        def run_async_fluid(G):
            wrapper = AsyncFluidWrapper(G, k=estimated_k)
            return wrapper.run()
        algorithms['AsyncFluid'] = run_async_fluid
    except Exception as e:
        print(f"  [SKIP] AsyncFluid: {e}")
    
    # 25. SCAN
    try:
        from clusternet.algorithms.cdlib_wrappers.structural_wrappers import SCANWrapper
        def run_scan(G):
            wrapper = SCANWrapper(G, epsilon=0.5, mu=3)
            return wrapper.run()
        algorithms['SCAN'] = run_scan
    except Exception as e:
        print(f"  [SKIP] SCAN: {e}")
    
    # 26. AGDL (auto-tuned in wrapper)
    try:
        from clusternet.algorithms.cdlib_wrappers.structural_wrappers import AGDLWrapper
        def run_agdl(G):
            wrapper = AGDLWrapper(G, auto_tune=True)
            return wrapper.run()
        algorithms['AGDL'] = run_agdl
    except Exception as e:
        print(f"  [SKIP] AGDL: {e}")
    
    # 27. GDMP2
    try:
        from clusternet.algorithms.cdlib_wrappers.structural_wrappers import GDMP2Wrapper
        def run_gdmp2(G):
            wrapper = GDMP2Wrapper(G, min_threshold=0.75)
            return wrapper.run()
        algorithms['GDMP2'] = run_gdmp2
    except Exception as e:
        print(f"  [SKIP] GDMP2: {e}")
    
    # 28. Angel (overlapping)
    try:
        from clusternet.algorithms.cdlib_wrappers.overlapping_wrappers import AngelWrapper
        def run_angel(G):
            wrapper = AngelWrapper(G, threshold=0.5, use_demon_fallback=True)
            return wrapper.run()
        algorithms['Angel'] = run_angel
    except Exception as e:
        print(f"  [SKIP] Angel: {e}")
    
    # 29. DEMON (overlapping - Angel's predecessor)
    try:
        from clusternet.algorithms.cdlib_wrappers.overlapping_wrappers import DEMONWrapper
        def run_demon(G):
            wrapper = DEMONWrapper(G, epsilon=0.25, min_com_size=3)
            return wrapper.run()
        algorithms['DEMON'] = run_demon
    except Exception as e:
        print(f"  [SKIP] DEMON: {e}")
    
    # 30. k-Clique (overlapping - clique percolation)
    try:
        from clusternet.algorithms.cdlib_wrappers.overlapping_wrappers import KCliqueWrapper
        def run_kclique(G):
            wrapper = KCliqueWrapper(G, k=3)
            return wrapper.run()
        algorithms['k-Clique'] = run_kclique
    except Exception as e:
        print(f"  [SKIP] k-Clique: {e}")
    
    # 31. Surprise Communities (quality-function based, disjoint)
    try:
        from clusternet.algorithms.cdlib_wrappers.physics_wrappers import SurpriseCommunitiesWrapper
        def run_surprise(G):
            wrapper = SurpriseCommunitiesWrapper(G)
            return wrapper.run()
        algorithms['Surprise'] = run_surprise
    except Exception as e:
        print(f"  [SKIP] Surprise: {e}")
    
    return algorithms


# ============================================================
# Main Test Runner
# ============================================================

def run_all_tests(G, ground_truth_labels, ground_truth_communities):
    """Run all algorithms and report results."""
    
    num_nodes = G.number_of_nodes()
    num_gt_communities = len(ground_truth_communities)
    
    # Create node mapping (node -> index)
    node_list = list(G.nodes())
    node_mapping = {node: idx for idx, node in enumerate(node_list)}
    
    print(f"\n{'='*60}")
    print("Loading ClusterNet Algorithms...")
    print(f"{'='*60}")
    
    algorithms = get_clusternet_algorithms(G)
    
    print(f"\n{'='*60}")
    print(f"Running {len(algorithms)} Algorithms")
    print(f"{'='*60}")
    
    results = []
    
    for name, run_func in algorithms.items():
        print(f"\n[{name}]")
        
        try:
            start_time = time.time()
            communities = run_func(G)
            elapsed = time.time() - start_time
            
            if communities is None or len(communities) == 0:
                print(f"  SKIPPED (no communities returned)")
                results.append((name, None, None, None, "SKIPPED"))
                continue
            
            # Compute AMI
            ami = compute_ami(communities, ground_truth_labels, num_nodes, node_mapping)
            
            # Compute modularity
            try:
                graph_nodes = set(G.nodes())
                node_to_comm = {}  # For converting overlapping to disjoint
                
                for comm_idx, comm in enumerate(communities):
                    for node in comm:
                        matched_node = None
                        # Try direct match first
                        if node in graph_nodes:
                            matched_node = node
                        # Try string-to-int conversion
                        elif isinstance(node, str):
                            try:
                                int_node = int(node)
                                if int_node in graph_nodes:
                                    matched_node = int_node
                            except ValueError:
                                pass
                        # Try int-to-string conversion
                        elif isinstance(node, (int, np.integer)):
                            if node in graph_nodes:
                                matched_node = node
                            else:
                                str_node = str(node)
                                if str_node in graph_nodes:
                                    matched_node = str_node
                        
                        if matched_node is not None:
                            # For overlapping: assign to first (largest) community seen
                            if matched_node not in node_to_comm:
                                node_to_comm[matched_node] = comm_idx
                
                # Build disjoint communities
                comm_dict = {}
                for node, comm_idx in node_to_comm.items():
                    if comm_idx not in comm_dict:
                        comm_dict[comm_idx] = set()
                    comm_dict[comm_idx].add(node)
                
                mod_communities = list(comm_dict.values())
                
                # Add uncovered nodes as singletons (for algorithms like SCAN that have outliers)
                covered_nodes = set(node_to_comm.keys())
                uncovered = graph_nodes - covered_nodes
                for node in uncovered:
                    mod_communities.append({node})
                
                if mod_communities:
                    modularity = nx.community.modularity(G, mod_communities)
                else:
                    modularity = 0.0
            except Exception as me:
                # print(f"  Modularity error: {me}")
                modularity = 0.0
            
            n_comms = len(communities)
            
            print(f"  Communities: {n_comms} (ground truth: {num_gt_communities})")
            print(f"  AMI: {ami:.4f}")
            print(f"  Modularity: {modularity:.4f}")
            print(f"  Time: {elapsed:.2f}s")
            
            results.append((name, ami, modularity, n_comms, elapsed))
            
        except Exception as e:
            print(f"  ERROR: {str(e)[:60]}")
            results.append((name, None, None, None, f"ERROR: {str(e)[:30]}"))
    
    return results


def print_summary(results, num_gt_communities):
    """Print summary table of results."""
    
    print(f"\n{'='*80}")
    print("SUMMARY: ClusterNet Algorithms on LFR Benchmark (n=1000, μ=0.3)")
    print(f"{'='*80}")
    print(f"Ground truth communities: {num_gt_communities}")
    print()
    
    # Header
    print(f"{'Algorithm':<20} {'AMI':>10} {'Modularity':>12} {'#Comms':>8} {'Time (s)':>10} {'Status':>12}")
    print("-" * 80)
    
    # Sort by AMI (descending)
    successful = [(n, a, m, c, t) for n, a, m, c, t in results if isinstance(t, float)]
    failed = [(n, a, m, c, t) for n, a, m, c, t in results if not isinstance(t, float)]
    
    successful_sorted = sorted(successful, key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
    
    for name, ami, mod, n_comms, elapsed in successful_sorted:
        ami_str = f"{ami:.4f}" if ami is not None else "N/A"
        mod_str = f"{mod:.4f}" if mod is not None else "N/A"
        comm_str = str(n_comms) if n_comms is not None else "N/A"
        time_str = f"{elapsed:.2f}" if elapsed is not None else "N/A"
        print(f"{name:<20} {ami_str:>10} {mod_str:>12} {comm_str:>8} {time_str:>10} {'✓':>12}")
    
    if failed:
        print("-" * 80)
        for name, ami, mod, n_comms, status in failed:
            status_short = str(status)[:25] if len(str(status)) > 25 else str(status)
            print(f"{name:<20} {'---':>10} {'---':>12} {'---':>8} {'---':>10} {status_short:>12}")
    
    print("-" * 80)
    
    # Statistics
    ami_values = [r[1] for r in successful if r[1] is not None]
    if ami_values:
        print(f"\n📊 AMI Statistics:")
        print(f"  🥇 Best:   {max(ami_values):.4f}")
        print(f"  🥉 Worst:  {min(ami_values):.4f}")
        print(f"  📈 Mean:   {np.mean(ami_values):.4f}")
        print(f"  📊 Median: {np.median(ami_values):.4f}")
        print(f"\n  ✅ Successful: {len(successful)}/{len(results)}")


def main():
    """Main entry point."""
    
    print("=" * 60)
    print("🔬 ClusterNet: Testing All Non-GNN Algorithms")
    print("📊 Benchmark: LFR with 1000 nodes (μ=0.3)")
    print("📦 Using ClusterNet wrappers (not raw igraph/cdlib)")
    print("=" * 60)
    
    # Generate LFR benchmark
    G, ground_truth_labels, ground_truth_communities = generate_lfr_benchmark(
        n=1000,
        mu=0.3,  # Moderate mixing (0.1=easy, 0.5=hard)
        seed=42
    )
    
    # Run all algorithms
    results = run_all_tests(G, ground_truth_labels, ground_truth_communities)
    
    # Print summary
    print_summary(results, len(ground_truth_communities))
    
    print("\n" + "=" * 60)
    print("✅ Testing Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
