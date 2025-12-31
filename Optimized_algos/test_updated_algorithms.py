#!/usr/bin/env python3
"""
Test script for updated igraph-based community detection algorithms.

This script tests that all the updated algorithms:
1. Can be imported correctly
2. Work with the ClusterNet wrapper interface
3. Work with the standalone function interface (matching main.py)
4. Produce valid community structures

Algorithms tested:
- Walktrap
- Fast Greedy
- Spin Glass
- Label Propagation
- Leading Eigenvector
- Girvan-Newman
- Spectral Clustering
"""

import sys
import os
import time

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clusternet'))

import networkx as nx
import traceback


def create_test_graph():
    """Create a test graph with known community structure."""
    # Karate club graph has 2 known communities
    G = nx.karate_club_graph()
    # Add weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G


def create_small_test_graph():
    """Create a smaller test graph for faster testing."""
    G = nx.Graph()
    # Community 1: nodes 0-4
    for i in range(5):
        for j in range(i+1, 5):
            G.add_edge(i, j, weight=1.0)
    # Community 2: nodes 5-9
    for i in range(5, 10):
        for j in range(i+1, 10):
            G.add_edge(i, j, weight=1.0)
    # Bridge between communities
    G.add_edge(4, 5, weight=0.5)
    return G


def validate_communities(communities, G):
    """Validate that communities are properly formed."""
    if not communities:
        return False, "Empty communities list"
    
    if not isinstance(communities, list):
        return False, f"Expected list, got {type(communities)}"
    
    # Check each community is a list
    for i, comm in enumerate(communities):
        if not isinstance(comm, list):
            return False, f"Community {i} is not a list: {type(comm)}"
    
    # Flatten and check nodes
    all_nodes = []
    for comm in communities:
        all_nodes.extend(comm)
    
    # Check for valid node types
    graph_nodes = set(G.nodes())
    for node in all_nodes:
        if node not in graph_nodes:
            # Allow int conversion
            if int(node) not in graph_nodes:
                return False, f"Invalid node: {node}"
    
    return True, "Valid"


def test_wrapper_interface(algorithm_class, G, **kwargs):
    """Test the ClusterNet wrapper interface."""
    try:
        algo = algorithm_class(G, **kwargs)
        communities = algo.run()
        valid, msg = validate_communities(communities, G)
        return communities, valid, msg
    except Exception as e:
        return None, False, f"Error: {str(e)}\n{traceback.format_exc()}"


def test_standalone_function(func, G, **kwargs):
    """Test the standalone function interface."""
    try:
        result = func(G=G, **kwargs)
        # Handle tuple return (communities, runtime)
        if isinstance(result, tuple):
            communities, runtime = result
        else:
            communities = result
            runtime = None
        valid, msg = validate_communities(communities, G)
        return communities, valid, msg, runtime
    except Exception as e:
        return None, False, f"Error: {str(e)}\n{traceback.format_exc()}", None


def print_result(name, success, message, num_communities=None, runtime=None):
    """Print test result."""
    status = "✓ PASS" if success else "✗ FAIL"
    info = f" ({num_communities} communities" if num_communities else ""
    if runtime is not None:
        info += f", {runtime:.4f}s"
    info += ")" if info else ""
    print(f"  {status}: {name}{info}")
    if not success:
        print(f"         {message}")


def main():
    print("=" * 60)
    print("Testing Updated Community Detection Algorithms")
    print("=" * 60)
    
    # Create test graphs
    G_karate = create_test_graph()
    G_small = create_small_test_graph()
    
    print(f"\nTest Graph 1 (Karate Club): {G_karate.number_of_nodes()} nodes, {G_karate.number_of_edges()} edges")
    print(f"Test Graph 2 (Small): {G_small.number_of_nodes()} nodes, {G_small.number_of_edges()} edges")
    
    all_passed = True
    
    # =========================================
    # Test 1: Walktrap
    # =========================================
    print("\n" + "-" * 40)
    print("1. WALKTRAP")
    print("-" * 40)
    
    try:
        from clusternet.algorithms.walktrap_wrapper import WalktrapWrapper
        from walktrap import walktrap, WalktrapAlgorithm
        
        # Test wrapper
        comms, valid, msg = test_wrapper_interface(WalktrapWrapper, G_karate, steps=10)
        print_result("Wrapper Interface", valid, msg, len(comms) if comms else None)
        all_passed &= valid
        
        # Test standalone function
        comms, valid, msg, runtime = test_standalone_function(walktrap, G_karate, steps=10)
        print_result("Standalone Function", valid, msg, len(comms) if comms else None, runtime)
        all_passed &= valid
        
        # Test algorithm class directly
        algo = WalktrapAlgorithm(G_karate, steps=10)
        comms, runtime = algo.run()
        valid, msg = validate_communities(comms, G_karate)
        print_result("Algorithm Class", valid, msg, len(comms), runtime)
        all_passed &= valid
        
    except Exception as e:
        print(f"  ✗ FAIL: Import/Setup Error - {str(e)}")
        all_passed = False
    
    # =========================================
    # Test 2: Fast Greedy
    # =========================================
    print("\n" + "-" * 40)
    print("2. FAST GREEDY")
    print("-" * 40)
    
    try:
        from clusternet.algorithms.fastgreedy_wrapper import FastGreedyWrapper
        from fastgreedy import fast_greedy, FastGreedyAlgorithm
        
        # Test wrapper
        comms, valid, msg = test_wrapper_interface(FastGreedyWrapper, G_karate)
        print_result("Wrapper Interface", valid, msg, len(comms) if comms else None)
        all_passed &= valid
        
        # Test standalone function
        comms, valid, msg, runtime = test_standalone_function(fast_greedy, G_karate)
        print_result("Standalone Function", valid, msg, len(comms) if comms else None, runtime)
        all_passed &= valid
        
        # Test algorithm class directly
        algo = FastGreedyAlgorithm(G_karate)
        comms, runtime = algo.run()
        valid, msg = validate_communities(comms, G_karate)
        print_result("Algorithm Class", valid, msg, len(comms), runtime)
        all_passed &= valid
        
    except Exception as e:
        print(f"  ✗ FAIL: Import/Setup Error - {str(e)}")
        traceback.print_exc()
        all_passed = False
    
    # =========================================
    # Test 3: Spin Glass
    # =========================================
    print("\n" + "-" * 40)
    print("3. SPIN GLASS")
    print("-" * 40)
    
    try:
        from clusternet.algorithms.spinglass_wrapper import SpinGlassWrapper
        from spinglass import spin_glass, SpinGlassAlgorithm
        
        # Test wrapper (use smaller graph, spin glass can be slow)
        comms, valid, msg = test_wrapper_interface(SpinGlassWrapper, G_small, spins=5)
        print_result("Wrapper Interface", valid, msg, len(comms) if comms else None)
        all_passed &= valid
        
        # Test standalone function
        comms, valid, msg, runtime = test_standalone_function(spin_glass, G_small, spins=5)
        print_result("Standalone Function", valid, msg, len(comms) if comms else None, runtime)
        all_passed &= valid
        
        # Test algorithm class directly
        algo = SpinGlassAlgorithm(G_small, spins=5)
        comms, runtime = algo.run()
        valid, msg = validate_communities(comms, G_small)
        print_result("Algorithm Class", valid, msg, len(comms), runtime)
        all_passed &= valid
        
    except Exception as e:
        print(f"  ✗ FAIL: Import/Setup Error - {str(e)}")
        traceback.print_exc()
        all_passed = False
    
    # =========================================
    # Test 4: Label Propagation
    # =========================================
    print("\n" + "-" * 40)
    print("4. LABEL PROPAGATION")
    print("-" * 40)
    
    try:
        from clusternet.algorithms.label_propagation_wrapper import LabelPropagationWrapper
        from label_propagation import label_propogation, LabelPropagationAlgorithm
        
        # Test wrapper
        comms, valid, msg = test_wrapper_interface(LabelPropagationWrapper, G_karate)
        print_result("Wrapper Interface", valid, msg, len(comms) if comms else None)
        all_passed &= valid
        
        # Test standalone function
        comms, valid, msg, runtime = test_standalone_function(label_propogation, G_karate)
        print_result("Standalone Function", valid, msg, len(comms) if comms else None, runtime)
        all_passed &= valid
        
        # Test algorithm class directly
        algo = LabelPropagationAlgorithm(G_karate)
        comms, runtime = algo.run()
        valid, msg = validate_communities(comms, G_karate)
        print_result("Algorithm Class", valid, msg, len(comms), runtime)
        all_passed &= valid
        
    except Exception as e:
        print(f"  ✗ FAIL: Import/Setup Error - {str(e)}")
        traceback.print_exc()
        all_passed = False
    
    # =========================================
    # Test 5: Leading Eigenvector
    # =========================================
    print("\n" + "-" * 40)
    print("5. LEADING EIGENVECTOR")
    print("-" * 40)
    
    try:
        from clusternet.algorithms.leading_eigen_wrapper import LeadingEigenWrapper
        from leading_eigen import leading_eigen_vector, LeadingEigenAlgorithm
        
        # Test wrapper
        comms, valid, msg = test_wrapper_interface(LeadingEigenWrapper, G_karate)
        print_result("Wrapper Interface", valid, msg, len(comms) if comms else None)
        all_passed &= valid
        
        # Test standalone function
        comms, valid, msg, runtime = test_standalone_function(leading_eigen_vector, G_karate)
        print_result("Standalone Function", valid, msg, len(comms) if comms else None, runtime)
        all_passed &= valid
        
        # Test algorithm class directly
        algo = LeadingEigenAlgorithm(G_karate)
        comms, runtime = algo.run()
        valid, msg = validate_communities(comms, G_karate)
        print_result("Algorithm Class", valid, msg, len(comms), runtime)
        all_passed &= valid
        
    except Exception as e:
        print(f"  ✗ FAIL: Import/Setup Error - {str(e)}")
        traceback.print_exc()
        all_passed = False
    
    # =========================================
    # Test 6: Girvan-Newman
    # =========================================
    print("\n" + "-" * 40)
    print("6. GIRVAN-NEWMAN")
    print("-" * 40)
    
    try:
        from clusternet.algorithms.girvan_newman_wrapper import GirvanNewmanWrapper
        from girvan_newman import girvan_newman, GirvanNewmanAlgorithm
        
        # Test wrapper (use small graph, GN can be slow)
        comms, valid, msg = test_wrapper_interface(GirvanNewmanWrapper, G_small)
        print_result("Wrapper Interface", valid, msg, len(comms) if comms else None)
        all_passed &= valid
        
        # Test standalone function
        comms, valid, msg, runtime = test_standalone_function(girvan_newman, G_small)
        print_result("Standalone Function", valid, msg, len(comms) if comms else None, runtime)
        all_passed &= valid
        
        # Test algorithm class directly
        algo = GirvanNewmanAlgorithm(G_small)
        comms, runtime = algo.run()
        valid, msg = validate_communities(comms, G_small)
        print_result("Algorithm Class", valid, msg, len(comms), runtime)
        all_passed &= valid
        
    except Exception as e:
        print(f"  ✗ FAIL: Import/Setup Error - {str(e)}")
        traceback.print_exc()
        all_passed = False
    
    # =========================================
    # Test 7: Spectral Clustering
    # =========================================
    print("\n" + "-" * 40)
    print("7. SPECTRAL CLUSTERING")
    print("-" * 40)
    
    try:
        from clusternet.algorithms.spectral_wrapper import SpectralWrapper
        from spectral_clustering import spectral_clustering, SpectralClusteringAlgorithm
        
        # Test wrapper
        comms, valid, msg = test_wrapper_interface(SpectralWrapper, G_karate, n_clusters=4, n_components=4)
        print_result("Wrapper Interface", valid, msg, len(comms) if comms else None)
        all_passed &= valid
        
        # Test standalone function
        comms, valid, msg, runtime = test_standalone_function(spectral_clustering, G_karate, n_clusters=4, n_components=4)
        print_result("Standalone Function", valid, msg, len(comms) if comms else None, runtime)
        all_passed &= valid
        
        # Test algorithm class directly
        algo = SpectralClusteringAlgorithm(G_karate, n_clusters=4, n_components=4)
        comms, runtime = algo.run()
        valid, msg = validate_communities(comms, G_karate)
        print_result("Algorithm Class", valid, msg, len(comms), runtime)
        all_passed &= valid
        
    except Exception as e:
        print(f"  ✗ FAIL: Import/Setup Error - {str(e)}")
        traceback.print_exc()
        all_passed = False
    
    # =========================================
    # Summary
    # =========================================
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

