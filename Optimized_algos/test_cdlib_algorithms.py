"""
Test script for cdlib-integrated algorithms in ClusterNet.
"""

import networkx as nx
from clusternet import CommunityDetector
from clusternet.registry import list_algorithms
import time


def create_test_graph():
    """Create a simple test graph with community structure."""
    G = nx.karate_club_graph()
    return G


def test_algorithm(algo_name, G, **params):
    """Test a single algorithm."""
    print(f"\n{'='*70}")
    print(f"Testing: {algo_name.upper()}")
    print(f"{'='*70}")
    
    try:
        start_time = time.time()
        detector = CommunityDetector(algorithm=algo_name, **params)
        communities = detector.detect(G)
        elapsed = time.time() - start_time
        
        print(f"✓ SUCCESS")
        print(f"  Communities found: {len(communities)}")
        print(f"  Sizes: {[len(c) for c in communities]}")
        print(f"  Time: {elapsed:.3f}s")
        return True
        
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def main():
    """Run tests for all cdlib-integrated algorithms."""
    print("\n" + "="*70)
    print("CDLIB ALGORITHMS INTEGRATION TEST")
    print("="*70)
    
    # Create test graph
    G = create_test_graph()
    print(f"\nTest graph: Zachary's Karate Club")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    # List all algorithms
    print(f"\n{'='*70}")
    print("All Available Algorithms:")
    print(f"{'='*70}")
    algorithms = list_algorithms(verbose=False)
    for i, algo in enumerate(algorithms, 1):
        print(f"{i:2d}. {algo}")
    
    # Test cdlib algorithms
    results = {}
    
    # Statistical algorithms
    print(f"\n\n{'#'*70}")
    print("# STATISTICAL ALGORITHMS")
    print(f"{'#'*70}")
    
    results['em'] = test_algorithm('em', G, k=2)
    results['sbm'] = test_algorithm('sbm', G)
    results['sbm_nested'] = test_algorithm('sbm_nested', G)
    
    # Physics-based algorithms
    print(f"\n\n{'#'*70}")
    print("# PHYSICS-BASED ALGORITHMS")
    print(f"{'#'*70}")
    
    results['cpm'] = test_algorithm('cpm', G, resolution_parameter=1.0)
    results['rb_pots'] = test_algorithm('rb_pots', G, resolution_parameter=1.0)
    results['rber_pots'] = test_algorithm('rber_pots', G, resolution_parameter=1.0)
    
    # Diffusion-based algorithms
    print(f"\n\n{'#'*70}")
    print("# DIFFUSION-BASED ALGORITHMS")
    print(f"{'#'*70}")
    
    results['der'] = test_algorithm('der', G)
    results['async_fluid'] = test_algorithm('async_fluid', G, k=2)
    
    # Structural algorithms
    print(f"\n\n{'#'*70}")
    print("# STRUCTURAL ALGORITHMS")
    print(f"{'#'*70}")
    
    results['scan'] = test_algorithm('scan', G, epsilon=0.5, mu=2)
    results['agdl'] = test_algorithm('agdl', G, number_communities=2, kc=3)
    results['gdmp2'] = test_algorithm('gdmp2', G)
    
    # Overlapping algorithms
    print(f"\n\n{'#'*70}")
    print("# OVERLAPPING ALGORITHMS")
    print(f"{'#'*70}")
    
    results['angel'] = test_algorithm('angel', G, threshold=0.25, min_community_size=3)
    results['surprise_communities'] = test_algorithm('surprise_communities', G)
    
    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    print(f"\nResults by algorithm:")
    for algo, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {algo}")
    
    if passed == total:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  Some tests failed. Check dependencies (graph-tool, angel-cd, etc.)")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()

