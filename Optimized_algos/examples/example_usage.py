"""
Example usage of ClusterNet package.

This script demonstrates various ways to use ClusterNet for community detection.
"""

import networkx as nx
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from clusternet import CommunityDetector, BatchDetector, list_algorithms
from clusternet.utils import (
    load_graph,
    save_communities,
    evaluate_communities,
    compare_communities,
    print_graph_summary
)


def example_1_basic_usage():
    """Example 1: Basic community detection."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    # Create or load a graph
    G = nx.karate_club_graph()
    print(f"Loaded Karate Club graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Detect communities using Louvain
    detector = CommunityDetector(algorithm='louvain', resolution=1.0)
    communities = detector.detect(G)
    
    print(f"\nFound {len(communities)} communities")
    for i, comm in enumerate(communities, 1):
        print(f"  Community {i}: {len(comm)} nodes")


def example_2_load_from_file():
    """Example 2: Load graph from file."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Load from File")
    print("="*70)
    
    # Check if network.dat exists
    if not os.path.exists('network.dat'):
        print("Skipping: network.dat not found")
        return
    
    # Load graph
    G = load_graph('network.dat', directed=False, weighted=True)
    print_graph_summary(G)
    
    # Detect communities
    detector = CommunityDetector(algorithm='leiden', resolution=1.0)
    communities = detector.detect(G)
    
    print(f"\nDetected {len(communities)} communities")
    
    # Save to file
    save_communities(communities, 'output_communities.txt', format='detailed')
    print("✓ Communities saved to output_communities.txt")


def example_3_compare_algorithms():
    """Example 3: Compare multiple algorithms."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Compare Multiple Algorithms")
    print("="*70)
    
    G = nx.karate_club_graph()
    
    # Run multiple algorithms
    algorithms = ['louvain', 'leiden', 'score']
    batch = BatchDetector(algorithms=algorithms)
    results = batch.detect_all(G)
    
    print("\nResults:")
    for algo, communities in results.items():
        if communities:
            print(f"  {algo:15s}: {len(communities):3d} communities")


def example_4_evaluate_quality():
    """Example 4: Evaluate community quality."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Evaluate Community Quality")
    print("="*70)
    
    G = nx.karate_club_graph()
    
    # Detect communities
    detector = CommunityDetector(algorithm='louvain')
    communities = detector.detect(G)
    
    # Evaluate
    metrics = evaluate_communities(G, communities, verbose=True)


def example_5_compare_results():
    """Example 5: Compare two community structures."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Compare Two Community Structures")
    print("="*70)
    
    G = nx.karate_club_graph()
    
    # Detect with Louvain
    detector1 = CommunityDetector(algorithm='louvain')
    communities1 = detector1.detect(G)
    print(f"Louvain: {len(communities1)} communities")
    
    # Detect with Leiden
    detector2 = CommunityDetector(algorithm='leiden')
    communities2 = detector2.detect(G)
    print(f"Leiden: {len(communities2)} communities")
    
    # Compare
    metrics = compare_communities(communities1, communities2, verbose=True)


def example_6_custom_parameters():
    """Example 6: Using custom parameters."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Parameters")
    print("="*70)
    
    G = nx.karate_club_graph()
    
    # Try different resolution values
    resolutions = [0.5, 1.0, 2.0]
    
    for res in resolutions:
        detector = CommunityDetector(algorithm='louvain', resolution=res)
        communities = detector.detect(G)
        print(f"Resolution {res}: {len(communities)} communities")


def example_7_list_algorithms():
    """Example 7: List all available algorithms."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Available Algorithms")
    print("="*70)
    
    list_algorithms(verbose=True)


def main():
    """Run all examples."""
    print("\n" + "#"*70)
    print("# CLUSTERNET USAGE EXAMPLES")
    print("#"*70)
    
    examples = [
        example_1_basic_usage,
        example_2_load_from_file,
        example_3_compare_algorithms,
        example_4_evaluate_quality,
        example_5_compare_results,
        example_6_custom_parameters,
        example_7_list_algorithms,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "#"*70)
    print("# ALL EXAMPLES COMPLETED")
    print("#"*70)


if __name__ == '__main__':
    main()

