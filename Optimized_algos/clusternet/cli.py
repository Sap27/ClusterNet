"""
Command-line interface for ClusterNet.
"""

import argparse
import sys
import time
from typing import Optional

from clusternet import CommunityDetector, list_algorithms
from clusternet.utils import (
    load_graph,
    save_communities,
    print_graph_summary,
    evaluate_communities
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ClusterNet: Community Detection Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Louvain on a network file
  clusternet network.dat louvain --output communities.txt
  
  # Run Leiden with custom resolution
  clusternet network.dat leiden --resolution 1.5
  
  # List all available algorithms
  clusternet --list-algorithms
  
  # Run SVT with custom parameters
  clusternet network.dat svt --params svd_k=100 n_clusters=50
        """
    )
    
    # Positional arguments
    parser.add_argument('input_file', nargs='?', help='Path to edge list file')
    parser.add_argument('algorithm', nargs='?', help='Algorithm to use')
    
    # Optional arguments
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--directed', action='store_true', help='Treat graph as directed')
    parser.add_argument('--unweighted', action='store_true', help='Ignore edge weights')
    parser.add_argument('--list-algorithms', '-l', action='store_true',
                       help='List all available algorithms')
    parser.add_argument('--evaluate', '-e', action='store_true',
                       help='Evaluate community quality')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--format', choices=['simple', 'detailed', 'clusternet'],
                       default='simple', help='Output format')
    
    # Algorithm-specific parameters (captured as key=value pairs)
    parser.add_argument('--params', nargs='*', help='Algorithm parameters (key=value pairs)')
    
    args = parser.parse_args()
    
    # Handle list algorithms
    if args.list_algorithms:
        list_algorithms(verbose=True)
        return 0
    
    # Validate required arguments
    if not args.input_file or not args.algorithm:
        parser.print_help()
        return 1
    
    # Parse algorithm parameters
    algo_params = {}
    if args.params:
        for param in args.params:
            if '=' not in param:
                print(f"Warning: Invalid parameter format '{param}', expected key=value")
                continue
            key, value = param.split('=', 1)
            # Try to convert to appropriate type
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Keep as string if conversion fails
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
            algo_params[key] = value
    
    try:
        # Start timing
        start_time = time.time()
        
        # Load graph
        print(f"\nLoading graph from {args.input_file}...")
        G = load_graph(args.input_file, 
                      directed=args.directed,
                      weighted=not args.unweighted)
        
        if args.verbose:
            print_graph_summary(G)
        
        # Create detector
        print(f"\nRunning {args.algorithm} algorithm...")
        detector = CommunityDetector(algorithm=args.algorithm, **algo_params)
        
        if args.verbose:
            info = detector.get_algorithm_info()
            print(f"  Parameters: {info['parameters']}")
        
        # Detect communities
        communities = detector.detect(G)
        
        # Print results
        elapsed = time.time() - start_time
        print(f"\n✓ Detection complete in {elapsed:.2f}s")
        print(f"  Found {len(communities)} communities")
        
        # Evaluate if requested
        if args.evaluate:
            print("\nEvaluating communities...")
            evaluate_communities(G, communities, verbose=True)
        
        # Save if requested
        if args.output:
            save_communities(communities, args.output, format=args.format)
        else:
            # Print summary
            sizes = [len(c) for c in communities]
            import numpy as np
            print(f"\nCommunity size statistics:")
            print(f"  Min: {min(sizes)}, Max: {max(sizes)}")
            print(f"  Mean: {np.mean(sizes):.1f}, Median: {np.median(sizes):.0f}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nAvailable algorithms:")
        for algo in list_algorithms():
            print(f"  - {algo}")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

