#!/usr/bin/env python3
"""
LFR Benchmark Master Runner
===========================

Run all LFR benchmarks with minimal configuration for Mac testing.

Usage:
    python run_all_benchmarks.py --minimal   # Quick test (2 realizations)
    python run_all_benchmarks.py --full      # Full benchmark (10 realizations)
    python run_all_benchmarks.py --accuracy  # Only accuracy benchmark
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import json


def print_banner(text):
    """Print a banner."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_benchmark(script_name: str, args: list, verbose: bool = True):
    """Run a benchmark script."""
    script_path = Path(__file__).parent / script_name
    
    cmd = [sys.executable, str(script_path)] + args
    
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def check_dependencies():
    """Check that required packages are installed."""
    required = ['networkx', 'numpy', 'pandas', 'sklearn']
    optional = ['matplotlib', 'seaborn', 'torch', 'torch_geometric', 'igraph', 'leidenalg']
    
    print("Checking dependencies...")
    
    missing_required = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} (REQUIRED)")
            missing_required.append(pkg)
    
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ○ {pkg} (optional)")
    
    if missing_required:
        print(f"\nERROR: Missing required packages: {missing_required}")
        print("Install with: pip install " + " ".join(missing_required))
        return False
    
    return True


def check_lfr_binaries():
    """Check that LFR binaries are compiled."""
    script_dir = Path(__file__).parent
    lfr_dir = script_dir.parent.parent.parent.parent / 'LFRbenchmarks'
    
    binaries = [
        'unweighted_undirected/benchmark',
        'hierarchical/hbenchmark',
    ]
    
    print("\nChecking LFR binaries...")
    
    missing = []
    for binary in binaries:
        path = lfr_dir / binary
        if path.exists():
            print(f"  ✓ {binary}")
        else:
            print(f"  ✗ {binary} (NOT FOUND)")
            missing.append(binary)
    
    if missing:
        print(f"\nLFR binaries not found. Compiling...")
        for binary in missing:
            binary_dir = lfr_dir / Path(binary).parent
            print(f"  Compiling in {binary_dir}...")
            result = subprocess.run(['make'], cwd=binary_dir, capture_output=True)
            if result.returncode == 0:
                print(f"    ✓ Compiled successfully")
            else:
                print(f"    ✗ Compilation failed: {result.stderr.decode()}")
                return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='LFR Benchmark Suite')
    parser.add_argument('--minimal', action='store_true', 
                        help='Quick test with 2 realizations')
    parser.add_argument('--full', action='store_true',
                        help='Full benchmark with 10 realizations')
    parser.add_argument('--accuracy', action='store_true',
                        help='Run only accuracy benchmark')
    parser.add_argument('--scalability', action='store_true',
                        help='Run only scalability benchmark')
    parser.add_argument('--hierarchical', action='store_true',
                        help='Run only hierarchical benchmark')
    parser.add_argument('--overlapping', action='store_true',
                        help='Run only overlapping benchmark')
    parser.add_argument('--no-gnn', action='store_true',
                        help='Skip GNN algorithms (faster)')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check dependencies and binaries')
    
    args = parser.parse_args()
    
    print_banner("LFR BENCHMARK SUITE")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Check binaries
    if not check_lfr_binaries():
        return 1
    
    if args.check_only:
        print("\nAll checks passed! Ready to run benchmarks.")
        return 0
    
    # Determine realizations
    if args.minimal:
        realizations = 2
        print("\nRunning MINIMAL benchmark (2 realizations per point)")
    elif args.full:
        realizations = 10
        print("\nRunning FULL benchmark (10 realizations per point)")
    else:
        realizations = 3
        print("\nRunning DEFAULT benchmark (3 realizations per point)")
    
    # Determine which benchmarks to run
    run_all = not (args.accuracy or args.scalability or args.hierarchical or args.overlapping)
    
    gnn_flag = ['--no-gnn'] if args.no_gnn else []
    
    results = {}
    start_time = datetime.now()
    
    # Accuracy benchmark
    if run_all or args.accuracy:
        print_banner("ACCURACY BENCHMARK")
        success = run_benchmark(
            'run_accuracy_benchmark.py',
            ['--all', '--realizations', str(realizations)] + gnn_flag
        )
        results['accuracy'] = 'SUCCESS' if success else 'FAILED'
    
    # Scalability benchmark
    if run_all or args.scalability:
        print_banner("SCALABILITY BENCHMARK")
        success = run_benchmark(
            'run_scalability_benchmark.py',
            ['--all', '--realizations', str(realizations)] + gnn_flag
        )
        results['scalability'] = 'SUCCESS' if success else 'FAILED'
    
    # Hierarchical benchmark
    if run_all or args.hierarchical:
        print_banner("HIERARCHICAL BENCHMARK")
        success = run_benchmark(
            'run_hierarchical_benchmark.py',
            ['--all', '--realizations', str(realizations)] + gnn_flag
        )
        results['hierarchical'] = 'SUCCESS' if success else 'FAILED'
    
    # Overlapping benchmark
    if run_all or args.overlapping:
        print_banner("OVERLAPPING BENCHMARK")
        success = run_benchmark(
            'run_overlapping_benchmark.py',
            ['--all', '--realizations', str(realizations)] + gnn_flag
        )
        results['overlapping'] = 'SUCCESS' if success else 'FAILED'
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print_banner("BENCHMARK SUMMARY")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print()
    
    for benchmark, status in results.items():
        icon = '✓' if status == 'SUCCESS' else '✗'
        print(f"  {icon} {benchmark}: {status}")
    
    print()
    print("Results saved to:")
    script_dir = Path(__file__).parent
    for benchmark in results.keys():
        results_dir = script_dir / benchmark / 'results'
        if results_dir.exists():
            print(f"  - {results_dir}")
    
    # Save summary
    summary = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'realizations': realizations,
        'include_gnn': not args.no_gnn,
        'results': results
    }
    
    with open(script_dir / 'benchmark_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return 0 if all(s == 'SUCCESS' for s in results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())




