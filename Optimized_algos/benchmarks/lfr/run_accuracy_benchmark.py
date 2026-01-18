#!/usr/bin/env python3
"""
LFR Accuracy Benchmark
======================

Test algorithm accuracy as community structure weakens (increasing μ).

X-axis: Mixing parameter μ (0.1 to 0.7)
Y-axis: AMI score
N realizations per μ value

Output:
- CSV with all results
- Plots showing AMI vs μ for each algorithm
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from config import ACCURACY_CONFIG, NETWORK_PARAMS, ALGORITHMS, ACTIVE_CATEGORIES
from lfr_generator import LFRGenerator, save_network, load_network
from algorithm_runner import AlgorithmRunner, GNNRunner
from metrics import compute_all_metrics


def generate_accuracy_networks(output_dir: str, config: dict, lfr_dir: str, verbose: bool = True):
    """
    Generate all LFR networks for accuracy benchmark.
    
    Args:
        output_dir: Directory to save networks
        config: Accuracy benchmark configuration
        lfr_dir: Path to LFR binaries
        verbose: Print progress
    """
    generator = LFRGenerator(lfr_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    networks_generated = []
    
    for mu in config['mu_values']:
        if verbose:
            print(f"\nGenerating networks for μ={mu}...")
        
        for realization in range(config['realizations']):
            name = f"mu{mu:.1f}_r{realization}"
            seed = int(mu * 1000) + realization  # Deterministic seed
            
            try:
                G, communities = generator.generate_standard(
                    N=config['network_size'],
                    mu=mu,
                    k=NETWORK_PARAMS['base']['k'],
                    maxk=NETWORK_PARAMS['base']['maxk'],
                    t1=NETWORK_PARAMS['base']['t1'],
                    t2=NETWORK_PARAMS['base']['t2'],
                    minc=NETWORK_PARAMS['base']['minc'],
                    maxc=NETWORK_PARAMS['base']['maxc'],
                    seed=seed
                )
                
                save_network(G, communities, str(output_path), name)
                
                networks_generated.append({
                    'name': name,
                    'mu': mu,
                    'realization': realization,
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'communities': len(communities),
                })
                
                if verbose:
                    print(f"  ✓ {name}: {G.number_of_nodes()} nodes, {len(communities)} communities")
                    
            except Exception as e:
                print(f"  ✗ {name}: Failed - {e}")
    
    # Save metadata
    metadata = {
        'benchmark': 'accuracy',
        'config': config,
        'networks': networks_generated,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return networks_generated


def run_accuracy_benchmark(
    networks_dir: str,
    results_dir: str,
    algorithms: list,
    include_gnn: bool = True,
    verbose: bool = True
):
    """
    Run accuracy benchmark on generated networks.
    
    Args:
        networks_dir: Directory containing networks
        results_dir: Directory to save results
        algorithms: List of algorithm names to test
        include_gnn: Include GNN algorithms
        verbose: Print progress
    """
    networks_path = Path(networks_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(networks_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    runner = AlgorithmRunner(verbose=False)
    gnn_runner = GNNRunner(device='cpu', verbose=False) if include_gnn else None
    
    all_results = []
    
    for network_info in metadata['networks']:
        name = network_info['name']
        mu = network_info['mu']
        
        if verbose:
            print(f"\nProcessing {name} (μ={mu})...")
        
        # Load network
        G, true_communities = load_network(str(networks_path), name)
        num_true_comms = len(true_communities)
        
        # Run classical algorithms
        for algo_name in algorithms:
            if verbose:
                print(f"  Running {algo_name}...", end=' ', flush=True)
            
            result = runner.run_algorithm(G, algo_name)
            
            if result.success and result.communities:
                metrics = compute_all_metrics(G, true_communities, result.communities)
                
                all_results.append({
                    'network': name,
                    'mu': mu,
                    'realization': network_info['realization'],
                    'algorithm': algo_name,
                    'algorithm_type': 'classical',
                    'success': True,
                    'runtime': result.runtime,
                    'num_detected': len(result.communities),
                    'num_true': num_true_comms,
                    **metrics
                })
                
                if verbose:
                    print(f"✓ AMI={metrics['ami']:.3f}")
            else:
                all_results.append({
                    'network': name,
                    'mu': mu,
                    'realization': network_info['realization'],
                    'algorithm': algo_name,
                    'algorithm_type': 'classical',
                    'success': False,
                    'runtime': result.runtime,
                    'error': result.error_message
                })
                if verbose:
                    print(f"✗")
        
        # Run GNN algorithms
        if gnn_runner and gnn_runner.torch_available:
            for gnn_name in ['dmon', 'mincut']:
                if verbose:
                    print(f"  Running {gnn_name}...", end=' ', flush=True)
                
                if gnn_name == 'dmon':
                    result = gnn_runner.run_dmon(G, num_communities=num_true_comms, epochs=200)
                elif gnn_name == 'mincut':
                    result = gnn_runner.run_mincut(G, num_communities=num_true_comms, epochs=200)
                
                if result.success and result.communities:
                    metrics = compute_all_metrics(G, true_communities, result.communities)
                    
                    all_results.append({
                        'network': name,
                        'mu': mu,
                        'realization': network_info['realization'],
                        'algorithm': gnn_name,
                        'algorithm_type': 'gnn',
                        'success': True,
                        'runtime': result.runtime,
                        'num_detected': len(result.communities),
                        'num_true': num_true_comms,
                        **metrics
                    })
                    
                    if verbose:
                        print(f"✓ AMI={metrics['ami']:.3f}")
                else:
                    all_results.append({
                        'network': name,
                        'mu': mu,
                        'realization': network_info['realization'],
                        'algorithm': gnn_name,
                        'algorithm_type': 'gnn',
                        'success': False,
                        'runtime': result.runtime,
                        'error': result.error_message
                    })
                    if verbose:
                        print(f"✗")
    
    # Save results
    df = pd.DataFrame(all_results)
    df.to_csv(results_path / 'accuracy_results.csv', index=False)
    
    # Compute summary statistics
    summary = df[df['success'] == True].groupby(['mu', 'algorithm']).agg({
        'ami': ['mean', 'std'],
        'nmi': ['mean', 'std'],
        'modularity': ['mean', 'std'],
        'runtime': ['mean', 'std'],
        'num_detected': 'mean'
    }).round(4)
    
    summary.to_csv(results_path / 'accuracy_summary.csv')
    
    if verbose:
        print("\n" + "="*60)
        print("ACCURACY BENCHMARK COMPLETE")
        print("="*60)
        print(f"Results saved to: {results_path}")
        print(f"Total runs: {len(all_results)}")
        print(f"Successful: {sum(1 for r in all_results if r.get('success', False))}")
    
    return df


def plot_accuracy_results(results_file: str, output_dir: str):
    """Generate plots from accuracy results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not available for plotting")
        return
    
    df = pd.read_csv(results_file)
    df_success = df[df['success'] == True]
    
    output_path = Path(output_dir)
    
    # Plot 1: AMI vs μ for all algorithms
    plt.figure(figsize=(12, 8))
    
    for algo in df_success['algorithm'].unique():
        algo_data = df_success[df_success['algorithm'] == algo]
        summary = algo_data.groupby('mu')['ami'].agg(['mean', 'std']).reset_index()
        
        plt.errorbar(
            summary['mu'],
            summary['mean'],
            yerr=summary['std'],
            marker='o',
            label=algo,
            capsize=3
        )
    
    plt.xlabel('Mixing Parameter (μ)', fontsize=12)
    plt.ylabel('Adjusted Mutual Information (AMI)', fontsize=12)
    plt.title('LFR Accuracy Benchmark: AMI vs Mixing Parameter', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_ami_vs_mu.png', dpi=150)
    plt.close()
    
    # Plot 2: Heatmap of mean AMI
    pivot = df_success.pivot_table(
        values='ami',
        index='algorithm',
        columns='mu',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
    plt.title('Mean AMI by Algorithm and Mixing Parameter')
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_heatmap.png', dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LFR Accuracy Benchmark')
    parser.add_argument('--generate', action='store_true', help='Generate networks')
    parser.add_argument('--run', action='store_true', help='Run benchmark')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--all', action='store_true', help='Generate, run, and plot')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--mu-values', type=str, default=None, 
                        help='Comma-separated μ values (e.g., "0.1,0.3,0.5")')
    parser.add_argument('--realizations', type=int, default=None,
                        help='Number of realizations per μ')
    parser.add_argument('--no-gnn', action='store_true', help='Skip GNN algorithms')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = Path(__file__).parent
    lfr_dir = script_dir.parent.parent.parent.parent / 'LFRbenchmarks'
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / 'accuracy'
    
    networks_dir = output_dir / 'networks'
    results_dir = output_dir / 'results'
    
    # Update config if needed
    config = ACCURACY_CONFIG.copy()
    if args.mu_values:
        config['mu_values'] = [float(x) for x in args.mu_values.split(',')]
    if args.realizations:
        config['realizations'] = args.realizations
    
    # Get algorithms to test
    algorithms = []
    for category in ACTIVE_CATEGORIES:
        algorithms.extend(ALGORITHMS.get(category, []))
    
    if args.all or args.generate:
        print("="*60)
        print("GENERATING LFR NETWORKS FOR ACCURACY BENCHMARK")
        print("="*60)
        print(f"μ values: {config['mu_values']}")
        print(f"Realizations per μ: {config['realizations']}")
        print(f"Network size: {config['network_size']}")
        
        generate_accuracy_networks(
            str(networks_dir),
            config,
            str(lfr_dir),
            verbose=True
        )
    
    if args.all or args.run:
        print("\n" + "="*60)
        print("RUNNING ACCURACY BENCHMARK")
        print("="*60)
        print(f"Algorithms: {algorithms}")
        print(f"Include GNN: {not args.no_gnn}")
        
        run_accuracy_benchmark(
            str(networks_dir),
            str(results_dir),
            algorithms,
            include_gnn=not args.no_gnn,
            verbose=True
        )
    
    if args.all or args.plot:
        results_file = results_dir / 'accuracy_results.csv'
        if results_file.exists():
            print("\n" + "="*60)
            print("GENERATING PLOTS")
            print("="*60)
            plot_accuracy_results(str(results_file), str(results_dir))
        else:
            print("No results file found. Run benchmark first.")


if __name__ == '__main__':
    main()




