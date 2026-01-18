#!/usr/bin/env python3
"""
LFR Scalability Benchmark
=========================

Test algorithm runtime as network size increases.

X-axis: Network size N
Y-axis: Runtime (seconds), also AMI for quality check
N realizations per network size

Output:
- CSV with all results
- Plots showing Runtime vs N and AMI vs N
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

# Add paths
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from config import SCALABILITY_CONFIG, NETWORK_PARAMS, ALGORITHMS, ACTIVE_CATEGORIES
from lfr_generator import LFRGenerator, save_network, load_network
from algorithm_runner import AlgorithmRunner, GNNRunner
from metrics import compute_all_metrics


def generate_scalability_networks(output_dir: str, config: dict, lfr_dir: str, verbose: bool = True):
    """Generate LFR networks for scalability benchmark."""
    generator = LFRGenerator(lfr_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    networks_generated = []
    
    for N in config['network_sizes']:
        if verbose:
            print(f"\nGenerating networks for N={N}...")
        
        # Adjust community sizes based on network size
        minc = max(10, N // 100)
        maxc = max(50, N // 20)
        maxk = min(100, N // 10)
        
        for realization in range(config['realizations']):
            name = f"N{N}_r{realization}"
            seed = N + realization
            
            try:
                G, communities = generator.generate_standard(
                    N=N,
                    mu=config['mu'],
                    k=NETWORK_PARAMS['base']['k'],
                    maxk=maxk,
                    t1=NETWORK_PARAMS['base']['t1'],
                    t2=NETWORK_PARAMS['base']['t2'],
                    minc=minc,
                    maxc=maxc,
                    seed=seed
                )
                
                save_network(G, communities, str(output_path), name)
                
                networks_generated.append({
                    'name': name,
                    'N': N,
                    'realization': realization,
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'communities': len(communities),
                })
                
                if verbose:
                    print(f"  ✓ {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                    
            except Exception as e:
                print(f"  ✗ {name}: Failed - {e}")
    
    metadata = {
        'benchmark': 'scalability',
        'config': config,
        'networks': networks_generated,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return networks_generated


def run_scalability_benchmark(
    networks_dir: str,
    results_dir: str,
    algorithms: list,
    include_gnn: bool = True,
    verbose: bool = True
):
    """Run scalability benchmark on generated networks."""
    networks_path = Path(networks_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    with open(networks_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    runner = AlgorithmRunner(verbose=False)
    gnn_runner = GNNRunner(device='cpu', verbose=False) if include_gnn else None
    
    all_results = []
    
    for network_info in metadata['networks']:
        name = network_info['name']
        N = network_info['N']
        
        if verbose:
            print(f"\nProcessing {name} (N={N})...")
        
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
                    'N': N,
                    'realization': network_info['realization'],
                    'edges': network_info['edges'],
                    'algorithm': algo_name,
                    'algorithm_type': 'classical',
                    'success': True,
                    'runtime': result.runtime,
                    'num_detected': len(result.communities),
                    'num_true': num_true_comms,
                    **metrics
                })
                
                if verbose:
                    print(f"✓ {result.runtime:.3f}s, AMI={metrics['ami']:.3f}")
            else:
                all_results.append({
                    'network': name,
                    'N': N,
                    'realization': network_info['realization'],
                    'edges': network_info['edges'],
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
            # Reduce epochs for larger networks
            epochs = max(50, 200 - N // 50)
            
            for gnn_name in ['dmon', 'mincut']:
                if verbose:
                    print(f"  Running {gnn_name}...", end=' ', flush=True)
                
                if gnn_name == 'dmon':
                    result = gnn_runner.run_dmon(G, num_communities=num_true_comms, epochs=epochs)
                elif gnn_name == 'mincut':
                    result = gnn_runner.run_mincut(G, num_communities=num_true_comms, epochs=epochs)
                
                if result.success and result.communities:
                    metrics = compute_all_metrics(G, true_communities, result.communities)
                    
                    all_results.append({
                        'network': name,
                        'N': N,
                        'realization': network_info['realization'],
                        'edges': network_info['edges'],
                        'algorithm': gnn_name,
                        'algorithm_type': 'gnn',
                        'success': True,
                        'runtime': result.runtime,
                        'num_detected': len(result.communities),
                        'num_true': num_true_comms,
                        **metrics
                    })
                    
                    if verbose:
                        print(f"✓ {result.runtime:.3f}s, AMI={metrics['ami']:.3f}")
                else:
                    all_results.append({
                        'network': name,
                        'N': N,
                        'realization': network_info['realization'],
                        'edges': network_info['edges'],
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
    df.to_csv(results_path / 'scalability_results.csv', index=False)
    
    # Compute summary
    summary = df[df['success'] == True].groupby(['N', 'algorithm']).agg({
        'runtime': ['mean', 'std'],
        'ami': ['mean', 'std'],
        'edges': 'first'
    }).round(4)
    
    summary.to_csv(results_path / 'scalability_summary.csv')
    
    if verbose:
        print("\n" + "="*60)
        print("SCALABILITY BENCHMARK COMPLETE")
        print("="*60)
        print(f"Results saved to: {results_path}")
    
    return df


def plot_scalability_results(results_file: str, output_dir: str):
    """Generate plots from scalability results."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not available for plotting")
        return
    
    df = pd.read_csv(results_file)
    df_success = df[df['success'] == True]
    
    output_path = Path(output_dir)
    
    # Plot 1: Runtime vs N (log scale)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Runtime plot
    ax1 = axes[0]
    for algo in df_success['algorithm'].unique():
        algo_data = df_success[df_success['algorithm'] == algo]
        summary = algo_data.groupby('N')['runtime'].agg(['mean', 'std']).reset_index()
        
        ax1.errorbar(
            summary['N'],
            summary['mean'],
            yerr=summary['std'],
            marker='o',
            label=algo,
            capsize=3
        )
    
    ax1.set_xlabel('Network Size (N)', fontsize=12)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Runtime vs Network Size', fontsize=14)
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # AMI plot
    ax2 = axes[1]
    for algo in df_success['algorithm'].unique():
        algo_data = df_success[df_success['algorithm'] == algo]
        summary = algo_data.groupby('N')['ami'].agg(['mean', 'std']).reset_index()
        
        ax2.errorbar(
            summary['N'],
            summary['mean'],
            yerr=summary['std'],
            marker='o',
            label=algo,
            capsize=3
        )
    
    ax2.set_xlabel('Network Size (N)', fontsize=12)
    ax2.set_ylabel('AMI', fontsize=12)
    ax2.set_title('Accuracy vs Network Size', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'scalability_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Scaling heatmap
    pivot = df_success.pivot_table(
        values='runtime',
        index='algorithm',
        columns='N',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(np.log10(pivot + 0.001), annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title('Log10(Runtime) by Algorithm and Network Size')
    plt.tight_layout()
    plt.savefig(output_path / 'scalability_heatmap.png', dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LFR Scalability Benchmark')
    parser.add_argument('--generate', action='store_true', help='Generate networks')
    parser.add_argument('--run', action='store_true', help='Run benchmark')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--all', action='store_true', help='Generate, run, and plot')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--sizes', type=str, default=None,
                        help='Comma-separated network sizes (e.g., "500,1000,2000")')
    parser.add_argument('--realizations', type=int, default=None,
                        help='Number of realizations per size')
    parser.add_argument('--no-gnn', action='store_true', help='Skip GNN algorithms')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    lfr_dir = script_dir.parent.parent.parent.parent / 'LFRbenchmarks'
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / 'scalability'
    
    networks_dir = output_dir / 'networks'
    results_dir = output_dir / 'results'
    
    config = SCALABILITY_CONFIG.copy()
    if args.sizes:
        config['network_sizes'] = [int(x) for x in args.sizes.split(',')]
    if args.realizations:
        config['realizations'] = args.realizations
    
    algorithms = []
    for category in ACTIVE_CATEGORIES:
        algorithms.extend(ALGORITHMS.get(category, []))
    
    if args.all or args.generate:
        print("="*60)
        print("GENERATING LFR NETWORKS FOR SCALABILITY BENCHMARK")
        print("="*60)
        print(f"Network sizes: {config['network_sizes']}")
        print(f"Realizations per size: {config['realizations']}")
        print(f"Fixed μ: {config['mu']}")
        
        generate_scalability_networks(
            str(networks_dir),
            config,
            str(lfr_dir),
            verbose=True
        )
    
    if args.all or args.run:
        print("\n" + "="*60)
        print("RUNNING SCALABILITY BENCHMARK")
        print("="*60)
        
        run_scalability_benchmark(
            str(networks_dir),
            str(results_dir),
            algorithms,
            include_gnn=not args.no_gnn,
            verbose=True
        )
    
    if args.all or args.plot:
        results_file = results_dir / 'scalability_results.csv'
        if results_file.exists():
            print("\n" + "="*60)
            print("GENERATING PLOTS")
            print("="*60)
            plot_scalability_results(str(results_file), str(results_dir))


if __name__ == '__main__':
    main()




