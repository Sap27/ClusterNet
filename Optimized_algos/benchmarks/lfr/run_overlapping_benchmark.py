#!/usr/bin/env python3
"""
LFR Overlapping Benchmark
=========================

Test algorithm ability to detect overlapping communities.

X-axis: Fraction of overlapping nodes (0% to 40%)
Y-axis: ONMI, Omega Index, F1 score

Tests both classical overlapping algorithms and GNNs.
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

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from config import OVERLAPPING_CONFIG, NETWORK_PARAMS, ALGORITHMS, OVERLAPPING_CATEGORIES
from lfr_generator import LFRGenerator, save_network, load_network
from algorithm_runner import AlgorithmRunner, GNNRunner
from metrics import compute_all_metrics, compute_onmi, compute_omega_index, compute_f1_communities
import networkx as nx


def generate_overlapping_networks(output_dir: str, config: dict, lfr_dir: str, verbose: bool = True):
    """Generate LFR networks with varying overlap fractions."""
    generator = LFRGenerator(lfr_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    networks_generated = []
    
    for overlap_frac in config['overlap_fractions']:
        if verbose:
            print(f"\nGenerating networks with {overlap_frac*100:.0f}% overlapping nodes...")
        
        for realization in range(config['realizations']):
            name = f"overlap{overlap_frac:.1f}_r{realization}"
            seed = int(overlap_frac * 1000) + realization
            
            try:
                G, communities = generator.generate_overlapping(
                    N=config['network_size'],
                    mu=config['mu'],
                    overlap_fraction=overlap_frac,
                    om=config['om'],
                    k=NETWORK_PARAMS['base']['k'],
                    maxk=NETWORK_PARAMS['base']['maxk'],
                    minc=NETWORK_PARAMS['base']['minc'],
                    maxc=NETWORK_PARAMS['base']['maxc'],
                    seed=seed
                )
                
                # Count actual overlapping nodes
                from collections import Counter
                node_memberships = Counter()
                for comm in communities:
                    for node in comm:
                        node_memberships[node] += 1
                overlapping_count = sum(1 for n, c in node_memberships.items() if c > 1)
                
                save_network(G, communities, str(output_path), name)
                
                networks_generated.append({
                    'name': name,
                    'overlap_fraction': overlap_frac,
                    'actual_overlapping': overlapping_count,
                    'actual_overlap_pct': overlapping_count / G.number_of_nodes(),
                    'realization': realization,
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'communities': len(communities),
                })
                
                if verbose:
                    print(f"  ✓ {name}: {overlapping_count} overlapping nodes ({overlapping_count/G.number_of_nodes()*100:.1f}%)")
                    
            except Exception as e:
                print(f"  ✗ {name}: Failed - {e}")
    
    metadata = {
        'benchmark': 'overlapping',
        'config': config,
        'networks': networks_generated,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return networks_generated


def run_overlapping_benchmark(
    networks_dir: str,
    results_dir: str,
    algorithms: list,
    include_gnn: bool = True,
    verbose: bool = True
):
    """Run overlapping benchmark on generated networks."""
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
        overlap_frac = network_info['overlap_fraction']
        
        if verbose:
            print(f"\nProcessing {name} (overlap={overlap_frac*100:.0f}%)...")
        
        G, true_communities = load_network(str(networks_path), name)
        num_nodes = G.number_of_nodes()
        
        # Run classical overlapping algorithms
        for algo_name in algorithms:
            if verbose:
                print(f"  Running {algo_name}...", end=' ', flush=True)
            
            result = runner.run_algorithm(G, algo_name)
            
            if result.success and result.communities:
                # Compute overlapping-specific metrics
                onmi = compute_onmi(true_communities, result.communities, num_nodes)
                omega = compute_omega_index(true_communities, result.communities, num_nodes)
                f1 = compute_f1_communities(true_communities, result.communities)
                
                # Also compute standard metrics (treating as disjoint)
                metrics = compute_all_metrics(G, true_communities, result.communities, overlapping=True)
                
                all_results.append({
                    'network': name,
                    'overlap_fraction': overlap_frac,
                    'realization': network_info['realization'],
                    'algorithm': algo_name,
                    'algorithm_type': 'overlapping' if algo_name in ['angel', 'demon', 'kclique'] else 'disjoint',
                    'success': True,
                    'runtime': result.runtime,
                    'onmi': onmi,
                    'omega_index': omega,
                    'f1': f1,
                    'ami': metrics['ami'],
                    'modularity': metrics['modularity'],
                    'num_detected': len(result.communities),
                    'num_true': len(true_communities),
                })
                
                if verbose:
                    print(f"✓ ONMI={onmi:.3f}, Omega={omega:.3f}")
            else:
                all_results.append({
                    'network': name,
                    'overlap_fraction': overlap_frac,
                    'realization': network_info['realization'],
                    'algorithm': algo_name,
                    'algorithm_type': 'overlapping' if algo_name in ['angel', 'demon', 'kclique'] else 'disjoint',
                    'success': False,
                    'runtime': result.runtime,
                    'error': result.error_message
                })
                if verbose:
                    print(f"✗")
        
        # Run GNN algorithms
        if gnn_runner and gnn_runner.torch_available:
            num_true_comms = len(true_communities)
            
            for gnn_name in ['dmon', 'mincut']:
                if verbose:
                    print(f"  Running {gnn_name}...", end=' ', flush=True)
                
                if gnn_name == 'dmon':
                    result = gnn_runner.run_dmon(G, num_communities=num_true_comms, epochs=200)
                else:
                    result = gnn_runner.run_mincut(G, num_communities=num_true_comms, epochs=200)
                
                if result.success and result.communities:
                    onmi = compute_onmi(true_communities, result.communities, num_nodes)
                    omega = compute_omega_index(true_communities, result.communities, num_nodes)
                    f1 = compute_f1_communities(true_communities, result.communities)
                    metrics = compute_all_metrics(G, true_communities, result.communities, overlapping=True)
                    
                    all_results.append({
                        'network': name,
                        'overlap_fraction': overlap_frac,
                        'realization': network_info['realization'],
                        'algorithm': gnn_name,
                        'algorithm_type': 'gnn',
                        'success': True,
                        'runtime': result.runtime,
                        'onmi': onmi,
                        'omega_index': omega,
                        'f1': f1,
                        'ami': metrics['ami'],
                        'modularity': metrics['modularity'],
                        'num_detected': len(result.communities),
                        'num_true': num_true_comms,
                    })
                    
                    if verbose:
                        print(f"✓ ONMI={onmi:.3f}")
                else:
                    all_results.append({
                        'network': name,
                        'overlap_fraction': overlap_frac,
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
    df.to_csv(results_path / 'overlapping_results.csv', index=False)
    
    # Summary
    df_success = df[df['success'] == True]
    summary = df_success.groupby(['overlap_fraction', 'algorithm']).agg({
        'onmi': ['mean', 'std'],
        'omega_index': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'ami': ['mean', 'std'],
        'runtime': 'mean',
    }).round(4)
    
    summary.to_csv(results_path / 'overlapping_summary.csv')
    
    if verbose:
        print("\n" + "="*60)
        print("OVERLAPPING BENCHMARK COMPLETE")
        print("="*60)
        print(f"Results saved to: {results_path}")
    
    return df


def plot_overlapping_results(results_file: str, output_dir: str):
    """Generate plots for overlapping benchmark."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not available for plotting")
        return
    
    df = pd.read_csv(results_file)
    df_success = df[df['success'] == True]
    
    output_path = Path(output_dir)
    
    # Plot 1: Metrics vs Overlap Fraction
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics_to_plot = ['onmi', 'omega_index', 'f1']
    titles = ['ONMI', 'Omega Index', 'F1 Score']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx]
        
        for algo in df_success['algorithm'].unique():
            algo_data = df_success[df_success['algorithm'] == algo]
            if metric in algo_data.columns and algo_data[metric].notna().any():
                summary = algo_data.groupby('overlap_fraction')[metric].agg(['mean', 'std']).reset_index()
                
                ax.errorbar(
                    summary['overlap_fraction'] * 100,
                    summary['mean'],
                    yerr=summary['std'],
                    marker='o',
                    label=algo,
                    capsize=3
                )
        
        ax.set_xlabel('Overlapping Nodes (%)')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs Overlap Fraction')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Overlapping Community Detection Performance', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'overlapping_metrics.png', dpi=150)
    plt.close()
    
    # Plot 2: Compare overlapping vs disjoint algorithms
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by algorithm type
    df_success['algo_type'] = df_success['algorithm_type']
    
    for algo_type in ['overlapping', 'disjoint', 'gnn']:
        type_data = df_success[df_success['algo_type'] == algo_type]
        if len(type_data) > 0 and 'onmi' in type_data.columns:
            summary = type_data.groupby('overlap_fraction')['onmi'].agg(['mean', 'std']).reset_index()
            
            ax.errorbar(
                summary['overlap_fraction'] * 100,
                summary['mean'],
                yerr=summary['std'],
                marker='o',
                label=f'{algo_type} algorithms',
                capsize=3,
                linewidth=2
            )
    
    ax.set_xlabel('Overlapping Nodes (%)', fontsize=12)
    ax.set_ylabel('ONMI', fontsize=12)
    ax.set_title('Overlapping vs Disjoint Algorithm Performance', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'overlapping_comparison.png', dpi=150)
    plt.close()
    
    # Heatmap
    if 'onmi' in df_success.columns:
        pivot = df_success.pivot_table(
            values='onmi',
            index='algorithm',
            columns='overlap_fraction',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
        plt.title('Mean ONMI by Algorithm and Overlap Fraction')
        plt.xlabel('Overlap Fraction')
        plt.tight_layout()
        plt.savefig(output_path / 'overlapping_heatmap.png', dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LFR Overlapping Benchmark')
    parser.add_argument('--generate', action='store_true', help='Generate networks')
    parser.add_argument('--run', action='store_true', help='Run benchmark')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--all', action='store_true', help='Generate, run, and plot')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--realizations', type=int, default=None)
    parser.add_argument('--no-gnn', action='store_true', help='Skip GNN algorithms')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    lfr_dir = script_dir.parent.parent.parent.parent / 'LFRbenchmarks'
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / 'overlapping'
    
    networks_dir = output_dir / 'networks'
    results_dir = output_dir / 'results'
    
    config = OVERLAPPING_CONFIG.copy()
    if args.realizations:
        config['realizations'] = args.realizations
    
    # Get overlapping + some disjoint algorithms for comparison
    algorithms = []
    for category in OVERLAPPING_CATEGORIES:
        algorithms.extend(ALGORITHMS.get(category, []))
    # Remove duplicates
    algorithms = list(dict.fromkeys(algorithms))
    
    if args.all or args.generate:
        print("="*60)
        print("GENERATING OVERLAPPING LFR NETWORKS")
        print("="*60)
        print(f"Overlap fractions: {config['overlap_fractions']}")
        print(f"Memberships per overlapping node: {config['om']}")
        print(f"Realizations: {config['realizations']}")
        
        generate_overlapping_networks(
            str(networks_dir),
            config,
            str(lfr_dir),
            verbose=True
        )
    
    if args.all or args.run:
        print("\n" + "="*60)
        print("RUNNING OVERLAPPING BENCHMARK")
        print("="*60)
        print(f"Algorithms: {algorithms}")
        
        run_overlapping_benchmark(
            str(networks_dir),
            str(results_dir),
            algorithms,
            include_gnn=not args.no_gnn,
            verbose=True
        )
    
    if args.all or args.plot:
        results_file = results_dir / 'overlapping_results.csv'
        if results_file.exists():
            print("\n" + "="*60)
            print("GENERATING PLOTS")
            print("="*60)
            plot_overlapping_results(str(results_file), str(results_dir))


if __name__ == '__main__':
    main()




