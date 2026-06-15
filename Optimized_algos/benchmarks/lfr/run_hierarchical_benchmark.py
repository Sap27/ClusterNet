#!/usr/bin/env python3
"""
LFR Hierarchical Benchmark
==========================

Test algorithm ability to detect hierarchical community structure.

Tests:
1. Detection of micro-communities (first level)
2. Detection of macro-communities (second level)
3. Ability to detect both levels

X-axis: μ1 (macro mixing) and μ2 (micro mixing) combinations
Y-axis: AMI at each level
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
from itertools import product

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent.parent))

from config import HIERARCHICAL_CONFIG, ALGORITHMS, HIERARCHICAL_CATEGORIES
from lfr_generator import LFRGenerator
from algorithm_runner import AlgorithmRunner
from metrics import compute_all_metrics, compute_ami
import networkx as nx

try:
    GNN_HIERARCHICAL = ALGORITHMS.get('gnn_hierarchical', [])
except Exception:
    GNN_HIERARCHICAL = []


def save_hierarchical_network(G, micro_comms, macro_comms, output_dir, name):
    """Save hierarchical network with both levels of communities."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    nx.write_edgelist(G, output_path / f'{name}_edges.txt', data=False)
    
    with open(output_path / f'{name}_micro.txt', 'w') as f:
        for i, comm in enumerate(micro_comms):
            f.write(f"{i}: {' '.join(map(str, comm))}\n")
    
    with open(output_path / f'{name}_macro.txt', 'w') as f:
        for i, comm in enumerate(macro_comms):
            f.write(f"{i}: {' '.join(map(str, comm))}\n")
    
    metadata = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'micro_communities': len(micro_comms),
        'macro_communities': len(macro_comms),
    }
    with open(output_path / f'{name}_meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def load_hierarchical_network(input_dir, name):
    """Load hierarchical network."""
    input_path = Path(input_dir)
    
    G = nx.read_edgelist(input_path / f'{name}_edges.txt', nodetype=int)
    
    micro_comms = []
    with open(input_path / f'{name}_micro.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                nodes = list(map(int, parts[1].split()))
                micro_comms.append(nodes)
    
    macro_comms = []
    with open(input_path / f'{name}_macro.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(': ')
            if len(parts) == 2:
                nodes = list(map(int, parts[1].split()))
                macro_comms.append(nodes)
    
    return G, micro_comms, macro_comms


def generate_hierarchical_networks(output_dir: str, config: dict, lfr_dir: str, verbose: bool = True):
    """Generate hierarchical LFR networks."""
    generator = LFRGenerator(lfr_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    networks_generated = []
    params = config['params']
    
    # Generate for each (μ1, μ2) combination
    for mu1, mu2 in product(config['mu1_values'], config['mu2_values']):
        if verbose:
            print(f"\nGenerating networks for μ1={mu1}, μ2={mu2}...")
        
        for realization in range(config['realizations']):
            name = f"mu1_{mu1:.1f}_mu2_{mu2:.1f}_r{realization}"
            seed = int(mu1 * 1000 + mu2 * 100) + realization
            
            try:
                G, micro_comms, macro_comms = generator.generate_hierarchical(
                    N=config['network_size'],
                    mu1=mu1,
                    mu2=mu2,
                    k=params['k'],
                    maxk=params['maxk'],
                    minc=params['minc'],
                    maxc=params['maxc'],
                    minC=params['minC'],
                    maxC=params['maxC'],
                    seed=seed
                )
                
                save_hierarchical_network(G, micro_comms, macro_comms, str(output_path), name)
                
                networks_generated.append({
                    'name': name,
                    'mu1': mu1,
                    'mu2': mu2,
                    'realization': realization,
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'micro_communities': len(micro_comms),
                    'macro_communities': len(macro_comms),
                })
                
                if verbose:
                    print(f"  ✓ {name}: {len(micro_comms)} micro, {len(macro_comms)} macro communities")
                    
            except Exception as e:
                print(f"  ✗ {name}: Failed - {e}")
    
    metadata = {
        'benchmark': 'hierarchical',
        'config': config,
        'networks': networks_generated,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return networks_generated


def run_hierarchical_benchmark(
    networks_dir: str,
    results_dir: str,
    algorithms: list,
    verbose: bool = True
):
    """Run classical hierarchical benchmark (no GNNs)."""
    networks_path = Path(networks_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    with open(networks_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    runner = AlgorithmRunner(verbose=False)
    all_results = []

    for network_info in metadata['networks']:
        name = network_info['name']
        mu1, mu2 = network_info['mu1'], network_info['mu2']

        if verbose:
            print(f"\nProcessing {name} (μ1={mu1}, μ2={mu2})...")

        G, micro_comms, macro_comms = load_hierarchical_network(str(networks_path), name)
        num_nodes = G.number_of_nodes()

        for algo_name in algorithms:
            if verbose:
                print(f"  Running {algo_name}...", end=' ', flush=True)

            result = runner.run_algorithm(G, algo_name)

            if result.success and result.communities:
                ami_micro = compute_ami(micro_comms, result.communities, num_nodes)
                ami_macro = compute_ami(macro_comms, result.communities, num_nodes)
                detected_level = 'micro' if ami_micro > ami_macro else 'macro'

                all_results.append({
                    'network': name, 'mu1': mu1, 'mu2': mu2,
                    'realization': network_info['realization'],
                    'algorithm': algo_name, 'algorithm_type': 'classical',
                    'success': True, 'runtime': result.runtime,
                    'ami_micro': ami_micro, 'ami_macro': ami_macro,
                    'detected_level': detected_level,
                    'num_detected': len(result.communities),
                    'num_micro': len(micro_comms),
                    'num_macro': len(macro_comms),
                })
                if verbose:
                    print(f"✓ micro={ami_micro:.3f}, macro={ami_macro:.3f}")
            else:
                all_results.append({
                    'network': name, 'mu1': mu1, 'mu2': mu2,
                    'realization': network_info['realization'],
                    'algorithm': algo_name, 'algorithm_type': 'classical',
                    'success': False, 'runtime': result.runtime,
                    'error': result.error_message
                })
                if verbose:
                    print("✗")

    df = pd.DataFrame(all_results)
    df.to_csv(results_path / 'hierarchical_results.csv', index=False)

    df_success = df[df['success'] == True]
    if len(df_success) > 0:
        summary = df_success.groupby(['mu1', 'mu2', 'algorithm']).agg({
            'ami_micro': ['mean', 'std'],
            'ami_macro': ['mean', 'std'],
            'runtime': 'mean',
        }).round(4)
        summary.to_csv(results_path / 'hierarchical_summary.csv')

    if verbose:
        print(f"\nClassical hierarchical results saved to: {results_path}")

    return df


def run_gnn_hierarchical_benchmark(
    networks_dir: str,
    results_dir: str,
    gnn_algorithms: list,
    verbose: bool = True,
    resume: bool = False,
    epochs: int = 200,
):
    """
    Run GNN algorithms on hierarchical networks at both micro and macro K.

    Each GNN is run twice per network: once targeting K=micro_communities,
    once targeting K=macro_communities. Both runs are scored against both
    ground truth levels.
    """
    networks_path = Path(networks_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    partitions_path = results_path / 'partitions' / 'gnn_hierarchical'
    partitions_path.mkdir(parents=True, exist_ok=True)

    csv_path = results_path / 'gnn_hierarchical_results.csv'

    with open(networks_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)

    existing_pairs = set()
    if resume and csv_path.is_file():
        try:
            df_existing = pd.read_csv(csv_path)
            for _, row in df_existing[df_existing['success'] == True].iterrows():
                existing_pairs.add((row['network'], row['algorithm']))
        except Exception:
            pass

    runner = AlgorithmRunner(verbose=False)
    all_results = []

    for network_info in metadata['networks']:
        name = network_info['name']
        mu1, mu2 = network_info['mu1'], network_info['mu2']
        realization = network_info['realization']

        if verbose:
            print(f"\nProcessing {name} (μ1={mu1}, μ2={mu2})...")

        G, micro_comms, macro_comms = load_hierarchical_network(str(networks_path), name)
        num_nodes = G.number_of_nodes()

        for algo_name in gnn_algorithms:
            for target_level, target_comms in [('micro', micro_comms), ('macro', macro_comms)]:
                full_name = f"{algo_name}_{target_level}"
                num_target = len(target_comms)

                if resume and (name, full_name) in existing_pairs:
                    if verbose:
                        print(f"  {full_name}... (skipped, cached) ✓")
                    continue

                if verbose:
                    print(f"  Running {full_name} (K={num_target})...", end=' ', flush=True)

                result = runner.run_algorithm(
                    G, algo_name,
                    num_clusters=num_target, epochs=epochs,
                )

                if result.success and result.communities:
                    ami_micro = compute_ami(micro_comms, result.communities, num_nodes)
                    ami_macro = compute_ami(macro_comms, result.communities, num_nodes)

                    all_results.append({
                        'network': name, 'mu1': mu1, 'mu2': mu2,
                        'realization': realization,
                        'algorithm': full_name, 'base_algorithm': algo_name,
                        'algorithm_type': 'gnn', 'target_level': target_level,
                        'success': True, 'runtime': result.runtime,
                        'ami_micro': ami_micro, 'ami_macro': ami_macro,
                        'detected_level': 'micro' if ami_micro > ami_macro else 'macro',
                        'num_detected': len(result.communities),
                        'num_micro': len(micro_comms),
                        'num_macro': len(macro_comms),
                    })
                    if verbose:
                        print(f"✓ micro={ami_micro:.3f}, macro={ami_macro:.3f}")
                else:
                    all_results.append({
                        'network': name, 'mu1': mu1, 'mu2': mu2,
                        'realization': realization,
                        'algorithm': full_name, 'base_algorithm': algo_name,
                        'algorithm_type': 'gnn', 'target_level': target_level,
                        'success': False, 'runtime': result.runtime,
                        'error': result.error_message
                    })
                    if verbose:
                        err = result.error_message or ''
                        print(f"✗ {err[:60]}")

    if resume and csv_path.is_file():
        try:
            df_old = pd.read_csv(csv_path)
            all_results = pd.concat([df_old, pd.DataFrame(all_results)],
                                    ignore_index=True).drop_duplicates(
                                        subset=['network', 'algorithm'], keep='last')
            all_results = all_results.to_dict('records')
        except Exception:
            pass

    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)

    if len(df) > 0 and 'ami_micro' in df.columns:
        df_ok = df[df['success'] == True]
        if len(df_ok) > 0:
            summary = df_ok.groupby(['mu1', 'mu2', 'algorithm']).agg({
                'ami_micro': ['mean', 'std'],
                'ami_macro': ['mean', 'std'],
                'runtime': 'mean',
            }).round(4)
            summary.to_csv(results_path / 'gnn_hierarchical_summary.csv')

    if verbose:
        print(f"\nGNN hierarchical results saved to: {csv_path}")

    return df


def plot_hierarchical_results(results_file: str, output_dir: str):
    """Generate plots for hierarchical benchmark."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Matplotlib/Seaborn not available for plotting")
        return
    
    df = pd.read_csv(results_file)
    df_success = df[df['success'] == True]
    
    output_path = Path(output_dir)
    
    # Only classical algorithms for cleaner plot
    df_classical = df_success[df_success['algorithm_type'] == 'classical']
    
    # Plot: Micro vs Macro AMI scatter for each algorithm
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    algorithms = df_classical['algorithm'].unique()[:4]  # Top 4
    
    for idx, algo in enumerate(algorithms):
        ax = axes[idx // 2, idx % 2]
        algo_data = df_classical[df_classical['algorithm'] == algo]
        
        scatter = ax.scatter(
            algo_data['ami_micro'],
            algo_data['ami_macro'],
            c=algo_data['mu1'],
            cmap='viridis',
            alpha=0.7
        )
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Equal detection')
        ax.set_xlabel('AMI (Micro-communities)')
        ax.set_ylabel('AMI (Macro-communities)')
        ax.set_title(f'{algo}')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.colorbar(scatter, ax=ax, label='μ1')
    
    plt.suptitle('Hierarchical Detection: Micro vs Macro AMI', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / 'hierarchical_scatter.png', dpi=150)
    plt.close()
    
    # Heatmap: Mean AMI by (mu1, mu2)
    for level in ['micro', 'macro']:
        pivot = df_classical.pivot_table(
            values=f'ami_{level}',
            index='algorithm',
            columns=['mu1', 'mu2'],
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
        plt.title(f'Mean AMI ({level.capitalize()}-communities) by Algorithm and (μ1, μ2)')
        plt.tight_layout()
        plt.savefig(output_path / f'hierarchical_heatmap_{level}.png', dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='LFR Hierarchical Benchmark')
    parser.add_argument('--generate', action='store_true', help='Generate networks')
    parser.add_argument('--run', action='store_true', help='Run classical benchmark')
    parser.add_argument('--run-gnn', action='store_true', help='Run GNN benchmark')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--all', action='store_true', help='Generate, run (classical), and plot')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--realizations', type=int, default=None)
    parser.add_argument('--resume', action='store_true', help='Resume GNN benchmark (skip completed)')
    parser.add_argument('--epochs', type=int, default=200, help='GNN training epochs (default: 200)')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    lfr_dir = script_dir.parent.parent.parent.parent / 'LFRbenchmarks'

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / 'hierarchical'

    networks_dir = output_dir / 'networks'
    results_dir = output_dir / 'results'

    config = HIERARCHICAL_CONFIG.copy()
    if args.realizations:
        config['realizations'] = args.realizations

    algorithms = []
    for category in HIERARCHICAL_CATEGORIES:
        algorithms.extend(ALGORITHMS.get(category, []))

    if args.all or args.generate:
        print("=" * 60)
        print("GENERATING HIERARCHICAL LFR NETWORKS")
        print("=" * 60)
        print(f"μ1 values: {config['mu1_values']}")
        print(f"μ2 values: {config['mu2_values']}")
        print(f"Realizations: {config['realizations']}")

        generate_hierarchical_networks(
            str(networks_dir), config, str(lfr_dir), verbose=True
        )

    if args.all or args.run:
        print("\n" + "=" * 60)
        print("RUNNING HIERARCHICAL BENCHMARK (CLASSICAL)")
        print("=" * 60)

        run_hierarchical_benchmark(
            str(networks_dir), str(results_dir), algorithms, verbose=True
        )

    if args.run_gnn:
        print("\n" + "=" * 60)
        print("RUNNING HIERARCHICAL BENCHMARK (GNN)")
        print("=" * 60)

        gnn_algos = GNN_HIERARCHICAL
        if not gnn_algos:
            print("No GNN algorithms configured in config.py (gnn_hierarchical)")
        else:
            print(f"GNNs: {gnn_algos}")
            print(f"Epochs: {args.epochs}, Resume: {args.resume}")
            run_gnn_hierarchical_benchmark(
                str(networks_dir), str(results_dir), gnn_algos,
                verbose=True, resume=args.resume, epochs=args.epochs,
            )

    if args.all or args.plot:
        results_file = results_dir / 'hierarchical_results.csv'
        if results_file.exists():
            print("\n" + "=" * 60)
            print("GENERATING PLOTS")
            print("=" * 60)
            plot_hierarchical_results(str(results_file), str(results_dir))


if __name__ == '__main__':
    main()




