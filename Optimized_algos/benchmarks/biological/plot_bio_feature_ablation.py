"""
Post-processing: Biological Feature Ablation Analysis.

Reads bio_gnn_ablation_results.csv and generates:
  1. Grouped bar chart comparing real vs structural vs random features
  2. Summary statistics table (LaTeX format)
  3. Per-network heatmap of modularity by algorithm and feature type

Usage:
    python plot_bio_feature_ablation.py [--results-dir PATH] [--output-dir PATH]
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

NETWORK_DISPLAY = {
    'signor': 'SIGNOR',
    'grn': 'GRN',
    'biogrid': 'BioGRID',
    'gtex_0.9': 'GTEx',
}
ALGO_DISPLAY = {
    'dmon_gnn': 'DMoN',
    'mincut_gnn': 'MinCut',
    'vgae_gnn': 'VGAE',
    'dgi_gnn': 'DGI',
    'mlp_kmeans': 'MLP+KMeans',
    'node2vec_gnn': 'Node2Vec',
    'gcn_supervised': 'GCN (sup)',
    'gat_supervised': 'GAT (sup)',
}
FEATURE_COLORS = {
    'real': '#2196F3',
    'structural': '#FF9800',
    'random': '#9E9E9E',
}


def load_ablation_results(results_dir):
    """Load ablation results CSV."""
    csv_path = results_dir / 'bio_gnn_ablation_results.csv'
    if not csv_path.is_file():
        raise FileNotFoundError(f"No ablation results at {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df['success'] == True].copy()
    return df


def plot_modularity_comparison(df, output_dir):
    """Grouped bar chart: modularity by feature type for each algorithm, per network."""
    networks = [n for n in ['signor', 'grn', 'biogrid', 'gtex_0.9'] if n in df['network'].unique()]
    fig, axes = plt.subplots(1, len(networks), figsize=(4*len(networks), 5), sharey=False)
    if len(networks) == 1:
        axes = [axes]

    for ax, net_id in zip(axes, networks):
        net_df = df[df['network'] == net_id]
        pivot = net_df.groupby(['algorithm', 'feature_type'])['modularity'].mean().unstack()

        algos = [a for a in ALGO_DISPLAY.keys() if a in pivot.index]
        feat_types = [f for f in ['real', 'structural', 'random'] if f in pivot.columns]

        x = np.arange(len(algos))
        width = 0.25
        offsets = np.linspace(-width, width, len(feat_types))

        for i, ft in enumerate(feat_types):
            vals = [pivot.loc[a, ft] if a in pivot.index and ft in pivot.columns else 0
                    for a in algos]
            ax.bar(x + offsets[i], vals, width, label=ft.capitalize(),
                   color=FEATURE_COLORS[ft], edgecolor='white', linewidth=0.5)

        ax.set_xlabel('')
        ax.set_ylabel('Modularity' if ax == axes[0] else '')
        ax.set_title(NETWORK_DISPLAY.get(net_id, net_id), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([ALGO_DISPLAY.get(a, a) for a in algos],
                           rotation=45, ha='right', fontsize=8)
        ax.axhline(0, color='black', linewidth=0.5)

    axes[-1].legend(loc='upper right', fontsize=9)
    plt.tight_layout()
    out_path = output_dir / 'bio_feature_ablation_modularity.pdf'
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_relative_performance(df, output_dir):
    """Heatmap of relative performance: (real - random) / |random| per network/algo."""
    networks = [n for n in ['signor', 'grn', 'biogrid', 'gtex_0.9'] if n in df['network'].unique()]
    algos = [a for a in ALGO_DISPLAY.keys() if a in df['algorithm'].unique()]

    # Compute mean modularity per (network, algorithm, feature_type)
    pivot = df.groupby(['network', 'algorithm', 'feature_type'])['modularity'].mean().reset_index()

    # Build improvement matrix
    improvement = np.full((len(algos), len(networks)), np.nan)
    for j, net in enumerate(networks):
        for i, algo in enumerate(algos):
            real_val = pivot[(pivot['network']==net) & (pivot['algorithm']==algo) &
                            (pivot['feature_type']=='real')]['modularity'].values
            rand_val = pivot[(pivot['network']==net) & (pivot['algorithm']==algo) &
                            (pivot['feature_type']=='random')]['modularity'].values
            if len(real_val) > 0 and len(rand_val) > 0:
                denom = max(abs(rand_val[0]), 0.01)
                improvement[i, j] = (real_val[0] - rand_val[0]) / denom

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(improvement, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=2)
    ax.set_xticks(range(len(networks)))
    ax.set_xticklabels([NETWORK_DISPLAY.get(n, n) for n in networks])
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([ALGO_DISPLAY.get(a, a) for a in algos])
    ax.set_title('Feature Benefit: (Real - Random) / |Random|', fontsize=11)

    for i in range(len(algos)):
        for j in range(len(networks)):
            val = improvement[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(val) > 1 else 'black')

    plt.colorbar(im, ax=ax, label='Relative improvement')
    plt.tight_layout()
    out_path = output_dir / 'bio_feature_ablation_heatmap.pdf'
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def generate_latex_table(df, output_dir):
    """Generate LaTeX table of mean modularity by algorithm, network, feature type."""
    networks = [n for n in ['signor', 'grn', 'biogrid', 'gtex_0.9'] if n in df['network'].unique()]
    algos = [a for a in ALGO_DISPLAY.keys() if a in df['algorithm'].unique()]

    pivot = df.groupby(['network', 'algorithm', 'feature_type'])['modularity'].agg(['mean', 'std']).reset_index()

    lines = []
    lines.append(r'\begin{table}[htbp]')
    lines.append(r'\centering')
    lines.append(r'\caption{Biological feature ablation: mean modularity ($\pm$ std) across seeds.}')
    lines.append(r'\label{tab:bio_ablation}')
    lines.append(r'\small')

    col_spec = 'l' + 'c' * (len(networks) * 3)
    lines.append(r'\begin{tabular}{' + col_spec + '}')
    lines.append(r'\toprule')

    header = 'Algorithm'
    for net in networks:
        header += f' & \\multicolumn{{3}}{{c}}{{{NETWORK_DISPLAY.get(net, net)}}}'
    header += r' \\'
    lines.append(header)

    subheader = ''
    for _ in networks:
        subheader += ' & Real & Struct. & Random'
    subheader += r' \\'
    lines.append(subheader)
    lines.append(r'\midrule')

    for algo in algos:
        row = ALGO_DISPLAY.get(algo, algo)
        for net in networks:
            for ft in ['real', 'structural', 'random']:
                mask = ((pivot['network'] == net) & (pivot['algorithm'] == algo) &
                        (pivot['feature_type'] == ft))
                vals = pivot[mask]
                if len(vals) > 0:
                    m = vals['mean'].values[0]
                    s = vals['std'].values[0]
                    row += f' & {m:.3f}'
                else:
                    row += ' & --'
        row += r' \\'
        lines.append(row)

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    tex_path = output_dir / 'bio_ablation_table.tex'
    with open(str(tex_path), 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {tex_path}")


def print_insights(df):
    """Print key findings from the ablation."""
    print("\n" + "="*60)
    print("KEY INSIGHTS FROM BIOLOGICAL FEATURE ABLATION")
    print("="*60)

    pivot = df.groupby(['network', 'algorithm', 'feature_type'])['modularity'].mean().reset_index()

    for net in df['network'].unique():
        net_data = pivot[pivot['network'] == net]
        real_mean = net_data[net_data['feature_type'] == 'real']['modularity'].mean()
        struct_mean = net_data[net_data['feature_type'] == 'structural']['modularity'].mean()
        rand_mean = net_data[net_data['feature_type'] == 'random']['modularity'].mean()

        print(f"\n  {NETWORK_DISPLAY.get(net, net)}:")
        print(f"    Real features mean Q:       {real_mean:.4f}")
        print(f"    Structural features mean Q: {struct_mean:.4f}")
        print(f"    Random features mean Q:     {rand_mean:.4f}")

        if abs(rand_mean) > 0.001:
            benefit = (real_mean - rand_mean) / abs(rand_mean) * 100
            print(f"    Feature benefit (real vs random): {benefit:+.1f}%")
        if real_mean > struct_mean:
            print(f"    -> Biological features outperform structural")
        else:
            print(f"    -> Structural features sufficient (topology dominates)")


def main():
    parser = argparse.ArgumentParser(description='Bio Feature Ablation Analysis')
    parser.add_argument('--results-dir', type=str,
                        default='/mnt/d/cluster/BIologicalNetworks/results',
                        help='Path to results directory')
    parser.add_argument('--output-dir', type=str,
                        default='/mnt/d/cluster/ClusterNet/paper/figures',
                        help='Path to output figures directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ablation results...", flush=True)
    df = load_ablation_results(results_dir)
    print(f"  {len(df)} successful runs loaded", flush=True)

    print("\nGenerating plots...", flush=True)
    plot_modularity_comparison(df, output_dir)
    plot_relative_performance(df, output_dir)

    print("\nGenerating LaTeX table...", flush=True)
    generate_latex_table(df, output_dir)

    print_insights(df)


if __name__ == '__main__':
    main()
