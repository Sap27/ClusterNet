"""Post-processing: Perturbation Analysis Summary and Figures."""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

CLASSICAL = ['leiden', 'louvain', 'rb_pots', 'fastgreedy', 'agdl']
GNN_ALGOS = ['dmon_gnn', 'mincut_gnn', 'dgi_gnn', 'gcn_supervised',
             'mlp_kmeans', 'vgae_gnn', 'node2vec_gnn', 'gat_supervised']

ALGO_DISPLAY = {
    'leiden': 'Leiden', 'louvain': 'Louvain', 'rb_pots': 'RB-Pots',
    'fastgreedy': 'FastGreedy', 'agdl': 'AGDL',
    'dmon_gnn': 'DMoN', 'mincut_gnn': 'MinCut', 'dgi_gnn': 'DGI',
    'gcn_supervised': 'GCN', 'gat_supervised': 'GAT',
    'mlp_kmeans': 'MLP+KM', 'vgae_gnn': 'VGAE', 'node2vec_gnn': 'Node2Vec',
}


def algo_type(a):
    return 'Classical' if a in CLASSICAL else 'GNN'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='/mnt/d/results-5/perturbation')
    parser.add_argument('--output-dir', type=str, default='/mnt/d/cluster/ClusterNet/paper/figures')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / 'grn_perturbation_results.csv')
    df['algo_type'] = df['algorithm'].apply(algo_type)

    print(f"Total rows: {len(df)}")
    print(f"Algorithms ({len(df['algorithm'].unique())}): {sorted(df['algorithm'].unique())}")
    print(f"Tests ({len(df['test'].unique())}): {sorted(df['test'].unique())}")

    # 1. Overall stability ranking
    print("\n" + "="*80)
    print("OVERALL STABILITY RANKING (mean NMI across ALL perturbations)")
    print("="*80)
    overall = df.groupby('algorithm').agg({
        'nmi': 'mean', 'ami': 'mean', 'jaccard': 'mean',
        'ffl_change': 'mean', 'modularity_change': 'mean',
    }).round(4)
    overall['type'] = overall.index.map(algo_type)
    overall = overall.sort_values('nmi', ascending=False)
    print(overall.to_string())

    # 2. Classical vs GNN
    print("\n" + "="*80)
    print("CLASSICAL vs GNN (mean across all perturbations)")
    print("="*80)
    type_comp = df.groupby('algo_type').agg(
        nmi_mean=('nmi', 'mean'),
        nmi_std=('nmi', 'std'),
        ffl_mean=('ffl_change', 'mean'),
        ffl_std=('ffl_change', 'std'),
    ).round(4)
    print(type_comp.to_string())

    # 3. TF knockout vs random control
    print("\n" + "="*80)
    print("TF KNOCKOUT vs RANDOM CONTROL")
    print("="*80)
    for k in [5, 10]:
        tf_test = f'tf_knockout_top{k}'
        ctrl_test = f'tf_knockout_control_random{k}'
        tf_nmi = df[df['test'] == tf_test].groupby('algorithm')['nmi'].mean()
        ctrl_nmi = df[df['test'] == ctrl_test].groupby('algorithm')['nmi'].mean()
        comp = pd.DataFrame({
            'NMI_TF_knockout': tf_nmi,
            'NMI_random_ctrl': ctrl_nmi,
            'gap': ctrl_nmi - tf_nmi
        }).round(4).sort_values('gap', ascending=False)
        print(f"\n  Top-{k} TFs removed vs {k} random genes:")
        print(comp.to_string())

    # 4. Edge dropout degradation
    print("\n" + "="*80)
    print("EDGE DROPOUT STABILITY (NMI by level, Classical vs GNN)")
    print("="*80)
    ed = df[df['test'].str.startswith('edge_dropout')]
    ed_comp = ed.groupby(['level', 'algo_type'])['nmi'].mean().round(4).unstack()
    print(ed_comp.to_string())

    # 5. FFL preservation
    print("\n" + "="*80)
    print("FFL PRESERVATION (excluding hub removal)")
    print("="*80)
    mild = df[~df['test'].str.startswith('gene_removal_hub')]
    ffl = mild.groupby('algorithm').agg(
        ffl_preserved=('ffl_preserved_pert', 'mean'),
        ffl_change=('ffl_change', 'mean'),
    ).round(4)
    ffl['type'] = ffl.index.map(algo_type)
    ffl = ffl.sort_values('ffl_preserved', ascending=False)
    print(ffl.to_string())

    # =========== FIGURES ===========

    # Figure 1: Edge dropout stability curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ed_algo = ed.groupby(['level', 'algorithm'])['nmi'].mean().reset_index()
    for algo in sorted(ed_algo['algorithm'].unique()):
        adf = ed_algo[ed_algo['algorithm'] == algo].sort_values('level')
        style = '-' if algo in CLASSICAL else '--'
        color = None
        lw = 1.5 if algo in CLASSICAL else 2.0
        ax.plot(adf['level'].astype(str), adf['nmi'],
                linestyle=style, linewidth=lw,
                label=ALGO_DISPLAY.get(algo, algo), marker='o', markersize=4)
    ax.set_xlabel('Edge Dropout Rate')
    ax.set_ylabel('NMI (stability)')
    ax.set_title('GRN: Community Stability Under Edge Dropout')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'perturbation_edge_dropout.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'perturbation_edge_dropout.pdf'}")

    # Figure 2: TF knockout comparison (grouped bar)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    algos_sorted = overall.index.tolist()
    tf5 = df[df['test'] == 'tf_knockout_top5'].groupby('algorithm')['nmi'].mean()
    ctrl5 = df[df['test'] == 'tf_knockout_control_random5'].groupby('algorithm')['nmi'].mean()
    algos_plot = [a for a in algos_sorted if a in tf5.index and a in ctrl5.index]
    x = np.arange(len(algos_plot))
    w = 0.35
    ax.bar(x - w/2, [tf5[a] for a in algos_plot], w, label='TF Knockout (top-5)',
           color='#F44336', alpha=0.8)
    ax.bar(x + w/2, [ctrl5[a] for a in algos_plot], w, label='Random Control (5 genes)',
           color='#4CAF50', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([ALGO_DISPLAY.get(a, a) for a in algos_plot], rotation=45, ha='right')
    ax.set_ylabel('NMI (stability)')
    ax.set_title('GRN: TF Knockout Impact on Community Structure')
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(output_dir / 'perturbation_tf_knockout.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'perturbation_tf_knockout.pdf'}")

    # Figure 3: Overall stability ranking (horizontal bar)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#2196F3' if algo_type(a) == 'Classical' else '#FF9800'
              for a in overall.index]
    ax.barh(range(len(overall)), overall['nmi'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(overall)))
    ax.set_yticklabels([ALGO_DISPLAY.get(a, a) for a in overall.index])
    ax.set_xlabel('Mean NMI (stability across all perturbations)')
    ax.set_title('GRN: Overall Perturbation Stability Ranking')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#2196F3', label='Classical'),
                       Patch(color='#FF9800', label='GNN')], loc='lower right')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'perturbation_stability_ranking.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'perturbation_stability_ranking.pdf'}")


if __name__ == '__main__':
    main()
