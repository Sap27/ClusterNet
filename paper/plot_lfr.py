"""
Generate all LFR benchmark figures for the paper.

Data sources:
  GPU accuracy classical:  D:/results-clusternet-classical-gpu-new/results/classical_accuracy_results.csv
  GPU accuracy GNN:        D:/results-clusternet-classical-gpu-3/results/gnn_accuracy_results.csv
  GPU scalability classical: D:/results-clusternet-classical-sca-gpu-new4/results/classical_accuracy_results.csv
  GPU scalability GNN:     D:/results-clusternet-classical-sca-gpu-3/results-clusternet-classical-sca-gpu-3/results/gnn_scalability_results.csv
  Hierarchical classical: GPU CSV + local Nested SBM AMI overlay (lfr_io.load_merged_hierarchical_classical)
  Hierarchical GNN:       D:/results-clusternet-classical-hierarchical-gnn-gpu/.../gnn_hierarchical_results.csv
  Local accuracy classical: D:/cluster/ClusterNet/Optimized_algos/benchmarks/lfr/accuracy/results/accuracy_results.csv
  Local scalability classical: D:/cluster/ClusterNet/Optimized_algos/benchmarks/lfr/scalability/results/scalability_results.csv
  Local GNN features:      D:/cluster/ClusterNet/Optimized_algos/benchmarks/lfr/accuracy/results/gnn_features_results.csv
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings('ignore')

OUT = pathlib.Path(__file__).parent / 'figures'
OUT.mkdir(exist_ok=True)

# --------------- path helper ---------------
import platform

from lfr_io import (
    load_gpu_accuracy_gnn_full,
    load_gpu_hierarchical_gnn,
    load_merged_accuracy_classical,
    load_merged_accuracy_gnn,
    load_merged_hierarchical_classical,
    load_merged_scalability_classical,
    load_merged_scalability_gnn,
)


def _p(win_path: str) -> pathlib.Path:
    """Convert D:/... to /mnt/d/... when running under WSL/Linux."""
    if platform.system() != 'Windows' and win_path[1:3] == ':/':
        drive = win_path[0].lower()
        return pathlib.Path(f'/mnt/{drive}/{win_path[3:]}')
    return pathlib.Path(win_path)

# --------------- paths (non-LFR merged loaders only) ---------------
LOC_GNN_FEAT= _p('D:/cluster/ClusterNet/Optimized_algos/benchmarks/lfr/accuracy/results/gnn_features_results.csv')

# --------------- style ---------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 8,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

ALGO_DISPLAY = {
    'louvain': 'Louvain', 'leiden': 'Leiden', 'fastgreedy': 'FastGreedy',
    'label_propagation': 'Label Prop.', 'walktrap': 'Walktrap',
    'spinglass': 'Spinglass', 'spectral': 'Spectral',
    'leading_eigen': 'Leading Eigen.', 'sbm': 'SBM', 'nested_sbm': 'Nested SBM',
    'rb_pots': 'RB-Pots', 'rber_pots': 'RBer-Pots', 'surprise': 'Surprise',
    'em': 'EM', 'scan': 'SCAN', 'async_fluid': 'Async Fluid',
    'cpm': 'CPM', 'angel': 'Angel', 'demon': 'Demon',
    'kclique': 'k-Clique', 'score': 'SCORE', 'teamcs': 'TeamCS',
    'bigs2': 'BigS2', 'csbio_iitm2': 'CSBIO-IITM2', 'agdl': 'AGDL',
    'der': 'DER', 'gdmp2': 'GDMP2', 'simnet': 'SimNet',
    'svt': 'SVT', 'tusk': 'Tusk', 'infomap': 'Infomap',
    'dmon_gnn': 'DMoN', 'mincut_gnn': 'MinCutPool',
    'vgae_gnn': 'VGAE+KM', 'dgi_gnn': 'DGI+KM',
    'gcn_supervised': 'GCN (20%)', 'gat_supervised': 'GAT (20%)',
    'gcn_supervised_40': 'GCN (40%)', 'gat_supervised_40': 'GAT (40%)',
    'gcn_supervised_60': 'GCN (60%)', 'gat_supervised_60': 'GAT (60%)',
    'gcn_supervised_80': 'GCN (80%)', 'gat_supervised_80': 'GAT (80%)',
    'mlp_kmeans': 'MLP+KM', 'node2vec_gnn': 'Node2Vec+KM',
}

# tier-1 classical algos to highlight (top performers + important baselines)
TOP_CLASSICAL = [
    'leiden', 'louvain', 'walktrap', 'fastgreedy', 'label_propagation',
    'spinglass', 'rb_pots', 'surprise', 'sbm', 'nested_sbm',
    'leading_eigen', 'spectral', 'em', 'scan',
]

CLASSICAL_COLORS = {
    'leiden': '#1f77b4', 'louvain': '#ff7f0e', 'walktrap': '#2ca02c',
    'fastgreedy': '#d62728', 'label_propagation': '#9467bd',
    'spinglass': '#8c564b', 'rb_pots': '#e377c2', 'surprise': '#7f7f7f',
    'sbm': '#bcbd22', 'nested_sbm': '#17becf',
    'leading_eigen': '#aec7e8', 'spectral': '#ffbb78',
    'em': '#98df8a', 'scan': '#ff9896',
}

GNN_BASE = ['dmon_gnn', 'mincut_gnn', 'vgae_gnn', 'dgi_gnn',
            'gcn_supervised', 'gat_supervised', 'mlp_kmeans', 'node2vec_gnn']

GNN_COLORS = {
    'dmon_gnn': '#1f77b4', 'mincut_gnn': '#ff7f0e',
    'vgae_gnn': '#2ca02c', 'dgi_gnn': '#d62728',
    'gcn_supervised': '#9467bd', 'gat_supervised': '#8c564b',
    'mlp_kmeans': '#e377c2', 'node2vec_gnn': '#7f7f7f',
}

GNN_MARKERS = {
    'dmon_gnn': 'o', 'mincut_gnn': 's', 'vgae_gnn': '^', 'dgi_gnn': 'D',
    'gcn_supervised': 'v', 'gat_supervised': 'P', 'mlp_kmeans': 'X', 'node2vec_gnn': '*',
}


def load_accuracy_classical():
    """Merged GPU + local supplement; SBM/Nested SBM AMI overlaid from local when mu matches (see lfr_io)."""
    df, _ = load_merged_accuracy_classical()
    print("  Classical accuracy: GPU rows + local supplement; SBM/Nested SBM AMI from local when mu matches.")
    return df


def load_accuracy_gnn():
    """Load GPU GNN accuracy results (base models only)."""
    return load_merged_accuracy_gnn()


def load_scalability_classical():
    """Merged GPU scalability + supplement; SBM/Nested SBM AMI overlaid from local when N matches."""
    df, _ = load_merged_scalability_classical()
    print("  Classical scalability: GPU rows + local supplement; SBM/Nested SBM AMI from local when N matches.")
    return df


def load_scalability_gnn():
    """Load GPU GNN scalability results."""
    return load_merged_scalability_gnn()


def load_hierarchical_classical():
    """Nested SBM / nested_sbm_coarse AMI from local hierarchical CSV; runtime from GPU (see lfr_io)."""
    merged, _ = load_merged_hierarchical_classical()
    return merged


def load_hierarchical_gnn():
    return load_gpu_hierarchical_gnn()


def mean_ci(group, col='ami'):
    """Mean and 95% CI from realizations."""
    vals = group[col].dropna()
    m = vals.mean()
    if len(vals) > 1:
        se = vals.std() / np.sqrt(len(vals))
    else:
        se = 0
    return m, 1.96 * se


# =====================================================================
# FIGURE 1: Accuracy vs mu  (2-panel: classical | GNN)
# =====================================================================
def fig1_accuracy():
    print("Generating Figure 1: Accuracy vs mu ...")
    cl = load_accuracy_classical()
    gn = load_accuracy_gnn()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # ---- Panel A: classical ----
    for algo in TOP_CLASSICAL:
        sub = cl[cl['algorithm'] == algo]
        if sub.empty:
            continue
        grp = sub.groupby('mu').apply(lambda g: pd.Series({
            'ami_mean': g['ami'].mean(),
            'ami_se': 1.96 * g['ami'].std() / max(np.sqrt(len(g)), 1)
        })).reset_index()
        grp = grp.sort_values('mu')
        color = CLASSICAL_COLORS.get(algo, '#333333')
        ax1.plot(grp['mu'], grp['ami_mean'], color=color,
                 label=ALGO_DISPLAY.get(algo, algo), marker='.', markersize=3)
        ax1.fill_between(grp['mu'],
                         grp['ami_mean'] - grp['ami_se'],
                         grp['ami_mean'] + grp['ami_se'],
                         alpha=0.12, color=color)

    ax1.set_xlabel(r'Mixing parameter $\mu$')
    ax1.set_ylabel('AMI')
    ax1.set_title('(a) Classical Algorithms')
    ax1.legend(ncol=2, loc='lower left', fontsize=7, framealpha=0.9)
    ax1.set_xlim(0.08, 0.82)
    ax1.set_ylim(-0.05, 1.05)
    ax1.axhline(0, color='grey', lw=0.5, ls='--')
    ax1.grid(True, alpha=0.3)

    # ---- Panel B: GNN ----
    for algo in GNN_BASE:
        sub = gn[gn['algorithm'] == algo]
        if sub.empty:
            continue
        grp = sub.groupby('mu').apply(lambda g: pd.Series({
            'ami_mean': g['ami'].mean(),
            'ami_se': 1.96 * g['ami'].std() / max(np.sqrt(len(g)), 1)
        })).reset_index()
        grp = grp.sort_values('mu')
        color = GNN_COLORS.get(algo, '#333333')
        marker = GNN_MARKERS.get(algo, 'o')
        ax2.plot(grp['mu'], grp['ami_mean'], color=color,
                 marker=marker, markersize=4,
                 label=ALGO_DISPLAY.get(algo, algo))
        ax2.fill_between(grp['mu'],
                         grp['ami_mean'] - grp['ami_se'],
                         grp['ami_mean'] + grp['ami_se'],
                         alpha=0.12, color=color)

    ax2.set_xlabel(r'Mixing parameter $\mu$')
    ax2.set_title('(b) GNN Algorithms')
    ax2.legend(ncol=2, loc='lower left', fontsize=7, framealpha=0.9)
    ax2.set_xlim(0.08, 0.82)
    ax2.axhline(0, color='grey', lw=0.5, ls='--')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / 'fig1_lfr_accuracy.pdf')
    fig.savefig(OUT / 'fig1_lfr_accuracy.png')
    plt.close(fig)
    print("  -> saved fig1_lfr_accuracy.pdf/.png")

    # ---- Print summary stats for the text ----
    print("\n  === Accuracy summary stats ===")
    for mu_val in [0.1, 0.3, 0.5, 0.7]:
        cl_sub = cl[np.isclose(cl['mu'], mu_val, atol=0.005)]
        if cl_sub.empty:
            continue
        best = cl_sub.groupby('algorithm')['ami'].mean().sort_values(ascending=False)
        print(f"  mu={mu_val:.1f}  top-5 classical: ", end='')
        for a, v in best.head(5).items():
            print(f"{ALGO_DISPLAY.get(a,a)}={v:.3f}  ", end='')
        print()
    for mu_val in [0.1, 0.3, 0.5, 0.7]:
        gn_sub = gn[np.isclose(gn['mu'], mu_val, atol=0.005)]
        if gn_sub.empty:
            continue
        best = gn_sub.groupby('algorithm')['ami'].mean().sort_values(ascending=False)
        print(f"  mu={mu_val:.1f}  top-5 GNN: ", end='')
        for a, v in best.head(5).items():
            print(f"{ALGO_DISPLAY.get(a,a)}={v:.3f}  ", end='')
        print()


# =====================================================================
# FIGURE 2: Scalability (2-panel: runtime | AMI)
# =====================================================================
def fig2_scalability():
    print("\nGenerating Figure 2: Scalability ...")
    cl = load_scalability_classical()
    gn = load_scalability_gnn()

    sca_classical = ['leiden', 'louvain', 'fastgreedy', 'label_propagation',
                     'walktrap', 'spinglass', 'sbm', 'nested_sbm',
                     'leading_eigen', 'spectral', 'surprise']
    sca_gnn = ['dmon_gnn', 'mincut_gnn', 'vgae_gnn', 'dgi_gnn',
               'gcn_supervised', 'node2vec_gnn']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Panel A: Runtime vs N (log-log) ----
    for algo in sca_classical:
        sub = cl[(cl['algorithm'] == algo) & (cl['success'] == True)]
        if sub.empty:
            continue
        grp = sub.groupby('N')['runtime'].mean().reset_index().sort_values('N')
        color = CLASSICAL_COLORS.get(algo, '#333333')
        ax1.plot(grp['N'], grp['runtime'], color=color,
                 label=ALGO_DISPLAY.get(algo, algo), marker='.', markersize=5)

    for algo in sca_gnn:
        sub = gn[(gn['algorithm'] == algo) & (gn['success'] == True)]
        if sub.empty:
            continue
        grp = sub.groupby('N')['runtime'].mean().reset_index().sort_values('N')
        color = GNN_COLORS.get(algo, '#333333')
        marker = GNN_MARKERS.get(algo, 'o')
        ax1.plot(grp['N'], grp['runtime'], color=color, ls='--',
                 label=ALGO_DISPLAY.get(algo, algo), marker=marker, markersize=5)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Network size $N$')
    ax1.set_ylabel('Runtime (s)')
    ax1.set_title('(a) Runtime Scaling')
    ax1.legend(ncol=2, fontsize=6.5, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, which='both')

    # ---- Panel B: AMI vs N ----
    for algo in sca_classical:
        sub = cl[(cl['algorithm'] == algo) & (cl['success'] == True)]
        if sub.empty:
            continue
        grp = sub.groupby('N')['ami'].mean().reset_index().sort_values('N')
        color = CLASSICAL_COLORS.get(algo, '#333333')
        ax2.plot(grp['N'], grp['ami'], color=color,
                 label=ALGO_DISPLAY.get(algo, algo), marker='.', markersize=5)

    for algo in sca_gnn:
        sub = gn[(gn['algorithm'] == algo) & (gn['success'] == True)]
        if sub.empty:
            continue
        grp = sub.groupby('N')['ami'].mean().reset_index().sort_values('N')
        color = GNN_COLORS.get(algo, '#333333')
        marker = GNN_MARKERS.get(algo, 'o')
        ax2.plot(grp['N'], grp['ami'], color=color, ls='--',
                 label=ALGO_DISPLAY.get(algo, algo), marker=marker, markersize=5)

    ax2.set_xscale('log')
    ax2.set_xlabel('Network size $N$')
    ax2.set_ylabel('AMI')
    ax2.set_title('(b) Quality at Scale')
    ax2.legend(ncol=2, fontsize=6.5, loc='lower left', framealpha=0.9)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(OUT / 'fig2_lfr_scalability.pdf')
    fig.savefig(OUT / 'fig2_lfr_scalability.png')
    plt.close(fig)
    print("  -> saved fig2_lfr_scalability.pdf/.png")

    # ---- Print summary stats ----
    print("\n  === Scalability summary stats ===")
    for N_val in sorted(cl['N'].unique()):
        cl_sub = cl[cl['N'] == N_val]
        if cl_sub.empty:
            continue
        best_ami = cl_sub.groupby('algorithm')['ami'].mean().sort_values(ascending=False)
        best_rt  = cl_sub.groupby('algorithm')['runtime'].mean().sort_values()
        print(f"  N={N_val}  top AMI: {ALGO_DISPLAY.get(best_ami.index[0], best_ami.index[0])}={best_ami.iloc[0]:.3f}"
              f"  fastest: {ALGO_DISPLAY.get(best_rt.index[0], best_rt.index[0])}={best_rt.iloc[0]:.3f}s")
    print("  GNN at N=25000:")
    for algo in sca_gnn:
        sub = gn[(gn['algorithm'] == algo) & (gn['N'] == 25000)]
        if not sub.empty:
            print(f"    {ALGO_DISPLAY.get(algo, algo)}: AMI={sub['ami'].mean():.3f}, RT={sub['runtime'].mean():.1f}s")


# =====================================================================
# FIGURE 3: Hierarchical (heatmaps)
# =====================================================================
def fig3_hierarchical():
    print("\nGenerating Figure 3: Hierarchical ...")
    cl = load_hierarchical_classical()
    gn = load_hierarchical_gnn()

    hier_algos_cl = sorted(cl['algorithm'].unique())
    # for GNN, separate micro/macro target
    gnn_base_algos = sorted(set(a.replace('_micro', '').replace('_macro', '')
                                for a in gn['algorithm'].unique()))

    mu_pairs = sorted(set(zip(cl['mu1'], cl['mu2'])))
    mu1_vals = sorted(cl['mu1'].unique())
    mu2_vals = sorted(cl['mu2'].unique())

    # Build classical summary table
    records_cl = []
    for algo in hier_algos_cl:
        sub = cl[cl['algorithm'] == algo]
        for mu1 in mu1_vals:
            for mu2 in mu2_vals:
                cell = sub[(sub['mu1'] == mu1) & (sub['mu2'] == mu2)]
                if cell.empty:
                    continue
                records_cl.append({
                    'algorithm': algo,
                    'mu1': mu1, 'mu2': mu2,
                    'ami_micro': cell['ami_micro'].mean(),
                    'ami_macro': cell['ami_macro'].mean(),
                })
    df_cl = pd.DataFrame(records_cl)

    # Build GNN summary table
    records_gnn = []
    for algo_raw in gn['algorithm'].unique():
        base = algo_raw.replace('_micro', '').replace('_macro', '')
        target = 'micro' if '_micro' in algo_raw else 'macro'
        sub = gn[gn['algorithm'] == algo_raw]
        for mu1 in mu1_vals:
            for mu2 in mu2_vals:
                cell = sub[(sub['mu1'] == mu1) & (sub['mu2'] == mu2)]
                if cell.empty:
                    continue
                records_gnn.append({
                    'algorithm': ALGO_DISPLAY.get(base, base),
                    'target': target,
                    'mu1': mu1, 'mu2': mu2,
                    'ami_micro': cell['ami_micro'].mean(),
                    'ami_macro': cell['ami_macro'].mean(),
                })
    df_gnn = pd.DataFrame(records_gnn)

    # ----- Heatmap: rows = algorithms, columns = (mu1,mu2) pairs -----
    # Show micro AMI for classical, both targets for GNN
    all_algos = []
    micro_data = []
    macro_data = []

    for algo in hier_algos_cl:
        label = ALGO_DISPLAY.get(algo, algo)
        all_algos.append(label + ' (Cl)')
        row_mi, row_ma = [], []
        for mu1 in mu1_vals:
            for mu2 in mu2_vals:
                c = df_cl[(df_cl['algorithm'] == algo) &
                          (df_cl['mu1'] == mu1) & (df_cl['mu2'] == mu2)]
                row_mi.append(c['ami_micro'].values[0] if len(c) else np.nan)
                row_ma.append(c['ami_macro'].values[0] if len(c) else np.nan)
        micro_data.append(row_mi)
        macro_data.append(row_ma)

    for algo_base in gnn_base_algos:
        label = ALGO_DISPLAY.get(algo_base, algo_base)
        # micro-targeted
        all_algos.append(label + ' (GNN-μ)')
        row_mi, row_ma = [], []
        for mu1 in mu1_vals:
            for mu2 in mu2_vals:
                c = df_gnn[(df_gnn['algorithm'] == label) &
                           (df_gnn['target'] == 'micro') &
                           (df_gnn['mu1'] == mu1) & (df_gnn['mu2'] == mu2)]
                row_mi.append(c['ami_micro'].values[0] if len(c) else np.nan)
                row_ma.append(c['ami_macro'].values[0] if len(c) else np.nan)
        micro_data.append(row_mi)
        macro_data.append(row_ma)
        # macro-targeted
        all_algos.append(label + ' (GNN-M)')
        row_mi, row_ma = [], []
        for mu1 in mu1_vals:
            for mu2 in mu2_vals:
                c = df_gnn[(df_gnn['algorithm'] == label) &
                           (df_gnn['target'] == 'macro') &
                           (df_gnn['mu1'] == mu1) & (df_gnn['mu2'] == mu2)]
                row_mi.append(c['ami_micro'].values[0] if len(c) else np.nan)
                row_ma.append(c['ami_macro'].values[0] if len(c) else np.nan)
        micro_data.append(row_mi)
        macro_data.append(row_ma)

    col_labels = [f'({m1:.1f},{m2:.1f})' for m1 in mu1_vals for m2 in mu2_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    micro_arr = np.array(micro_data)
    macro_arr = np.array(macro_data)

    im1 = ax1.imshow(micro_arr, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax1.set_xticks(range(len(col_labels)))
    ax1.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
    ax1.set_yticks(range(len(all_algos)))
    ax1.set_yticklabels(all_algos, fontsize=7)
    ax1.set_title('(a) Micro-level AMI')
    for i in range(micro_arr.shape[0]):
        for j in range(micro_arr.shape[1]):
            v = micro_arr[i, j]
            if not np.isnan(v):
                ax1.text(j, i, f'{v:.2f}', ha='center', va='center',
                         fontsize=5.5, color='black' if v > 0.4 else 'white')
    plt.colorbar(im1, ax=ax1, shrink=0.6)

    im2 = ax2.imshow(macro_arr, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_xticks(range(len(col_labels)))
    ax2.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
    ax2.set_yticks(range(len(all_algos)))
    ax2.set_yticklabels(all_algos, fontsize=7)
    ax2.set_title('(b) Macro-level AMI')
    for i in range(macro_arr.shape[0]):
        for j in range(macro_arr.shape[1]):
            v = macro_arr[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f'{v:.2f}', ha='center', va='center',
                         fontsize=5.5, color='black' if v > 0.4 else 'white')
    plt.colorbar(im2, ax=ax2, shrink=0.6)

    fig.tight_layout()
    fig.savefig(OUT / 'fig3_lfr_hierarchical.pdf')
    fig.savefig(OUT / 'fig3_lfr_hierarchical.png')
    plt.close(fig)
    print("  -> saved fig3_lfr_hierarchical.pdf/.png")

    # ---- Print summary ----
    print("\n  === Hierarchical summary stats ===")
    easy = df_cl[(df_cl['mu1'] == 0.1) & (df_cl['mu2'] == 0.1)]
    hard = df_cl[(df_cl['mu1'] == 0.3) & (df_cl['mu2'] == 0.3)]
    print(f"  Easy (0.1,0.1) classical:")
    for _, r in easy.iterrows():
        print(f"    {ALGO_DISPLAY.get(r['algorithm'], r['algorithm'])}: "
              f"micro={r['ami_micro']:.3f}, macro={r['ami_macro']:.3f}")
    print(f"  Hard (0.3,0.3) classical:")
    for _, r in hard.iterrows():
        print(f"    {ALGO_DISPLAY.get(r['algorithm'], r['algorithm'])}: "
              f"micro={r['ami_micro']:.3f}, macro={r['ami_macro']:.3f}")


# =====================================================================
# FIGURE 4: Label fraction ablation (GCN/GAT)
# =====================================================================
def fig4_label_ablation():
    print("\nGenerating Figure 4: Label fraction ablation ...")
    gn = load_gpu_accuracy_gnn_full()

    label_algos = {
        'gcn_supervised': 20, 'gcn_supervised_40': 40,
        'gcn_supervised_60': 60, 'gcn_supervised_80': 80,
        'gat_supervised': 20, 'gat_supervised_40': 40,
        'gat_supervised_60': 60, 'gat_supervised_80': 80,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    gcn_colors = {20: '#1f77b4', 40: '#ff7f0e', 60: '#2ca02c', 80: '#d62728'}
    gat_colors = {20: '#1f77b4', 40: '#ff7f0e', 60: '#2ca02c', 80: '#d62728'}

    for algo, frac in label_algos.items():
        if 'gcn' not in algo:
            continue
        sub = gn[gn['algorithm'] == algo]
        if sub.empty:
            continue
        grp = sub.groupby('mu').apply(lambda g: pd.Series({
            'ami_mean': g['ami'].mean(),
            'ami_se': 1.96 * g['ami'].std() / max(np.sqrt(len(g)), 1)
        })).reset_index().sort_values('mu')
        ax1.plot(grp['mu'], grp['ami_mean'], color=gcn_colors[frac],
                 label=f'{frac}% labels', marker='o', markersize=3)
        ax1.fill_between(grp['mu'],
                         grp['ami_mean'] - grp['ami_se'],
                         grp['ami_mean'] + grp['ami_se'],
                         alpha=0.15, color=gcn_colors[frac])

    ax1.set_xlabel(r'Mixing parameter $\mu$')
    ax1.set_ylabel('AMI')
    ax1.set_title('(a) GCN Label Fraction Ablation')
    ax1.legend(fontsize=8)
    ax1.set_xlim(0.08, 0.82)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    for algo, frac in label_algos.items():
        if 'gat' not in algo:
            continue
        sub = gn[gn['algorithm'] == algo]
        if sub.empty:
            continue
        grp = sub.groupby('mu').apply(lambda g: pd.Series({
            'ami_mean': g['ami'].mean(),
            'ami_se': 1.96 * g['ami'].std() / max(np.sqrt(len(g)), 1)
        })).reset_index().sort_values('mu')
        ax2.plot(grp['mu'], grp['ami_mean'], color=gat_colors[frac],
                 label=f'{frac}% labels', marker='s', markersize=3)
        ax2.fill_between(grp['mu'],
                         grp['ami_mean'] - grp['ami_se'],
                         grp['ami_mean'] + grp['ami_se'],
                         alpha=0.15, color=gat_colors[frac])

    ax2.set_xlabel(r'Mixing parameter $\mu$')
    ax2.set_title('(b) GAT Label Fraction Ablation')
    ax2.legend(fontsize=8)
    ax2.set_xlim(0.08, 0.82)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / 'fig4_lfr_label_ablation.pdf')
    fig.savefig(OUT / 'fig4_lfr_label_ablation.png')
    plt.close(fig)
    print("  -> saved fig4_lfr_label_ablation.pdf/.png")

    # ---- Print summary stats ----
    print("\n  === Label ablation summary ===")
    for algo, frac in label_algos.items():
        sub = gn[gn['algorithm'] == algo]
        for mu_val in [0.3, 0.5]:
            cell = sub[np.isclose(sub['mu'], mu_val, atol=0.005)]
            if not cell.empty:
                print(f"  {ALGO_DISPLAY.get(algo, algo)} @ mu={mu_val}: AMI={cell['ami'].mean():.3f}")


# =====================================================================
# FIGURE 5: Combined overview - all algorithms ranked at key mu values
# =====================================================================
def fig5_bar_ranking():
    print("\nGenerating Figure 5: Algorithm ranking bar chart ...")
    cl = load_accuracy_classical()
    gn = load_accuracy_gnn()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

    for idx, mu_val in enumerate([0.3, 0.5, 0.7]):
        ax = axes[idx]
        cl_sub = cl[np.isclose(cl['mu'], mu_val, atol=0.005)]
        gn_sub = gn[np.isclose(gn['mu'], mu_val, atol=0.005)]
        # only base GNN (no label ablation variants)
        gn_sub = gn_sub[gn_sub['algorithm'].isin(GNN_BASE)]

        cl_means = cl_sub.groupby('algorithm')['ami'].mean()
        gn_means = gn_sub.groupby('algorithm')['ami'].mean()
        all_means = pd.concat([cl_means, gn_means]).sort_values(ascending=True)

        # top 20
        top = all_means.tail(20)
        colors = []
        for a in top.index:
            if a in GNN_BASE:
                colors.append('#d62728')
            else:
                colors.append('#1f77b4')

        labels = [ALGO_DISPLAY.get(a, a) for a in top.index]
        ax.barh(range(len(top)), top.values, color=colors, edgecolor='white', height=0.7)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('AMI')
        ax.set_title(f'$\\mu = {mu_val}$')
        ax.set_xlim(0, 1.05)
        ax.grid(True, axis='x', alpha=0.3)

    # custom legend
    from matplotlib.patches import Patch
    leg = [Patch(facecolor='#1f77b4', label='Classical'),
           Patch(facecolor='#d62728', label='GNN')]
    axes[2].legend(handles=leg, loc='lower right', fontsize=8)

    fig.suptitle('Algorithm Ranking by AMI at Key Mixing Parameters', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / 'fig5_lfr_ranking.pdf')
    fig.savefig(OUT / 'fig5_lfr_ranking.png')
    plt.close(fig)
    print("  -> saved fig5_lfr_ranking.pdf/.png")


# =====================================================================
# FIGURE 6: Detected communities vs true communities
# =====================================================================
def fig6_cluster_count():
    print("\nGenerating Figure 6: Detected vs true community count ...")
    cl = load_accuracy_classical()
    gn = load_accuracy_gnn()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for algo in TOP_CLASSICAL[:8]:
        sub = cl[cl['algorithm'] == algo]
        if sub.empty:
            continue
        grp = sub.groupby('mu').apply(lambda g: pd.Series({
            'ratio': (g['num_detected'] / g['num_true']).mean()
        })).reset_index().sort_values('mu')
        color = CLASSICAL_COLORS.get(algo, '#333333')
        ax1.plot(grp['mu'], grp['ratio'], color=color,
                 label=ALGO_DISPLAY.get(algo, algo), marker='.', markersize=3)

    ax1.axhline(1.0, color='black', ls='--', lw=1, alpha=0.5, label='True K')
    ax1.set_xlabel(r'Mixing parameter $\mu$')
    ax1.set_ylabel(r'$K_{\rm detected} / K_{\rm true}$')
    ax1.set_title('(a) Classical Algorithms')
    ax1.legend(ncol=2, fontsize=7, loc='upper left')
    ax1.set_xlim(0.08, 0.82)
    ax1.set_yscale('log')
    ax1.set_ylim(0.05, 100)
    ax1.grid(True, alpha=0.3)

    for algo in GNN_BASE:
        sub = gn[gn['algorithm'] == algo]
        if sub.empty:
            continue
        grp = sub.groupby('mu').apply(lambda g: pd.Series({
            'ratio': (g['num_detected'] / g['num_true']).mean()
        })).reset_index().sort_values('mu')
        color = GNN_COLORS.get(algo, '#333333')
        marker = GNN_MARKERS.get(algo, 'o')
        ax2.plot(grp['mu'], grp['ratio'], color=color, marker=marker,
                 markersize=4, label=ALGO_DISPLAY.get(algo, algo))

    ax2.axhline(1.0, color='black', ls='--', lw=1, alpha=0.5, label='True K')
    ax2.set_xlabel(r'Mixing parameter $\mu$')
    ax2.set_title('(b) GNN Algorithms')
    ax2.legend(ncol=2, fontsize=7, loc='upper left')
    ax2.set_xlim(0.08, 0.82)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / 'fig6_lfr_cluster_count.pdf')
    fig.savefig(OUT / 'fig6_lfr_cluster_count.png')
    plt.close(fig)
    print("  -> saved fig6_lfr_cluster_count.pdf/.png")


# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("LFR Benchmark Figure Generator")
    print("=" * 60)
    fig1_accuracy()
    fig2_scalability()
    fig3_hierarchical()
    fig4_label_ablation()
    fig5_bar_ranking()
    fig6_cluster_count()
    print("\n" + "=" * 60)
    print("All figures generated in:", OUT)
    print("=" * 60)
