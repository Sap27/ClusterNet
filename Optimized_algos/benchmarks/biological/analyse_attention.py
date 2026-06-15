"""
Post-processing: GAT Attention Weight Analysis for Biological Networks.

Reads saved attention weight files (JSON) and:
  1. Computes mean attention per edge across heads
  2. Cross-references high-attention edges with directed edges (SIGNOR, GRN)
  3. On SIGNOR: compares attention distribution for activating vs inhibiting edges
  4. On GRN: compares attention to TF-target regulatory confidence
  5. Generates figures

Usage:
    python analyse_attention.py [--results-dir PATH] [--output-dir PATH]
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from collections import defaultdict
from scipy import stats

NETWORK_DISPLAY = {
    'signor': 'SIGNOR',
    'grn': 'GRN',
    'biogrid': 'BioGRID',
    'gtex_0.9': 'GTEx',
}

DATA_DIR = Path('/mnt/d/cluster/BIologicalNetworks/data')


def load_attention_data(results_dir, net_id, seed=0):
    """Load attention weights for a network."""
    attn_file = results_dir / 'attention' / f'{net_id}_gat_s{seed}.json'
    if not attn_file.is_file():
        return None
    return json.load(open(str(attn_file)))


def load_directed_edges(net_id):
    """Load directed edges for GRN or SIGNOR."""
    de_path = DATA_DIR / net_id / 'directed_edges.json'
    if not de_path.is_file():
        return None
    return json.load(open(str(de_path)))


def compute_mean_attention(attn_data):
    """Compute mean attention score per edge across all layers and heads.

    Returns:
        edge_index: np.ndarray [2, E]
        mean_attn: np.ndarray [E] - mean attention across heads/layers
    """
    edge_index = np.array(attn_data['edge_index'])
    attention_weights = attn_data['attention_weights']

    # attention_weights is a list (per layer) of [E_layer, num_heads] or [E_layer]
    # We use the first layer's attention as it directly reflects input edge importance
    first_layer_attn = np.array(attention_weights[0])
    if first_layer_attn.ndim == 2:
        mean_attn = first_layer_attn.mean(axis=1)
    else:
        mean_attn = first_layer_attn

    return edge_index, mean_attn


def analyse_signor_attention(attn_data, results_dir, output_dir):
    """Compare attention on activating vs inhibiting edges in SIGNOR."""
    print("\n  SIGNOR: Activating vs Inhibiting edge attention", flush=True)

    edge_index, mean_attn = compute_mean_attention(attn_data)
    node_list = attn_data.get('node_list') or json.load(
        open(DATA_DIR / 'signor' / 'node_list.json'))
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    directed_edges = load_directed_edges('signor')
    if directed_edges is None:
        print("    No directed edges file for SIGNOR.")
        return

    # Build edge -> sign mapping from directed edges
    edge_sign = {}
    for src, tgt, w in directed_edges:
        i_src = node_to_idx.get(src)
        i_tgt = node_to_idx.get(tgt)
        if i_src is not None and i_tgt is not None:
            edge_sign[(i_src, i_tgt)] = 'activating' if w > 0 else 'inhibiting'
            edge_sign[(i_tgt, i_src)] = 'activating' if w > 0 else 'inhibiting'

    # Classify attention scores
    act_attn = []
    inh_attn = []
    for e_idx in range(edge_index.shape[1]):
        src, tgt = int(edge_index[0, e_idx]), int(edge_index[1, e_idx])
        sign = edge_sign.get((src, tgt))
        if sign == 'activating':
            act_attn.append(mean_attn[e_idx])
        elif sign == 'inhibiting':
            inh_attn.append(mean_attn[e_idx])

    act_attn = np.array(act_attn)
    inh_attn = np.array(inh_attn)

    print(f"    Activating edges: {len(act_attn)}, mean attn = {act_attn.mean():.4f}")
    print(f"    Inhibiting edges: {len(inh_attn)}, mean attn = {inh_attn.mean():.4f}")

    if len(act_attn) > 0 and len(inh_attn) > 0:
        t_stat, p_val = stats.mannwhitneyu(act_attn, inh_attn, alternative='two-sided')
        print(f"    Mann-Whitney U test: p = {p_val:.2e}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(act_attn, bins=50, alpha=0.6, label=f'Activating (n={len(act_attn)})',
                color='#4CAF50', density=True)
        ax.hist(inh_attn, bins=50, alpha=0.6, label=f'Inhibiting (n={len(inh_attn)})',
                color='#F44336', density=True)
        ax.set_xlabel('Mean Attention Score')
        ax.set_ylabel('Density')
        ax.set_title(f'SIGNOR: GAT Attention by Edge Type (p={p_val:.2e})')
        ax.legend()
        plt.tight_layout()
        out_path = output_dir / 'signor_attention_by_sign.pdf'
        plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {out_path}")

    return {
        'network': 'signor',
        'n_activating': len(act_attn),
        'n_inhibiting': len(inh_attn),
        'mean_attn_activating': float(act_attn.mean()) if len(act_attn) > 0 else None,
        'mean_attn_inhibiting': float(inh_attn.mean()) if len(inh_attn) > 0 else None,
        'p_value': float(p_val) if len(act_attn) > 0 and len(inh_attn) > 0 else None,
    }


def analyse_grn_attention(attn_data, results_dir, output_dir):
    """Compare attention to TF out-degree (regulatory importance) in GRN."""
    print("\n  GRN: Attention vs regulatory importance", flush=True)

    edge_index, mean_attn = compute_mean_attention(attn_data)
    node_list = attn_data.get('node_list') or json.load(
        open(DATA_DIR / 'grn' / 'node_list.json'))
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    directed_edges = load_directed_edges('grn')
    if directed_edges is None:
        print("    No directed edges file for GRN.")
        return

    # Compute out-degree per TF
    out_degree = defaultdict(int)
    for src, tgt, w in directed_edges:
        i_src = node_to_idx.get(src)
        if i_src is not None:
            out_degree[i_src] += 1

    # For each node that is a TF, compute its mean incoming attention
    node_in_attn = defaultdict(list)
    for e_idx in range(edge_index.shape[1]):
        tgt = int(edge_index[1, e_idx])
        node_in_attn[tgt].append(mean_attn[e_idx])

    # Correlate: for TFs, does out-degree correlate with attention received?
    tf_nodes = [n for n in out_degree if out_degree[n] > 0]
    if not tf_nodes:
        print("    No TF nodes identified.")
        return

    out_degs = []
    mean_attns = []
    for tf in tf_nodes:
        if tf in node_in_attn and node_in_attn[tf]:
            out_degs.append(out_degree[tf])
            mean_attns.append(np.mean(node_in_attn[tf]))

    out_degs = np.array(out_degs)
    mean_attns = np.array(mean_attns)

    if len(out_degs) < 5:
        print("    Too few TF data points for correlation.")
        return

    rho, p_val = stats.spearmanr(out_degs, mean_attns)
    print(f"    TF out-degree vs mean attention: rho={rho:.3f}, p={p_val:.2e}")
    print(f"    ({len(out_degs)} TFs analysed)")

    # Top-10 most attended TFs
    top_idx = np.argsort(mean_attns)[-10:][::-1]
    print(f"    Top-10 attended TFs:")
    for idx in top_idx:
        tf_node = tf_nodes[idx] if idx < len(tf_nodes) else None
        if tf_node is not None:
            print(f"      {node_list[tf_node]}: attn={mean_attns[idx]:.4f}, "
                  f"out-degree={out_degs[idx]}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(out_degs, mean_attns, alpha=0.4, s=10, c='#2196F3')
    ax.set_xlabel('TF Out-degree (regulatory targets)')
    ax.set_ylabel('Mean Incoming Attention')
    ax.set_title(f'GRN: GAT Attention vs TF Regulatory Importance\n'
                 f'(Spearman rho={rho:.3f}, p={p_val:.2e})')
    ax.set_xscale('log')
    plt.tight_layout()
    out_path = output_dir / 'grn_attention_vs_outdegree.pdf'
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {out_path}")

    return {
        'network': 'grn',
        'n_tfs': len(out_degs),
        'spearman_rho': float(rho),
        'p_value': float(p_val),
    }


def plot_attention_distribution(results_dir, output_dir, preloaded=None):
    """Plot attention score distribution across networks.
    
    Args:
        preloaded: dict mapping net_id -> attn_data already in memory (avoids reloading large files)
    """
    attn_dir = results_dir / 'attention'
    if not attn_dir.is_dir():
        print("  No attention directory.")
        return

    # Only process networks that are manageable in size or already loaded
    networks = [n for n in ['signor', 'grn', 'biogrid', 'gtex_0.9']
                if (attn_dir / f'{n}_gat_s0.json').is_file()]
    if not networks:
        print("  No attention files found.")
        return

    # For large networks (>200MB), skip unless preloaded
    MAX_FILE_SIZE = 200 * 1024 * 1024
    if preloaded is None:
        preloaded = {}
    loadable = []
    for net_id in networks:
        fpath = attn_dir / f'{net_id}_gat_s0.json'
        if net_id in preloaded:
            loadable.append(net_id)
        elif fpath.stat().st_size < MAX_FILE_SIZE:
            loadable.append(net_id)
        else:
            print(f"    Skipping {net_id} distribution (file too large: "
                  f"{fpath.stat().st_size / 1e6:.0f} MB)")
    networks = loadable
    if not networks:
        return

    fig, axes = plt.subplots(1, len(networks), figsize=(4*len(networks), 3.5))
    if len(networks) == 1:
        axes = [axes]

    for ax, net_id in zip(axes, networks):
        if net_id in preloaded:
            attn_data = preloaded[net_id]
        else:
            attn_data = load_attention_data(results_dir, net_id)
        _, mean_attn = compute_mean_attention(attn_data)
        ax.hist(mean_attn, bins=100, color='#2196F3', alpha=0.7, density=True)
        ax.axvline(mean_attn.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean={mean_attn.mean():.4f}')
        ax.set_title(NETWORK_DISPLAY[net_id], fontsize=11, fontweight='bold')
        ax.set_xlabel('Attention score')
        ax.set_ylabel('Density' if ax == axes[0] else '')
        ax.legend(fontsize=8)

        entropy = -np.sum(mean_attn * np.log(mean_attn + 1e-10)) / len(mean_attn)
        ax.text(0.95, 0.95, f'H={entropy:.3f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='gray')

    plt.suptitle('GAT Attention Score Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = output_dir / 'attention_distribution.pdf'
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='GAT Attention Analysis')
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

    print("="*60, flush=True)
    print("GAT ATTENTION WEIGHT ANALYSIS", flush=True)
    print("="*60, flush=True)

    summary = []
    preloaded = {}

    # SIGNOR analysis
    signor_attn = load_attention_data(results_dir, 'signor')
    if signor_attn:
        preloaded['signor'] = signor_attn
        result = analyse_signor_attention(signor_attn, results_dir, output_dir)
        if result:
            summary.append(result)

    # GRN analysis
    grn_attn = load_attention_data(results_dir, 'grn')
    if grn_attn:
        preloaded['grn'] = grn_attn
        result = analyse_grn_attention(grn_attn, results_dir, output_dir)
        if result:
            summary.append(result)

    # Attention distribution across all networks
    print("\n  Generating attention distribution plots...", flush=True)
    plot_attention_distribution(results_dir, output_dir, preloaded=preloaded)

    # Save summary
    if summary:
        summary_path = results_dir / 'attention_analysis_summary.json'
        with open(str(summary_path), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved: {summary_path}")


if __name__ == '__main__':
    main()
