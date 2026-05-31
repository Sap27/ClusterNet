"""
Post-processing: GNN Embedding Visualization for Biological Networks.

Reads saved embeddings (.npy) and generates t-SNE/UMAP composite figures
colored by:
  - Detected community assignment
  - GO Slim biological category
  - Network-specific annotation

Usage:
    python plot_bio_embeddings.py [--results-dir PATH] [--output-dir PATH]
"""
import argparse
import json
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from collections import defaultdict
from sklearn.manifold import TSNE

NETWORK_DISPLAY = {
    'signor': 'SIGNOR',
    'grn': 'GRN',
    'biogrid': 'BioGRID',
    'gtex_0.9': 'GTEx',
}
MODEL_DISPLAY = {
    'dgi_gnn': 'DGI',
    'dmon_gnn': 'DMoN',
    'vgae_gnn': 'VGAE',
    'gat_supervised': 'GAT (sup)',
}

DATA_DIR = Path('/mnt/d/cluster/BIologicalNetworks/data')
GAF_PATH = Path('/mnt/d/cluster/BIologicalNetworks/goa_human.gaf.gz')

GO_SLIM_BP = {
    'GO:0008152': 'Metabolism',
    'GO:0009987': 'Cellular process',
    'GO:0050896': 'Response to stimulus',
    'GO:0065007': 'Biological regulation',
    'GO:0032502': 'Development',
    'GO:0051179': 'Localization',
    'GO:0023052': 'Signaling',
    'GO:0006950': 'Stress response',
    'GO:0002376': 'Immune process',
    'GO:0007049': 'Cell cycle',
    'GO:0012501': 'Apoptosis',
    'GO:0007155': 'Cell adhesion',
    'GO:0048870': 'Cell motility',
    'GO:0030154': 'Differentiation',
    'GO:0019725': 'Homeostasis',
}


def load_go_slim_labels(net_id):
    """Load GO Slim labels for a network's nodes."""
    node_list = json.load(open(DATA_DIR / net_id / 'node_list.json'))
    meta = json.load(open(DATA_DIR / net_id / 'metadata.json'))
    id_col = 1 if meta['id_format'] == 'uniprot' else 2

    gene_to_terms = defaultdict(set)
    with gzip.open(str(GAF_PATH), 'rt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            gene_id = parts[id_col]
            go_term = parts[4]
            if parts[8] == 'P':
                gene_to_terms[gene_id].add(go_term)

    labels = []
    for gene in node_list:
        terms = gene_to_terms.get(gene, set())
        found = None
        for t in terms:
            if t in GO_SLIM_BP:
                found = GO_SLIM_BP[t]
                break
        labels.append(found)

    return labels, node_list


def load_community_labels(net_id, algo_name, results_dir, seed=0):
    """Load partition from results and return per-node community labels."""
    partitions_dir = results_dir / 'partitions'
    # Try with 'real' feature type first, then without feature type
    candidates = [
        partitions_dir / f'{net_id}_{algo_name}_real_s{seed}.json',
        partitions_dir / f'{net_id}_{algo_name}_s{seed}.json',
    ]
    part_file = None
    for cand in candidates:
        if cand.is_file():
            part_file = cand
            break
    if part_file is None:
        return None

    communities = json.load(open(str(part_file)))
    if not communities:
        return None
    n = max(max(c) for c in communities if c) + 1
    labels = np.full(n, -1)
    for ci, comm in enumerate(communities):
        for node in comm:
            labels[node] = ci
    return labels


def run_tsne(embeddings, perplexity=30, seed=42):
    """Run t-SNE on embeddings, with PCA pre-reduction if dims are high."""
    n = embeddings.shape[0]
    # If embedding dim is very high, reduce with PCA first
    if embeddings.shape[1] > 64:
        from sklearn.decomposition import PCA
        target_dim = min(50, embeddings.shape[1])
        embeddings = PCA(n_components=target_dim, random_state=seed).fit_transform(embeddings)
    perp = min(perplexity, n // 4)
    if perp < 5:
        perp = 5
    tsne = TSNE(n_components=2, perplexity=perp, random_state=seed,
                learning_rate='auto', init='pca')
    return tsne.fit_transform(embeddings)


def plot_composite_figure(results_dir, output_dir):
    """Generate a composite figure: rows=networks, cols=models, colored by community."""
    embeddings_dir = results_dir / 'embeddings'
    if not embeddings_dir.is_dir():
        print("  No embeddings directory found. Skipping composite figure.")
        return

    networks = [n for n in ['signor', 'grn', 'biogrid', 'gtex_0.9']
                if (embeddings_dir / f'{n}_dgi_gnn_s0.npy').is_file()]
    models = [m for m in MODEL_DISPLAY.keys()]

    if not networks:
        print("  No embedding files found.")
        return

    fig, axes = plt.subplots(len(networks), len(models),
                             figsize=(4*len(models), 3.5*len(networks)))
    if len(networks) == 1:
        axes = axes[np.newaxis, :]
    if len(models) == 1:
        axes = axes[:, np.newaxis]

    for i, net_id in enumerate(networks):
        go_labels, node_list = load_go_slim_labels(net_id)

        for j, model_name in enumerate(models):
            ax = axes[i, j]
            emb_file = embeddings_dir / f'{net_id}_{model_name}_s0.npy'

            if not emb_file.is_file():
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(MODEL_DISPLAY[model_name], fontsize=11, fontweight='bold')
                if j == 0:
                    ax.set_ylabel(NETWORK_DISPLAY[net_id], fontsize=11, fontweight='bold')
                continue

            # For very large embedding files (DMoN soft assignments), load and
            # subsample rows if necessary before PCA + t-SNE
            if emb_file.stat().st_size > 300 * 1024 * 1024:
                emb = np.load(str(emb_file), mmap_mode='r')
                n_nodes = emb.shape[0]
                if n_nodes > 15000:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(n_nodes, 15000, replace=False)
                    idx.sort()
                    emb = np.array(emb[idx])
                else:
                    emb = np.array(emb)
            else:
                emb = np.load(str(emb_file))
            coords = run_tsne(emb)

            # Color by community assignment
            comm_labels = load_community_labels(net_id, model_name, results_dir)
            if comm_labels is not None and len(comm_labels) == len(coords):
                colors = comm_labels
                cmap = 'tab20'
            else:
                colors = np.zeros(len(coords))
                cmap = 'gray'

            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors,
                                cmap=cmap, s=3, alpha=0.6, rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.set_title(MODEL_DISPLAY[model_name], fontsize=11, fontweight='bold')
            if j == 0:
                ax.set_ylabel(NETWORK_DISPLAY[net_id], fontsize=11, fontweight='bold')

    plt.suptitle('GNN Latent Space Embeddings (t-SNE, colored by community)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = output_dir / 'bio_embedding_tsne_composite.pdf'
    plt.savefig(str(out_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def plot_go_colored(results_dir, output_dir):
    """Same as composite but colored by GO Slim category."""
    embeddings_dir = results_dir / 'embeddings'
    if not embeddings_dir.is_dir():
        return

    networks = [n for n in ['signor', 'grn', 'biogrid', 'gtex_0.9']
                if (embeddings_dir / f'{n}_dgi_gnn_s0.npy').is_file()]

    if not networks:
        return

    # Use DGI as representative model for GO coloring
    fig, axes = plt.subplots(1, len(networks), figsize=(4*len(networks), 4))
    if len(networks) == 1:
        axes = [axes]

    unique_categories = sorted(set(GO_SLIM_BP.values()))
    cat_to_idx = {c: i for i, c in enumerate(unique_categories)}
    cmap = plt.cm.get_cmap('tab20', len(unique_categories))

    for ax, net_id in zip(axes, networks):
        emb_file = embeddings_dir / f'{net_id}_dgi_gnn_s0.npy'
        if not emb_file.is_file():
            continue

        emb = np.load(str(emb_file))
        coords = run_tsne(emb)
        go_labels, _ = load_go_slim_labels(net_id)

        # Plot unlabeled nodes in gray
        unlabeled = [i for i, l in enumerate(go_labels) if l is None]
        labeled = [i for i, l in enumerate(go_labels) if l is not None]

        ax.scatter(coords[unlabeled, 0], coords[unlabeled, 1],
                   c='lightgray', s=2, alpha=0.3, rasterized=True)

        colors = [cat_to_idx[go_labels[i]] for i in labeled]
        ax.scatter(coords[labeled, 0], coords[labeled, 1],
                   c=colors, cmap=cmap, s=5, alpha=0.7, rasterized=True,
                   vmin=0, vmax=len(unique_categories)-1)

        ax.set_title(NETWORK_DISPLAY[net_id], fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle('DGI Embeddings Colored by GO Slim Category',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = output_dir / 'bio_embedding_go_colored.pdf'
    plt.savefig(str(out_path), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Bio Embedding Visualization')
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

    print("Generating composite t-SNE figure (community-colored)...", flush=True)
    plot_composite_figure(results_dir, output_dir)

    print("\nGenerating GO-colored DGI embedding figure...", flush=True)
    plot_go_colored(results_dir, output_dir)


if __name__ == '__main__':
    main()
