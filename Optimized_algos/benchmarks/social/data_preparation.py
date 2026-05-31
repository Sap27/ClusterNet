"""
Phase 1: Social Network Data Preparation.

Downloads Cora, Citeseer, Amazon Photo via PyTorch Geometric and saves
them in the standard format used by run_benchmark.py:
    data/<net_id>/edges.npy, features.npy, labels.npy,
                  node_list.json, metadata.json

Also pre-computes structural features (degree, clustering coeff, PageRank,
spectral embedding) for feature ablation experiments.

Usage:
    python data_preparation.py                # prepare all networks
    python data_preparation.py --network cora # single network
"""
import os, sys, json, argparse
import numpy as np
import networkx as nx
from pathlib import Path

# Local config first (before adding LFR to path, which has its own config.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import NETWORKS, NETWORK_ORDER, OUTPUT_DIR, SOCIAL_DATA_DIR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lfr'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def download_pyg_dataset(net_cfg):
    """Download a dataset via PyTorch Geometric and return (edge_index, features, labels)."""
    import torch
    raw_dir = str(SOCIAL_DATA_DIR / 'raw_pyg')

    if net_cfg['pyg_dataset'] == 'Planetoid':
        from torch_geometric.datasets import Planetoid
        ds = Planetoid(root=raw_dir, name=net_cfg['pyg_name'])
    elif net_cfg['pyg_dataset'] == 'Amazon':
        from torch_geometric.datasets import Amazon
        ds = Amazon(root=raw_dir, name=net_cfg['pyg_name'])
    else:
        raise ValueError(f"Unknown pyg_dataset: {net_cfg['pyg_dataset']}")

    data = ds[0]
    edge_index = data.edge_index.numpy()          # (2, num_edges)
    features = data.x.numpy().astype(np.float32)  # (num_nodes, feat_dim)
    labels = data.y.numpy().astype(np.int64)       # (num_nodes,)
    return edge_index, features, labels


def build_structural_features(G, dim=16):
    """
    Build structural features for feature ablation:
      [degree, clustering_coeff, pagerank, spectral_embedding(dim-3)]
    """
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())

    degree = np.array([G.degree(v) for v in nodes], dtype=np.float32)
    degree_norm = degree / (degree.max() + 1e-8)

    cc = nx.clustering(G)
    clustering = np.array([cc[v] for v in nodes], dtype=np.float32)

    pr = nx.pagerank(G, max_iter=100)
    pagerank = np.array([pr[v] for v in nodes], dtype=np.float32)
    pagerank_norm = pagerank / (pagerank.max() + 1e-8)

    spectral_dim = max(1, dim - 3)
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        A = nx.adjacency_matrix(G, nodelist=nodes).astype(np.float32)
        D_inv_sqrt = csr_matrix(np.diag(1.0 / np.sqrt(np.maximum(degree, 1))))
        L_norm = csr_matrix(np.eye(n, dtype=np.float32)) - D_inv_sqrt @ A @ D_inv_sqrt
        k = min(spectral_dim + 1, n - 2)
        if k < 2:
            spectral = np.zeros((n, spectral_dim), dtype=np.float32)
        else:
            eigenvalues, eigenvectors = eigsh(L_norm, k=k, which='SM')
            spectral = eigenvectors[:, 1:spectral_dim + 1].astype(np.float32)
            if spectral.shape[1] < spectral_dim:
                pad = np.zeros((n, spectral_dim - spectral.shape[1]), dtype=np.float32)
                spectral = np.concatenate([spectral, pad], axis=1)
    except Exception as e:
        print(f"  Warning: spectral embedding failed ({e}), using zeros", flush=True)
        spectral = np.zeros((n, spectral_dim), dtype=np.float32)

    features = np.column_stack([degree_norm, clustering, pagerank_norm, spectral])
    return features


def prepare_network(net_id):
    """Download, preprocess, and save a single social network."""
    net_cfg = NETWORKS[net_id]
    out_dir = OUTPUT_DIR / net_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Preparing: {net_id} — {net_cfg['description']}", flush=True)
    print(f"{'='*60}", flush=True)

    # Download
    print("  Downloading via PyTorch Geometric...", flush=True)
    edge_index, features, labels = download_pyg_dataset(net_cfg)
    n = features.shape[0]
    print(f"  Raw: {n} nodes, {edge_index.shape[1]} directed edges, "
          f"{features.shape[1]} features, {len(np.unique(labels))} classes", flush=True)

    # Build undirected NetworkX graph (deduplicate edges)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges_set = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u != v:
            edges_set.add((min(u, v), max(u, v)))
    G.add_edges_from(edges_set)
    num_edges = G.number_of_edges()
    print(f"  Undirected graph: {n} nodes, {num_edges} edges", flush=True)

    # Save edges as structured array (same format as biological)
    edge_dtype = np.dtype([('src', 'i4'), ('tgt', 'i4'), ('weight', 'f4')])
    edge_arr = np.empty(num_edges, dtype=edge_dtype)
    for i, (u, v) in enumerate(sorted(G.edges())):
        edge_arr[i] = (u, v, 1.0)
    np.save(str(out_dir / 'edges.npy'), edge_arr)

    # Save real features
    np.save(str(out_dir / 'features.npy'), features)

    # Save labels (ground truth)
    np.save(str(out_dir / 'labels.npy'), labels)

    # Save node list (just integer IDs as strings for compatibility)
    node_list = [str(i) for i in range(n)]
    with open(str(out_dir / 'node_list.json'), 'w') as f:
        json.dump(node_list, f)

    # Build and save structural features
    print("  Computing structural features...", flush=True)
    struct_feats = build_structural_features(G, dim=16)
    np.save(str(out_dir / 'features_structural.npy'), struct_feats)

    # Save random features (same dim as real, fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    random_feats = rng.randn(n, features.shape[1]).astype(np.float32)
    np.save(str(out_dir / 'features_random.npy'), random_feats)

    # Metadata
    num_classes = len(np.unique(labels))
    homophily = compute_edge_homophily(G, labels)
    metadata = {
        'net_id': net_id,
        'num_nodes': n,
        'num_edges': num_edges,
        'num_classes': num_classes,
        'feature_dim': int(features.shape[1]),
        'structural_feature_dim': int(struct_feats.shape[1]),
        'edge_homophily': round(float(homophily), 4),
        'description': net_cfg['description'],
        'pyg_dataset': net_cfg['pyg_dataset'],
        'pyg_name': net_cfg['pyg_name'],
    }
    with open(str(out_dir / 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved to {out_dir}", flush=True)
    print(f"  Nodes: {n}, Edges: {num_edges}, Classes: {num_classes}, "
          f"Feature dim: {features.shape[1]}, Homophily: {homophily:.3f}", flush=True)
    return metadata


def compute_edge_homophily(G, labels):
    """Fraction of edges connecting nodes with the same label."""
    same = 0
    total = 0
    for u, v in G.edges():
        total += 1
        if labels[u] == labels[v]:
            same += 1
    return same / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description='Social Network Data Preparation')
    parser.add_argument('--network', type=str, default=None,
                        help='Prepare specific network (cora, citeseer, amazon_photo)')
    args = parser.parse_args()

    if args.network:
        if args.network not in NETWORKS:
            print(f"Unknown network: {args.network}. Available: {list(NETWORKS.keys())}")
            sys.exit(1)
        nets = [args.network]
    else:
        nets = NETWORK_ORDER

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Social Network Data Preparation", flush=True)
    print(f"Output: {OUTPUT_DIR}", flush=True)
    print(f"Networks: {nets}", flush=True)

    all_meta = {}
    for net_id in nets:
        meta = prepare_network(net_id)
        all_meta[net_id] = meta

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Network':<15} {'Nodes':>8} {'Edges':>10} {'Classes':>8} "
          f"{'Features':>9} {'Homophily':>10}", flush=True)
    print(f"  {'-'*62}", flush=True)
    for net_id in nets:
        m = all_meta[net_id]
        print(f"  {net_id:<15} {m['num_nodes']:>8} {m['num_edges']:>10} "
              f"{m['num_classes']:>8} {m['feature_dim']:>9} "
              f"{m['edge_homophily']:>10.3f}", flush=True)

    print("\nData preparation complete.", flush=True)


if __name__ == '__main__':
    main()
