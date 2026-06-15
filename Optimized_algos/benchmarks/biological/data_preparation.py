"""
Phase 1: Data Preparation for Biological Benchmarks.

Loads raw edge lists, builds node features (expression / GO terms),
symmetrizes directed networks, enumerates FFLs, and saves everything
to data/<network_id>/ as numpy/json files.

Usage:
    python data_preparation.py            # prepare all networks
    python data_preparation.py --network gtex_0.8   # prepare one network
"""
import os, sys, csv, gzip, json, argparse, time
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    RAW_NETWORKS, NETWORKS, OUTPUT_DIR, GTEx_TPM_PATH, GO_GAF_PATH, GO_OBO_PATH,
    EXPRESSION_PCA_DIM, GO_TERM_MIN_GENES, GO_TERM_MAX_GENES,
)


# =============================================================================
# ID MAPPING
# =============================================================================

def build_uniprot_to_symbol(gaf_path):
    """Parse GAF file to build UniProt accession -> gene symbol mapping."""
    print("  Loading UniProt -> symbol mapping from GAF...", flush=True)
    mapping = {}
    with gzip.open(str(gaf_path), 'rt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 11:
                continue
            uniprot = parts[1]
            symbol = parts[2]
            if uniprot not in mapping:
                mapping[uniprot] = symbol
    print(f"  Mapped {len(mapping):,} UniProt IDs to gene symbols", flush=True)
    return mapping


# =============================================================================
# NETWORK LOADING
# =============================================================================

def load_edge_list(path, threshold=None):
    """Load a CSV edge list (Source, Target, Weight). Returns edges + node set."""
    edges = []
    nodes = set()
    with open(str(path), 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            src, tgt = row[0], row[1]
            w = float(row[2]) if len(row) > 2 else 1.0
            if threshold is not None and w < threshold:
                continue
            edges.append((src, tgt, w))
            nodes.add(src)
            nodes.add(tgt)
    return edges, nodes


def symmetrize_edges(edges):
    """Convert directed edges to undirected by adding reverse edges.
    For duplicate (u,v) and (v,u), keep the max weight."""
    edge_map = {}
    for src, tgt, w in edges:
        key = tuple(sorted([src, tgt]))
        if key not in edge_map or abs(w) > abs(edge_map[key]):
            edge_map[key] = w
    return [(u, v, w) for (u, v), w in edge_map.items()]


# =============================================================================
# EXPRESSION FEATURES (GTEx TPM)
# =============================================================================

def load_tpm_matrix(tpm_path, target_genes, id_format='symbol', uniprot_map=None):
    """
    Load GTEx TPM matrix and extract rows for target genes.

    Args:
        tpm_path: Path to .gct file
        target_genes: set of gene identifiers present in the network
        id_format: 'symbol' or 'uniprot'
        uniprot_map: dict UniProt->symbol (required if id_format='uniprot')

    Returns:
        gene_order: list of gene IDs (in network's ID format) with TPM data
        tpm_matrix: np.ndarray of shape (len(gene_order), num_samples)
    """
    if id_format == 'uniprot' and uniprot_map is None:
        raise ValueError("uniprot_map required for id_format='uniprot'")

    if id_format == 'uniprot':
        symbol_to_uniprots = defaultdict(list)
        for uni, sym in uniprot_map.items():
            if uni in target_genes:
                symbol_to_uniprots[sym].append(uni)
        target_symbols = set(symbol_to_uniprots.keys())
    else:
        target_symbols = target_genes

    print(f"  Loading TPM matrix from {Path(tpm_path).name}...", flush=True)
    gene_ids = []
    rows = []

    with open(str(tpm_path), 'r') as f:
        f.readline()  # #1.2
        f.readline()  # dimensions
        f.readline()  # header (Name, Description, samples...)
        for line in f:
            parts = line.split('\t')
            symbol = parts[1]
            if symbol not in target_symbols:
                continue
            values = np.array([float(x) for x in parts[2:]], dtype=np.float32)

            if id_format == 'uniprot':
                for uni in symbol_to_uniprots[symbol]:
                    gene_ids.append(uni)
                    rows.append(values)
            else:
                gene_ids.append(symbol)
                rows.append(values)

    tpm_matrix = np.stack(rows, axis=0) if rows else np.empty((0, 0))
    print(f"  Loaded TPM for {len(gene_ids):,} / {len(target_genes):,} genes "
          f"({tpm_matrix.shape[1]} samples)", flush=True)
    return gene_ids, tpm_matrix


def build_expression_features(tpm_matrix, pca_dim):
    """Log-transform + PCA reduce expression matrix."""
    print(f"  Log-transforming and PCA to {pca_dim} dimensions...", flush=True)
    log_tpm = np.log2(tpm_matrix + 1)
    actual_dim = min(pca_dim, log_tpm.shape[0], log_tpm.shape[1])
    pca = PCA(n_components=actual_dim, random_state=42)
    features = pca.fit_transform(log_tpm)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {actual_dim} components, {100*var_explained:.1f}% variance explained", flush=True)
    return features.astype(np.float32)


# =============================================================================
# GO TERM FEATURES
# =============================================================================

def load_go_annotations(gaf_path, target_genes, min_genes, max_genes):
    """
    Parse GAF to build gene -> GO term mapping, filtered to target genes.

    Returns:
        gene_order: list of genes (subset of target_genes with GO annotations)
        go_terms: list of GO term IDs
        feature_matrix: np.ndarray binary (len(gene_order), len(go_terms))
    """
    print(f"  Loading GO annotations from GAF...", flush=True)
    gene_to_terms = defaultdict(set)

    with gzip.open(str(gaf_path), 'rt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
            symbol = parts[2]
            go_term = parts[4]
            if symbol in target_genes:
                gene_to_terms[symbol].add(go_term)

    # Count genes per GO term
    term_counts = defaultdict(int)
    for terms in gene_to_terms.values():
        for t in terms:
            term_counts[t] += 1

    # Filter GO terms by frequency
    valid_terms = sorted(
        t for t, c in term_counts.items()
        if min_genes <= c <= max_genes
    )
    term_idx = {t: i for i, t in enumerate(valid_terms)}
    print(f"  {len(valid_terms)} GO terms retained (annotating {min_genes}-{max_genes} genes)", flush=True)

    # Build binary matrix
    gene_order = sorted(gene_to_terms.keys())
    matrix = np.zeros((len(gene_order), len(valid_terms)), dtype=np.float32)
    for i, gene in enumerate(gene_order):
        for term in gene_to_terms[gene]:
            if term in term_idx:
                matrix[i, term_idx[term]] = 1.0

    print(f"  GO feature matrix: {matrix.shape} "
          f"(density: {matrix.mean():.4f})", flush=True)
    return gene_order, valid_terms, matrix


def build_sign_aware_features(edges, node_list, go_features):
    """Append sign-aware degree features (in-degree from activators,
    in-degree from inhibitors) to GO features for SIGNOR."""
    node_idx = {n: i for i, n in enumerate(node_list)}
    act_in = np.zeros(len(node_list), dtype=np.float32)
    inh_in = np.zeros(len(node_list), dtype=np.float32)
    for src, tgt, w in edges:
        if tgt in node_idx:
            if w > 0:
                act_in[node_idx[tgt]] += 1
            else:
                inh_in[node_idx[tgt]] += 1

    # Normalize
    act_max = act_in.max() or 1.0
    inh_max = inh_in.max() or 1.0
    sign_feats = np.stack([act_in / act_max, inh_in / inh_max], axis=1)
    combined = np.hstack([go_features, sign_feats])
    print(f"  Added sign-aware degree features: {go_features.shape[1]} + 2 = "
          f"{combined.shape[1]} dims", flush=True)
    return combined


# =============================================================================
# FFL ENUMERATION (GRN)
# =============================================================================

def enumerate_ffls(directed_edges, nodes):
    """
    Find feed-forward loops in a directed graph.
    FFL: TF1 -> TF2, TF1 -> Gene, TF2 -> Gene (all three edges present).

    Returns list of (tf1, tf2, gene) triples.
    """
    print("  Enumerating feed-forward loops...", flush=True)
    out_neighbors = defaultdict(set)
    for src, tgt, _ in directed_edges:
        out_neighbors[src].add(tgt)

    ffls = []
    sources = sorted(out_neighbors.keys())
    for i, tf1 in enumerate(sources):
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(sources)} sources, "
                  f"{len(ffls)} FFLs so far...", flush=True)
        targets_tf1 = out_neighbors[tf1]
        for tf2 in targets_tf1:
            if tf2 not in out_neighbors:
                continue
            common = targets_tf1 & out_neighbors[tf2]
            for gene in common:
                ffls.append((tf1, tf2, gene))

    print(f"  Found {len(ffls):,} feed-forward loops", flush=True)
    return ffls


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def prepare_network(net_id, net_cfg):
    """Prepare a single biological network: load, featurize, save."""
    print(f"\n{'='*60}", flush=True)
    print(f"Preparing {net_id}: {net_cfg['description']}", flush=True)
    print(f"{'='*60}", flush=True)

    out_dir = OUTPUT_DIR / net_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load raw edges ---
    raw_path = RAW_NETWORKS[net_cfg['source']]
    edges, nodes = load_edge_list(raw_path, threshold=net_cfg.get('threshold'))
    print(f"  Loaded: {len(nodes):,} nodes, {len(edges):,} edges", flush=True)

    # --- ID mapping for GRN (UniProt -> symbol) ---
    uniprot_map = None
    if net_cfg['id_format'] == 'uniprot':
        uniprot_map = build_uniprot_to_symbol(GO_GAF_PATH)

    # --- Build node features ---
    if net_cfg['feature_source'] == 'expression':
        gene_order, tpm_matrix = load_tpm_matrix(
            GTEx_TPM_PATH, nodes, net_cfg['id_format'], uniprot_map
        )
        features = build_expression_features(tpm_matrix, EXPRESSION_PCA_DIM)
        feature_genes = set(gene_order)
    elif net_cfg['feature_source'] == 'go_terms':
        gene_order, go_terms, features = load_go_annotations(
            GO_GAF_PATH, nodes, GO_TERM_MIN_GENES, GO_TERM_MAX_GENES
        )
        feature_genes = set(gene_order)
        np.save(str(out_dir / 'go_terms.npy'), np.array(go_terms))
        if net_cfg.get('signed'):
            features = build_sign_aware_features(edges, gene_order, features)
    else:
        raise ValueError(f"Unknown feature_source: {net_cfg['feature_source']}")

    # --- Filter network to genes with features ---
    edges_filtered = [(s, t, w) for s, t, w in edges
                      if s in feature_genes and t in feature_genes]
    nodes_filtered = set()
    for s, t, _ in edges_filtered:
        nodes_filtered.add(s)
        nodes_filtered.add(t)

    print(f"  After feature filtering: {len(nodes_filtered):,} nodes, "
          f"{len(edges_filtered):,} edges "
          f"(dropped {len(nodes) - len(nodes_filtered):,} nodes without features)",
          flush=True)

    # Reindex features to match filtered nodes
    gene_idx = {g: i for i, g in enumerate(gene_order)}
    node_list = sorted(nodes_filtered)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    feature_matrix = np.zeros((len(node_list), features.shape[1]), dtype=np.float32)
    for node in node_list:
        if node in gene_idx:
            feature_matrix[node_to_idx[node]] = features[gene_idx[node]]

    # --- Save directed edges (for evaluation) ---
    if net_cfg['directed']:
        directed_edges = [(s, t, w) for s, t, w in edges_filtered]
        with open(str(out_dir / 'directed_edges.json'), 'w') as f:
            json.dump(directed_edges, f)
        print(f"  Saved {len(directed_edges):,} directed edges for evaluation", flush=True)

        # Enumerate FFLs for GRN
        if net_id == 'grn':
            ffls = enumerate_ffls(directed_edges, nodes_filtered)
            with open(str(out_dir / 'ffls.json'), 'w') as f:
                json.dump(ffls, f)

    # --- Symmetrize directed networks for community detection ---
    if net_cfg['directed']:
        undirected_edges = symmetrize_edges(edges_filtered)
        print(f"  Symmetrized: {len(edges_filtered):,} directed -> "
              f"{len(undirected_edges):,} undirected edges", flush=True)
    else:
        undirected_edges = edges_filtered

    # --- Save everything ---
    # Edge list (undirected, for community detection)
    edge_array = np.array(
        [(node_to_idx[s], node_to_idx[t], w) for s, t, w in undirected_edges],
        dtype=[('src', np.int32), ('tgt', np.int32), ('weight', np.float32)]
    )
    np.save(str(out_dir / 'edges.npy'), edge_array)

    # Node features
    np.save(str(out_dir / 'features.npy'), feature_matrix)

    # Node list (preserves ID -> index mapping)
    with open(str(out_dir / 'node_list.json'), 'w') as f:
        json.dump(node_list, f)

    # Metadata
    avg_deg = 2 * len(undirected_edges) / len(node_list) if node_list else 0
    metadata = {
        'network_id': net_id,
        'description': net_cfg['description'],
        'num_nodes': len(node_list),
        'num_edges_undirected': len(undirected_edges),
        'num_edges_directed': len(edges_filtered) if net_cfg['directed'] else None,
        'avg_degree': round(avg_deg, 1),
        'feature_source': net_cfg['feature_source'],
        'feature_dim': feature_matrix.shape[1],
        'directed': net_cfg['directed'],
        'signed': net_cfg.get('signed', False),
        'id_format': net_cfg['id_format'],
        'threshold': net_cfg.get('threshold'),
        'num_ffls': len(ffls) if net_id == 'grn' else None,
    }
    with open(str(out_dir / 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved to {out_dir}/", flush=True)
    print(f"  Summary: {metadata['num_nodes']:,} nodes, "
          f"{metadata['num_edges_undirected']:,} undirected edges, "
          f"{metadata['feature_dim']} feature dims", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Prepare biological network data')
    parser.add_argument('--network', type=str, default=None,
                        help='Prepare specific network (e.g. gtex_0.8, grn, biogrid, signor)')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if args.network:
        if args.network not in NETWORKS:
            print(f"Unknown network: {args.network}. Available: {list(NETWORKS.keys())}")
            sys.exit(1)
        networks_to_prepare = {args.network: NETWORKS[args.network]}
    else:
        networks_to_prepare = NETWORKS

    all_metadata = {}
    for net_id, net_cfg in networks_to_prepare.items():
        meta = prepare_network(net_id, net_cfg)
        all_metadata[net_id] = meta

    # Save combined metadata
    with open(str(OUTPUT_DIR / 'metadata.json'), 'w') as f:
        json.dump(all_metadata, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"All done in {elapsed:.0f}s. Prepared {len(all_metadata)} networks.", flush=True)
    for net_id, meta in all_metadata.items():
        print(f"  {net_id}: {meta['num_nodes']:,} nodes, "
              f"{meta['num_edges_undirected']:,} edges, "
              f"{meta['feature_dim']} feat dims", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
