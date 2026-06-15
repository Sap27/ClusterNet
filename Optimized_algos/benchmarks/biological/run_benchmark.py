"""
Phase 2: Biological Network Benchmark Runner.

Runs all classical + GNN community detection algorithms on preprocessed
biological networks. Saves partitions, modularity, runtime to CSV.

Usage:
    python run_benchmark.py --run-classical --run-gnn          # run everything
    python run_benchmark.py --run-classical --network signor    # one network, classical only
    python run_benchmark.py --run-gnn --resume --epochs 600     # resume GNN runs
    python run_benchmark.py --run-gnn --feature-ablation        # GNN with real/structural/random features
    python run_benchmark.py --run-gnn --save-embeddings         # export latent embeddings
    python run_benchmark.py --run-gnn --save-attention          # export GAT attention weights
"""
import os, sys, json, time, argparse, signal
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lfr'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithm_runner import AlgorithmRunner
from config import (
    NETWORKS, CLASSICAL_ALGORITHMS, CLASSICAL_ALGORITHMS_MEMSAFE,
    MEMSAFE_THRESHOLD, GNN_UNSUPERVISED, GNN_BASELINES,
    GNN_SUPERVISED, ALL_GNN, GNN_EPOCHS, GNN_SEEDS, OUTPUT_DIR, RESULTS_DIR,
    NETWORK_ORDER, GO_GAF_PATH,
)

DATA_DIR = OUTPUT_DIR
ALGO_TIMEOUT = 600  # seconds per algorithm (10 min)

FEATURE_SETTINGS = ['real', 'structural', 'random']
STRUCTURAL_FEATURE_DIM = 16


class _Timeout(BaseException):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("Algorithm timed out")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_bio_network(net_id):
    """
    Load a preprocessed biological network.

    Returns:
        G: NetworkX graph (undirected, weighted)
        features: np.ndarray (num_nodes, feature_dim)
        node_list: list of original gene/protein IDs
        metadata: dict
    """
    net_dir = DATA_DIR / net_id
    metadata = json.load(open(net_dir / 'metadata.json'))
    node_list = json.load(open(net_dir / 'node_list.json'))
    features = np.load(str(net_dir / 'features.npy'))
    edges = np.load(str(net_dir / 'edges.npy'))

    G = nx.Graph()
    G.add_nodes_from(range(len(node_list)))
    for row in edges:
        # Use abs(weight) for community detection; raw signed weights
        # are preserved in edges.npy for sign-coherence analysis (Phase 3).
        G.add_edge(int(row['src']), int(row['tgt']),
                   weight=abs(float(row['weight'])))

    return G, features, node_list, metadata


def get_louvain_k(G):
    """Run Louvain to determine a proxy K for GNN methods."""
    import community as community_louvain
    partition = community_louvain.best_partition(G)
    k = len(set(partition.values()))
    return k


def generate_structural_features(G, dim=STRUCTURAL_FEATURE_DIM):
    """Generate structural features: degree, clustering coeff, PageRank, spectral PE."""
    n = G.number_of_nodes()
    features = np.zeros((n, dim), dtype=np.float32)
    nodes = list(G.nodes())

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    for node in nodes:
        features[node, 0] = degrees[node] / max_deg

    clustering = nx.clustering(G)
    for node in nodes:
        features[node, 1] = clustering[node]

    try:
        pagerank = nx.pagerank(G, max_iter=100)
        max_pr = max(pagerank.values()) or 1.0
        for node in nodes:
            features[node, 2] = pagerank[node] / max_pr
    except Exception:
        pass

    # Spectral positional encoding (remaining dims)
    remaining = dim - 3
    if remaining > 0:
        try:
            import scipy.sparse as sp
            from scipy.sparse.linalg import eigsh
            L = nx.laplacian_matrix(G).astype(float)
            k_eig = min(remaining, n - 2)
            if k_eig > 0:
                _, eigvecs = eigsh(L, k=k_eig, which='SM')
                features[:, 3:3+k_eig] = eigvecs.astype(np.float32)
        except Exception:
            pass

    return features


def generate_random_features(n, dim):
    """Generate random Gaussian features matching a given dimensionality."""
    return np.random.randn(n, dim).astype(np.float32)


def compute_modularity(G, communities):
    """Compute modularity of a partition."""
    if not communities:
        return 0.0
    comm_sets = [set(c) for c in communities if len(c) > 0]
    if not comm_sets:
        return 0.0
    try:
        return nx.community.modularity(G, comm_sets)
    except Exception:
        return 0.0


# =============================================================================
# GO SLIM PSEUDO-LABELS (for semi-supervised GCN/GAT)
# =============================================================================

def build_go_slim_labels(net_id, node_list):
    """
    Build pseudo-labels from GO Slim categories for semi-supervised GNNs.
    Uses the GO annotations already loaded during Phase 1.

    Returns:
        labels: np.ndarray of int labels (one per node, -1 if no annotation)
        num_classes: int
    """
    import gzip
    from collections import defaultdict, Counter

    gaf_path = str(GO_GAF_PATH)
    meta = json.load(open(DATA_DIR / net_id / 'metadata.json'))
    id_format = meta['id_format']

    # GO Slim Biological Process categories (top-level)
    # Using a curated set of broad GO terms
    go_slim_bp = {
        'GO:0008152': 'metabolism',
        'GO:0009987': 'cellular_process',
        'GO:0050896': 'response_to_stimulus',
        'GO:0065007': 'biological_regulation',
        'GO:0032502': 'developmental_process',
        'GO:0051179': 'localization',
        'GO:0023052': 'signaling',
        'GO:0006950': 'stress_response',
        'GO:0002376': 'immune_process',
        'GO:0007049': 'cell_cycle',
        'GO:0012501': 'programmed_cell_death',
        'GO:0007155': 'cell_adhesion',
        'GO:0048870': 'cell_motility',
        'GO:0030154': 'cell_differentiation',
        'GO:0019725': 'cellular_homeostasis',
    }

    # Collect GO terms per gene from GAF
    gene_to_terms = defaultdict(set)
    id_col = 1 if id_format == 'uniprot' else 2  # UniProt accession vs symbol

    with gzip.open(gaf_path, 'rt') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 7:
                continue
            gene_id = parts[id_col]
            go_term = parts[4]
            aspect = parts[8]  # P = Biological Process
            if aspect == 'P':
                gene_to_terms[gene_id].add(go_term)

    # Map each gene to its GO Slim category
    node_set = set(node_list)
    gene_to_slim = {}
    for gene in node_list:
        terms = gene_to_terms.get(gene, set())
        # Find which slim categories this gene's terms match
        slim_hits = []
        for term in terms:
            if term in go_slim_bp:
                slim_hits.append(term)
        if slim_hits:
            # Pick most common slim category (arbitrary tiebreak)
            gene_to_slim[gene] = slim_hits[0]

    # Convert to numeric labels
    slim_terms_used = sorted(set(gene_to_slim.values()))
    slim_to_idx = {t: i for i, t in enumerate(slim_terms_used)}

    labels = np.full(len(node_list), -1, dtype=np.int64)
    for i, gene in enumerate(node_list):
        if gene in gene_to_slim:
            labels[i] = slim_to_idx[gene_to_slim[gene]]

    labeled_count = (labels >= 0).sum()
    num_classes = len(slim_terms_used)
    print(f"  GO Slim labels: {labeled_count}/{len(node_list)} labeled "
          f"({100*labeled_count/len(node_list):.1f}%), {num_classes} classes", flush=True)

    return labels, num_classes


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def _print_classical_summary(csv_path):
    """Print a summary table of classical results across all networks."""
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 90, flush=True)
    print("CLASSICAL BENCHMARK SUMMARY", flush=True)
    print("=" * 90, flush=True)

    net_order = [n for n in NETWORK_ORDER if n in df['network'].unique()]
    algos = df['algorithm'].unique()

    # Per-network summary
    for net_id in net_order:
        ndf = df[df['network'] == net_id].copy()
        ok = ndf[ndf['success'] == True].sort_values('modularity', ascending=False)
        fail = ndf[ndf['success'] == False]
        total_time = ndf['runtime'].sum()

        print(f"\n--- {net_id} ({len(ok)} OK, {len(fail)} failed, "
              f"total {total_time:.0f}s / {total_time/60:.1f} min) ---", flush=True)
        print(f"  {'Algorithm':<22} {'K':>6} {'Q':>8} {'Runtime':>10}", flush=True)
        print(f"  {'-'*48}", flush=True)
        for _, r in ok.iterrows():
            print(f"  {r['algorithm']:<22} {int(r['num_communities']):>6} "
                  f"{r['modularity']:>8.4f} {r['runtime']:>9.1f}s", flush=True)
        for _, r in fail.iterrows():
            err = str(r.get('error', 'fail'))[:30]
            print(f"  {r['algorithm']:<22} {'FAIL':>6} {'':>8} "
                  f"{r['runtime']:>9.1f}s  ({err})", flush=True)

    # Cross-network top-5 leaderboard
    print(f"\n{'='*90}", flush=True)
    print("TOP 5 PER NETWORK (by modularity)", flush=True)
    print(f"{'='*90}", flush=True)
    header = f"  {'Rank':<5}"
    for net_id in net_order:
        header += f" {net_id:>20}"
    print(header, flush=True)
    print(f"  {'-'*(5 + 21*len(net_order))}", flush=True)

    for rank in range(5):
        row = f"  {rank+1:<5}"
        for net_id in net_order:
            ndf = df[(df['network'] == net_id) & (df['success'] == True)]
            ndf = ndf.sort_values('modularity', ascending=False)
            if rank < len(ndf):
                r = ndf.iloc[rank]
                row += f" {r['algorithm'][:14]:>14} {r['modularity']:.3f}"
            else:
                row += f" {'':>20}"
        print(row, flush=True)


def run_classical_benchmark(networks, resume=False):
    """Run all classical algorithms on specified networks.

    Uses adaptive pruning: algorithms that timeout or fail on a smaller
    network are automatically skipped on larger ones.
    """
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'bio_classical_results.csv'
    partitions_dir = results_dir / 'partitions'
    partitions_dir.mkdir(parents=True, exist_ok=True)

    csv_cols = ['network', 'algorithm', 'algorithm_type', 'seed',
                'success', 'runtime', 'num_communities', 'modularity', 'error']

    def _append_row(row_dict):
        """Append a single result row to the CSV immediately."""
        row_dict.setdefault('error', '')
        row = pd.DataFrame([row_dict], columns=csv_cols)
        header = not csv_path.is_file()
        row.to_csv(csv_path, mode='a', header=header, index=False)

    existing_pairs = set()
    pruned_algos = set()
    if resume:
        if csv_path.is_file():
            try:
                df = pd.read_csv(csv_path)
                for _, row in df[df['success'] == True].iterrows():
                    existing_pairs.add((row['network'], row['algorithm']))
                for _, row in df[df['success'] == False].iterrows():
                    err = str(row.get('error', ''))
                    if 'timeout' in err.lower():
                        pruned_algos.add(row['algorithm'])
            except Exception:
                pass
        # Recover partition files not yet in CSV (crash recovery)
        if partitions_dir.is_dir():
            recovered = 0
            for pf in partitions_dir.glob('*.json'):
                name = pf.stem
                for net_id_check in NETWORK_ORDER:
                    if name.startswith(net_id_check + '_'):
                        algo = name[len(net_id_check) + 1:]
                        if algo.endswith(tuple(f'_s{i}' for i in range(10))):
                            break
                        pair = (net_id_check, algo)
                        if pair not in existing_pairs:
                            try:
                                comms = json.load(open(str(pf)))
                                G_tmp, _, _, _ = load_bio_network(net_id_check)
                                mod = compute_modularity(G_tmp, comms)
                                _append_row({
                                    'network': net_id_check, 'algorithm': algo,
                                    'algorithm_type': 'classical', 'seed': 0,
                                    'success': True, 'runtime': -1,
                                    'num_communities': len(comms), 'modularity': mod,
                                })
                                recovered += 1
                            except Exception as e:
                                print(f"  Recovery failed for {name}: {e}",
                                      flush=True)
                            existing_pairs.add(pair)
                        break
            if recovered:
                print(f"  Recovered {recovered} partitions from crash",
                      flush=True)
        if existing_pairs:
            print(f"  Resume: {len(existing_pairs)} (network, algo) pairs cached",
                  flush=True)
        if pruned_algos:
            print(f"  Resume: pruned from previous timeouts: {sorted(pruned_algos)}",
                  flush=True)

    runner = AlgorithmRunner(verbose=False)

    for net_id in networks:
        print(f"\n{'='*60}", flush=True)
        print(f"Classical benchmark: {net_id}", flush=True)
        print(f"{'='*60}", flush=True)

        G, features, node_list, metadata = load_bio_network(net_id)
        print(f"  Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges",
              flush=True)
        if pruned_algos:
            print(f"  Pruned from previous networks: {sorted(pruned_algos)}", flush=True)

        num_edges = G.number_of_edges()
        if num_edges > MEMSAFE_THRESHOLD:
            algo_list = CLASSICAL_ALGORITHMS_MEMSAFE
            print(f"  Using memory-safe algorithm list ({num_edges} edges > {MEMSAFE_THRESHOLD})",
                  flush=True)
        else:
            algo_list = CLASSICAL_ALGORITHMS
        active_algos = [a for a in algo_list if a not in pruned_algos]
        net_start = time.time()

        for algo_name in active_algos:
            if resume and (net_id, algo_name) in existing_pairs:
                print(f"  {algo_name}... (skipped, cached)", flush=True)
                continue

            print(f"  Running {algo_name}...", end=' ', flush=True)

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(ALGO_TIMEOUT)
            timed_out = False
            try:
                result = runner.run_algorithm(G, algo_name)
            except _Timeout:
                timed_out = True
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                print(f"TIMEOUT ({ALGO_TIMEOUT}s) — pruned for larger networks",
                      flush=True)
                pruned_algos.add(algo_name)
                _append_row({
                    'network': net_id, 'algorithm': algo_name,
                    'algorithm_type': 'classical', 'seed': 0,
                    'success': False, 'runtime': ALGO_TIMEOUT,
                    'num_communities': 0, 'modularity': 0.0,
                    'error': f'timeout ({ALGO_TIMEOUT}s)',
                })
                continue
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            if result.success and result.communities:
                mod = compute_modularity(G, result.communities)
                nc = len(result.communities)
                print(f"OK  K={nc}, Q={mod:.3f} ({result.runtime:.1f}s)", flush=True)

                part_file = partitions_dir / f"{net_id}_{algo_name}.json"
                with open(str(part_file), 'w') as f:
                    json.dump(result.communities, f)

                _append_row({
                    'network': net_id, 'algorithm': algo_name,
                    'algorithm_type': 'classical', 'seed': 0,
                    'success': True, 'runtime': result.runtime,
                    'num_communities': nc, 'modularity': mod,
                })
            else:
                err = result.error_message or 'unknown error'
                print(f"FAIL  ({err[:60]})", flush=True)
                _append_row({
                    'network': net_id, 'algorithm': algo_name,
                    'algorithm_type': 'classical', 'seed': 0,
                    'success': False, 'runtime': result.runtime,
                    'num_communities': 0, 'modularity': 0.0,
                    'error': err,
                })

        net_elapsed = time.time() - net_start
        print(f"\n  {net_id} done in {net_elapsed:.0f}s ({net_elapsed/60:.1f} min)",
              flush=True)

    print(f"\nClassical results saved to {csv_path}", flush=True)
    if pruned_algos:
        print(f"Pruned algorithms (timeout/fail): {sorted(pruned_algos)}", flush=True)

    # Print summary table
    _print_classical_summary(csv_path)


def run_gnn_benchmark(networks, epochs=600, seeds=3, resume=False,
                      feature_ablation=False, save_embeddings=False,
                      save_attention=False):
    """Run all GNN algorithms on specified networks.

    When feature_ablation=True, each GNN runs with 3 feature types:
      - 'real'       : original bio features (expression PCA / GO binary)
      - 'structural' : degree + clustering coeff + pagerank + spectral PE (16-dim)
      - 'random'     : Gaussian noise matching real feature dimensionality

    When save_embeddings=True, latent vectors are saved as .npy after training.
    When save_attention=True, GAT attention weights are exported per edge.
    """
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'bio_gnn_results.csv'
    if feature_ablation:
        csv_path = results_dir / 'bio_gnn_ablation_results.csv'
    partitions_dir = results_dir / 'partitions'
    partitions_dir.mkdir(parents=True, exist_ok=True)

    if save_embeddings:
        embeddings_dir = results_dir / 'embeddings'
        embeddings_dir.mkdir(parents=True, exist_ok=True)
    if save_attention:
        attention_dir = results_dir / 'attention'
        attention_dir.mkdir(parents=True, exist_ok=True)

    existing_pairs = set()
    if resume and csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            for _, row in df[df['success'] == True].iterrows():
                ft = row.get('feature_type', 'real')
                existing_pairs.add((row['network'], row['algorithm'],
                                    ft, int(row['seed'])))
        except Exception:
            pass

    runner = AlgorithmRunner(verbose=False)
    all_results = []

    feat_types = FEATURE_SETTINGS if feature_ablation else ['real']

    for net_id in networks:
        print(f"\n{'='*60}", flush=True)
        print(f"GNN benchmark: {net_id}", flush=True)
        print(f"{'='*60}", flush=True)

        G, features_real, node_list, metadata = load_bio_network(net_id)
        n = G.number_of_nodes()
        feat_dim = features_real.shape[1]
        print(f"  Loaded: {n} nodes, {G.number_of_edges()} edges, "
              f"{feat_dim} feature dims", flush=True)

        # Pre-compute feature variants
        features_dict = {'real': features_real}
        if feature_ablation:
            print(f"  Generating structural features ({STRUCTURAL_FEATURE_DIM}-dim)...",
                  flush=True)
            features_dict['structural'] = generate_structural_features(
                G, dim=STRUCTURAL_FEATURE_DIM)
            # Random features match real dim for fair comparison
            features_dict['random'] = generate_random_features(n, feat_dim)

        louvain_k = get_louvain_k(G)
        print(f"  Louvain proxy K = {louvain_k}", flush=True)

        # GO Slim labels for semi-supervised methods
        go_labels, num_go_classes = build_go_slim_labels(net_id, node_list)
        labeled_mask = go_labels >= 0

        all_gnn = GNN_UNSUPERVISED + GNN_BASELINES + GNN_SUPERVISED

        for feat_type in feat_types:
            feats = features_dict[feat_type]
            print(f"\n  --- Feature type: {feat_type} "
                  f"(dim={feats.shape[1]}) ---", flush=True)

            for seed in range(seeds):
                print(f"\n  --- Seed {seed} ---", flush=True)
                np.random.seed(seed)

                rng = np.random.RandomState(seed)
                train_mask = np.zeros(n, dtype=bool)
                labeled_indices = np.where(labeled_mask)[0]
                if len(labeled_indices) > 0:
                    train_size = max(1, int(0.2 * len(labeled_indices)))
                    train_idx = rng.choice(labeled_indices, size=train_size,
                                           replace=False)
                    train_mask[train_idx] = True

                for algo_name in all_gnn:
                    key = (net_id, algo_name, feat_type, seed)
                    if resume and key in existing_pairs:
                        print(f"  {algo_name} ({feat_type}, seed {seed})... "
                              f"(skipped, cached)", flush=True)
                        continue

                    print(f"  Running {algo_name} ({feat_type}, seed {seed})...",
                          end=' ', flush=True)

                    kwargs = {
                        'num_clusters': louvain_k,
                        'epochs': epochs,
                        'features': feats,
                    }

                    if algo_name in GNN_SUPERVISED:
                        if num_go_classes < 2:
                            print(f"SKIP (need >=2 GO Slim classes)", flush=True)
                            all_results.append({
                                'network': net_id, 'algorithm': algo_name,
                                'algorithm_type': 'gnn_supervised',
                                'feature_type': feat_type, 'seed': seed,
                                'success': False, 'runtime': 0.0,
                                'num_communities': 0, 'modularity': 0.0,
                                'error': 'insufficient GO Slim classes',
                            })
                            continue
                        kwargs['labels'] = go_labels
                        kwargs['train_mask'] = train_mask
                        kwargs['num_clusters'] = num_go_classes

                    result = runner.run_algorithm(G, algo_name, **kwargs)

                    if result.success and result.communities:
                        mod = compute_modularity(G, result.communities)
                        nc = len(result.communities)
                        print(f"OK  K={nc}, Q={mod:.3f} ({result.runtime:.1f}s)",
                              flush=True)

                        # Save partition
                        suffix = f"_{feat_type}" if feature_ablation else ""
                        part_file = (partitions_dir /
                                     f"{net_id}_{algo_name}{suffix}_s{seed}.json")
                        with open(str(part_file), 'w') as f:
                            json.dump(result.communities, f)

                        # Save embeddings if requested
                        if save_embeddings and hasattr(result, 'model_instance'):
                            try:
                                model_inst = result.model_instance
                                emb = model_inst.get_embeddings()
                                emb_file = (embeddings_dir /
                                            f"{net_id}_{algo_name}{suffix}_s{seed}.npy")
                                np.save(str(emb_file), emb)
                                print(f"    -> Embeddings saved: {emb.shape}",
                                      flush=True)
                            except Exception as e:
                                print(f"    -> Embedding export failed: {e}",
                                      flush=True)

                        # Save attention weights (GAT only)
                        if (save_attention and 'gat' in algo_name
                                and hasattr(result, 'model_instance')):
                            try:
                                import torch
                                model_inst = result.model_instance
                                model_inst.model.eval()
                                data = model_inst.data
                                attn_weights = model_inst.model.get_attention_weights(
                                    data.x, data.edge_index)
                                attn_data = {
                                    'edge_index': data.edge_index.cpu().numpy().tolist(),
                                    'attention_weights': [
                                        a.cpu().numpy().tolist()
                                        for a in attn_weights
                                    ],
                                }
                                attn_file = (attention_dir /
                                             f"{net_id}_{algo_name}{suffix}_s{seed}.json")
                                with open(str(attn_file), 'w') as f:
                                    json.dump(attn_data, f)
                                print(f"    -> Attention weights saved",
                                      flush=True)
                            except Exception as e:
                                print(f"    -> Attention export failed: {e}",
                                      flush=True)

                        algo_type = ('gnn_supervised' if algo_name in GNN_SUPERVISED
                                     else ('gnn_baseline' if algo_name in GNN_BASELINES
                                           else 'gnn_unsupervised'))

                        all_results.append({
                            'network': net_id, 'algorithm': algo_name,
                            'algorithm_type': algo_type,
                            'feature_type': feat_type, 'seed': seed,
                            'success': True, 'runtime': result.runtime,
                            'num_communities': nc, 'modularity': mod,
                        })
                    else:
                        err = result.error_message or 'unknown error'
                        print(f"FAIL  ({err[:60]})", flush=True)
                        all_results.append({
                            'network': net_id, 'algorithm': algo_name,
                            'algorithm_type': 'gnn',
                            'feature_type': feat_type, 'seed': seed,
                            'success': False, 'runtime': result.runtime,
                            'num_communities': 0, 'modularity': 0.0,
                            'error': err,
                        })

    # Save results
    df_new = pd.DataFrame(all_results)
    if resume and csv_path.is_file():
        df_old = pd.read_csv(csv_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_path, index=False)
    print(f"\nGNN results saved to {csv_path}", flush=True)


def run_embedding_export(networks, epochs=600, seeds=1):
    """Run selected GNN models and export latent embeddings.

    Saves n x d numpy arrays to results/embeddings/ for downstream
    t-SNE / UMAP visualization.
    """
    from clusternet.gnn.models import (
        DMoNCluster, VGAECluster, DGICluster, GATSupervised,
    )
    results_dir = RESULTS_DIR
    embeddings_dir = results_dir / 'embeddings'
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    MODELS = {
        'dgi_gnn': DGICluster,
        'dmon_gnn': DMoNCluster,
        'vgae_gnn': VGAECluster,
    }

    for net_id in networks:
        print(f"\n{'='*60}", flush=True)
        print(f"Embedding export: {net_id}", flush=True)
        print(f"{'='*60}", flush=True)

        G, features, node_list, metadata = load_bio_network(net_id)
        n = G.number_of_nodes()
        louvain_k = get_louvain_k(G)
        print(f"  Loaded: {n} nodes, K={louvain_k}", flush=True)

        for algo_name, ModelClass in MODELS.items():
            for seed in range(seeds):
                print(f"  {algo_name} (seed {seed})...", end=' ', flush=True)
                np.random.seed(seed)
                try:
                    import torch
                    torch.manual_seed(seed)
                    algo = ModelClass(
                        G, num_clusters=louvain_k,
                        epochs=epochs, hidden_channels=64,
                        features=features,
                    )
                    algo.train()
                    algo.model.eval()
                    emb = algo.get_embeddings()
                    out_file = embeddings_dir / f"{net_id}_{algo_name}_s{seed}.npy"
                    np.save(str(out_file), emb)
                    print(f"OK  shape={emb.shape}", flush=True)
                except Exception as e:
                    print(f"FAIL ({e})", flush=True)

        # GAT supervised (needs labels)
        go_labels, num_go_classes = build_go_slim_labels(net_id, node_list)
        if num_go_classes >= 2:
            for seed in range(seeds):
                print(f"  gat_supervised (seed {seed})...", end=' ', flush=True)
                np.random.seed(seed)
                try:
                    import torch
                    torch.manual_seed(seed)
                    labeled_mask = go_labels >= 0
                    rng = np.random.RandomState(seed)
                    train_mask = np.zeros(n, dtype=bool)
                    labeled_indices = np.where(labeled_mask)[0]
                    train_size = max(1, int(0.2 * len(labeled_indices)))
                    train_idx = rng.choice(labeled_indices, size=train_size,
                                           replace=False)
                    train_mask[train_idx] = True

                    algo = GATSupervised(
                        G, num_clusters=num_go_classes, labels=go_labels,
                        train_mask=train_mask, epochs=epochs,
                        hidden_channels=8, features=features,
                    )
                    algo.train()
                    algo.model.eval()
                    emb = algo.get_embeddings()
                    out_file = embeddings_dir / f"{net_id}_gat_supervised_s{seed}.npy"
                    np.save(str(out_file), emb)
                    print(f"OK  shape={emb.shape}", flush=True)
                except Exception as e:
                    print(f"FAIL ({e})", flush=True)

    print(f"\nEmbeddings saved to {embeddings_dir}", flush=True)


def run_attention_export(networks, epochs=600, seeds=3):
    """Export GAT attention weights for biological interpretation.

    For each network and seed, trains GAT and saves per-edge attention scores
    along with the edge index for downstream pathway overlap analysis.
    """
    from clusternet.gnn.models import GATSupervised
    results_dir = RESULTS_DIR
    attention_dir = results_dir / 'attention'
    attention_dir.mkdir(parents=True, exist_ok=True)

    for net_id in networks:
        print(f"\n{'='*60}", flush=True)
        print(f"Attention export: {net_id}", flush=True)
        print(f"{'='*60}", flush=True)

        G, features, node_list, metadata = load_bio_network(net_id)
        n = G.number_of_nodes()
        go_labels, num_go_classes = build_go_slim_labels(net_id, node_list)
        if num_go_classes < 2:
            print(f"  SKIP (need >=2 GO classes, got {num_go_classes})", flush=True)
            continue

        labeled_mask = go_labels >= 0

        for seed in range(seeds):
            print(f"  GAT attention (seed {seed})...", end=' ', flush=True)
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
                rng = np.random.RandomState(seed)
                train_mask = np.zeros(n, dtype=bool)
                labeled_indices = np.where(labeled_mask)[0]
                train_size = max(1, int(0.2 * len(labeled_indices)))
                train_idx = rng.choice(labeled_indices, size=train_size,
                                       replace=False)
                train_mask[train_idx] = True

                algo = GATSupervised(
                    G, num_clusters=num_go_classes, labels=go_labels,
                    train_mask=train_mask, epochs=epochs,
                    hidden_channels=8, features=features,
                )
                algo.train()
                algo.model.eval()

                data = algo.data
                with torch.no_grad():
                    attn_weights = algo.model.get_attention_weights(
                        data.x, data.edge_index)

                attn_data = {
                    'edge_index': data.edge_index.cpu().numpy().tolist(),
                    'node_list': node_list,
                    'attention_weights': [
                        a.cpu().numpy().tolist() for a in attn_weights
                    ],
                    'network': net_id,
                    'seed': seed,
                    'num_heads': algo.model.convs[0].heads if hasattr(algo.model.convs[0], 'heads') else 1,
                }
                out_file = attention_dir / f"{net_id}_gat_s{seed}.json"
                with open(str(out_file), 'w') as f:
                    json.dump(attn_data, f)
                print(f"OK  edges={data.edge_index.shape[1]}", flush=True)
            except Exception as e:
                print(f"FAIL ({e})", flush=True)

    print(f"\nAttention data saved to {attention_dir}", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Biological Network Benchmark')
    parser.add_argument('--run-classical', action='store_true',
                        help='Run classical algorithms')
    parser.add_argument('--run-gnn', action='store_true',
                        help='Run GNN algorithms')
    parser.add_argument('--feature-ablation', action='store_true',
                        help='Run GNNs with all 3 feature types (real, structural, random)')
    parser.add_argument('--save-embeddings', action='store_true',
                        help='Save GNN latent embeddings as .npy files')
    parser.add_argument('--save-attention', action='store_true',
                        help='Save GAT attention weights per edge')
    parser.add_argument('--network', type=str, default=None,
                        help='Run on specific network (gtex_0.9, grn, biogrid, signor)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run (skip completed pairs)')
    parser.add_argument('--epochs', type=int, default=GNN_EPOCHS,
                        help=f'GNN training epochs (default: {GNN_EPOCHS})')
    parser.add_argument('--seeds', type=int, default=GNN_SEEDS,
                        help=f'Number of random seeds for GNNs (default: {GNN_SEEDS})')

    args = parser.parse_args()

    if not args.run_classical and not args.run_gnn:
        print("Specify --run-classical and/or --run-gnn")
        sys.exit(1)

    if args.network:
        if args.network not in NETWORKS:
            print(f"Unknown network: {args.network}. "
                  f"Available: {list(NETWORKS.keys())}")
            sys.exit(1)
        networks = [args.network]
    else:
        networks = [n for n in NETWORK_ORDER if n in NETWORKS]

    print(f"Networks: {networks}", flush=True)
    print(f"Classical: {args.run_classical}, GNN: {args.run_gnn}", flush=True)
    if args.run_gnn:
        print(f"Epochs: {args.epochs}, Seeds: {args.seeds}, "
              f"Feature ablation: {args.feature_ablation}, "
              f"Save embeddings: {args.save_embeddings}, "
              f"Save attention: {args.save_attention}", flush=True)

    t0 = time.time()

    if args.run_classical:
        run_classical_benchmark(networks, resume=args.resume)

    if args.run_gnn:
        run_gnn_benchmark(networks, epochs=args.epochs,
                          seeds=args.seeds, resume=args.resume,
                          feature_ablation=args.feature_ablation,
                          save_embeddings=args.save_embeddings,
                          save_attention=args.save_attention)

    if args.save_embeddings and not args.run_gnn:
        run_embedding_export(networks, epochs=args.epochs, seeds=args.seeds)

    if args.save_attention and not args.run_gnn:
        run_attention_export(networks, epochs=args.epochs, seeds=args.seeds)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
