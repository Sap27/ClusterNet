"""
Phase 2: Social Network Benchmark Runner.

Two experiment sets:
  A) Community Detection Sweep — all classical + GNN algorithms, AMI against
     ground-truth labels, modularity, runtime.
  B) Feature Ablation (GNN only) — run each GNN with three feature settings:
     (real, structural, random) to quantify when node attributes help.

Usage:
    python run_benchmark.py --run-classical                         # classical sweep
    python run_benchmark.py --run-gnn                               # GNN sweep (real features)
    python run_benchmark.py --run-gnn --feature-ablation            # GNN sweep (all 3 feature types)
    python run_benchmark.py --run-classical --run-gnn --resume      # resume everything
    python run_benchmark.py --network cora --run-classical --run-gnn  # single network
"""
import os, sys, json, time, argparse, signal
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

# Local config first (before adding LFR to path, which has its own config.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    NETWORKS, NETWORK_ORDER, CLASSICAL_ALGORITHMS,
    GNN_UNSUPERVISED, GNN_BASELINES, GNN_SUPERVISED, ALL_GNN,
    GNN_EPOCHS, GNN_SEEDS, OUTPUT_DIR, RESULTS_DIR, ALGO_TIMEOUT,
    FEATURE_SETTINGS,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lfr'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithm_runner import AlgorithmRunner

DATA_DIR = OUTPUT_DIR


class _Timeout(BaseException):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout("Algorithm timed out")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_social_network(net_id):
    """Load a preprocessed social network.

    Returns:
        G: NetworkX graph
        features: dict of {setting_name: np.ndarray}
        labels: np.ndarray of ground-truth labels
        metadata: dict
    """
    net_dir = DATA_DIR / net_id
    metadata = json.load(open(net_dir / 'metadata.json'))
    node_list = json.load(open(net_dir / 'node_list.json'))
    edges = np.load(str(net_dir / 'edges.npy'))
    labels = np.load(str(net_dir / 'labels.npy'))

    features = {
        'real': np.load(str(net_dir / 'features.npy')),
        'structural': np.load(str(net_dir / 'features_structural.npy')),
        'random': np.load(str(net_dir / 'features_random.npy')),
    }

    G = nx.Graph()
    G.add_nodes_from(range(len(node_list)))
    for row in edges:
        G.add_edge(int(row['src']), int(row['tgt']),
                   weight=float(row['weight']))

    return G, features, labels, metadata


def get_louvain_k(G):
    """Run Louvain to determine a proxy K for GNN methods."""
    import community as community_louvain
    partition = community_louvain.best_partition(G)
    return len(set(partition.values()))


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


def communities_to_labels(communities, n):
    """Convert list-of-lists partition to a flat label array."""
    labels = np.full(n, -1, dtype=np.int64)
    for cid, members in enumerate(communities):
        for node in members:
            labels[node] = cid
    return labels


def compute_ami(true_labels, communities, n):
    """AMI between ground-truth labels and detected communities."""
    pred = communities_to_labels(communities, n)
    mask = (true_labels >= 0) & (pred >= 0)
    if mask.sum() < 10:
        return float('nan')
    return adjusted_mutual_info_score(true_labels[mask], pred[mask])


def compute_nmi(true_labels, communities, n):
    """NMI between ground-truth labels and detected communities."""
    pred = communities_to_labels(communities, n)
    mask = (true_labels >= 0) & (pred >= 0)
    if mask.sum() < 10:
        return float('nan')
    return normalized_mutual_info_score(true_labels[mask], pred[mask])


# =============================================================================
# CLASSICAL BENCHMARK
# =============================================================================

def run_classical_benchmark(networks, resume=False):
    """Run all classical algorithms on specified social networks."""
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'social_classical_results.csv'
    partitions_dir = results_dir / 'partitions'
    partitions_dir.mkdir(parents=True, exist_ok=True)

    csv_cols = ['network', 'algorithm', 'algorithm_type', 'seed',
                'success', 'runtime', 'num_communities', 'modularity',
                'ami', 'nmi', 'error']

    def _append_row(row_dict):
        row_dict.setdefault('error', '')
        row_dict.setdefault('ami', float('nan'))
        row_dict.setdefault('nmi', float('nan'))
        row = pd.DataFrame([row_dict], columns=csv_cols)
        header = not csv_path.is_file()
        row.to_csv(csv_path, mode='a', header=header, index=False)

    existing_pairs = set()
    pruned_algos = set()
    if resume and csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            for _, row in df[df['success'] == True].iterrows():
                existing_pairs.add((row['network'], row['algorithm']))
        except Exception:
            pass

    runner = AlgorithmRunner(verbose=False)

    for net_id in networks:
        print(f"\n{'='*60}", flush=True)
        print(f"Classical benchmark: {net_id}", flush=True)
        print(f"{'='*60}", flush=True)

        G, features_dict, labels, metadata = load_social_network(net_id)
        n = G.number_of_nodes()
        print(f"  Loaded: {n} nodes, {G.number_of_edges()} edges, "
              f"{metadata['num_classes']} classes, homophily={metadata['edge_homophily']:.3f}",
              flush=True)

        net_start = time.time()
        for algo_name in CLASSICAL_ALGORITHMS:
            if algo_name in pruned_algos:
                continue
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
                print(f"TIMEOUT ({ALGO_TIMEOUT}s)", flush=True)
                pruned_algos.add(algo_name)
                _append_row({
                    'network': net_id, 'algorithm': algo_name,
                    'algorithm_type': 'classical', 'seed': 0,
                    'success': False, 'runtime': ALGO_TIMEOUT,
                    'num_communities': 0, 'modularity': 0.0,
                    'error': f'timeout ({ALGO_TIMEOUT}s)',
                })
                continue
            except Exception as e:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                print(f"EXCEPTION ({str(e)[:50]})", flush=True)
                _append_row({
                    'network': net_id, 'algorithm': algo_name,
                    'algorithm_type': 'classical', 'seed': 0,
                    'success': False, 'runtime': 0.0,
                    'num_communities': 0, 'modularity': 0.0,
                    'error': str(e)[:200],
                })
                continue
            finally:
                if not timed_out:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            if result.success and result.communities:
                mod = compute_modularity(G, result.communities)
                nc = len(result.communities)
                ami = compute_ami(labels, result.communities, n)
                nmi = compute_nmi(labels, result.communities, n)
                print(f"OK  K={nc}, Q={mod:.3f}, AMI={ami:.3f} ({result.runtime:.1f}s)",
                      flush=True)

                part_file = partitions_dir / f"{net_id}_{algo_name}.json"
                with open(str(part_file), 'w') as f:
                    json.dump(result.communities, f)

                _append_row({
                    'network': net_id, 'algorithm': algo_name,
                    'algorithm_type': 'classical', 'seed': 0,
                    'success': True, 'runtime': result.runtime,
                    'num_communities': nc, 'modularity': mod,
                    'ami': ami, 'nmi': nmi,
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
    _print_summary(csv_path, 'CLASSICAL')


# =============================================================================
# GNN BENCHMARK  (with feature ablation)
# =============================================================================

def run_gnn_benchmark(networks, epochs=600, seeds=3, resume=False,
                      feature_ablation=False):
    """Run GNN algorithms on social networks.

    When feature_ablation=True, each GNN is run three times per seed:
      - 'real'       : original node features (bag-of-words / product features)
      - 'structural' : degree + clustering coeff + pagerank + spectral embedding
      - 'random'     : Gaussian noise, same dim as real features
    """
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'social_gnn_results.csv'
    partitions_dir = results_dir / 'partitions'
    partitions_dir.mkdir(parents=True, exist_ok=True)

    csv_cols = ['network', 'algorithm', 'algorithm_type', 'feature_type',
                'seed', 'success', 'runtime', 'num_communities',
                'modularity', 'ami', 'nmi', 'error']

    existing_keys = set()
    if resume and csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            for _, row in df[df['success'] == True].iterrows():
                existing_keys.add((row['network'], row['algorithm'],
                                   row['feature_type'], int(row['seed'])))
        except Exception:
            pass

    runner = AlgorithmRunner(verbose=False)
    all_results = []

    feat_types = FEATURE_SETTINGS if feature_ablation else ['real']

    for net_id in networks:
        print(f"\n{'='*60}", flush=True)
        print(f"GNN benchmark: {net_id}", flush=True)
        print(f"{'='*60}", flush=True)

        G, features_dict, labels, metadata = load_social_network(net_id)
        n = G.number_of_nodes()
        print(f"  Loaded: {n} nodes, {G.number_of_edges()} edges, "
              f"{metadata['num_classes']} classes", flush=True)

        louvain_k = get_louvain_k(G)
        num_classes = metadata['num_classes']
        print(f"  Louvain proxy K = {louvain_k}, true K = {num_classes}", flush=True)

        for feat_type in feat_types:
            feats = features_dict[feat_type]
            print(f"\n  --- Feature type: {feat_type} "
                  f"(dim={feats.shape[1]}) ---", flush=True)

            for seed in range(seeds):
                print(f"\n  --- Seed {seed} ---", flush=True)
                rng = np.random.RandomState(seed)

                # Train mask for semi-supervised: 20% of nodes
                train_mask = np.zeros(n, dtype=bool)
                train_size = max(1, int(0.2 * n))
                train_idx = rng.choice(n, size=train_size, replace=False)
                train_mask[train_idx] = True

                for algo_name in ALL_GNN:
                    key = (net_id, algo_name, feat_type, seed)
                    if resume and key in existing_keys:
                        print(f"    {algo_name} ({feat_type}, s{seed})... "
                              f"(skipped, cached)", flush=True)
                        continue

                    print(f"    {algo_name} ({feat_type}, s{seed})...",
                          end=' ', flush=True)

                    kwargs = {
                        'num_clusters': louvain_k,
                        'epochs': epochs,
                        'features': feats,
                    }

                    if algo_name in GNN_SUPERVISED:
                        kwargs['labels'] = labels
                        kwargs['train_mask'] = train_mask
                        kwargs['num_clusters'] = num_classes

                    try:
                        result = runner.run_algorithm(G, algo_name, **kwargs)
                    except Exception as e:
                        print(f"EXCEPTION ({str(e)[:50]})", flush=True)
                        all_results.append({
                            'network': net_id, 'algorithm': algo_name,
                            'algorithm_type': 'gnn', 'feature_type': feat_type,
                            'seed': seed, 'success': False, 'runtime': 0.0,
                            'num_communities': 0, 'modularity': 0.0,
                            'ami': float('nan'), 'nmi': float('nan'),
                            'error': str(e)[:200],
                        })
                        continue

                    if result.success and result.communities:
                        mod = compute_modularity(G, result.communities)
                        nc = len(result.communities)
                        ami = compute_ami(labels, result.communities, n)
                        nmi = compute_nmi(labels, result.communities, n)
                        print(f"OK  K={nc}, Q={mod:.3f}, AMI={ami:.3f} "
                              f"({result.runtime:.1f}s)", flush=True)

                        suffix = f"_{feat_type}" if feature_ablation else ""
                        part_file = (partitions_dir /
                                     f"{net_id}_{algo_name}{suffix}_s{seed}.json")
                        with open(str(part_file), 'w') as f:
                            json.dump(result.communities, f)

                        algo_type = 'gnn_supervised' if algo_name in GNN_SUPERVISED \
                            else ('gnn_baseline' if algo_name in GNN_BASELINES
                                  else 'gnn_unsupervised')

                        all_results.append({
                            'network': net_id, 'algorithm': algo_name,
                            'algorithm_type': algo_type,
                            'feature_type': feat_type, 'seed': seed,
                            'success': True, 'runtime': result.runtime,
                            'num_communities': nc, 'modularity': mod,
                            'ami': ami, 'nmi': nmi, 'error': '',
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
                            'ami': float('nan'), 'nmi': float('nan'),
                            'error': err,
                        })

    # Save results
    df_new = pd.DataFrame(all_results, columns=csv_cols)
    if resume and csv_path.is_file():
        df_old = pd.read_csv(csv_path)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(csv_path, index=False)
    print(f"\nGNN results saved to {csv_path}", flush=True)
    _print_summary(csv_path, 'GNN')


# =============================================================================
# SUMMARY
# =============================================================================

def _print_summary(csv_path, label):
    """Print a summary table of results."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return

    ok = df[df['success'] == True].copy()
    if ok.empty:
        print(f"  No successful results to summarize.", flush=True)
        return

    print(f"\n{'='*90}", flush=True)
    print(f"{label} BENCHMARK SUMMARY", flush=True)
    print(f"{'='*90}", flush=True)

    for net_id in NETWORK_ORDER:
        ndf = ok[ok['network'] == net_id]
        if ndf.empty:
            continue

        if 'feature_type' in ndf.columns:
            groups = ndf.groupby('feature_type') if ndf['feature_type'].nunique() > 1 \
                else [('all', ndf)]
        else:
            groups = [('all', ndf)]

        for feat_label, gdf in groups:
            # Average over seeds
            avg = gdf.groupby('algorithm').agg({
                'ami': 'mean', 'nmi': 'mean', 'modularity': 'mean',
                'num_communities': 'mean', 'runtime': 'mean',
            }).sort_values('ami', ascending=False)

            tag = f" [{feat_label}]" if feat_label != 'all' else ""
            print(f"\n--- {net_id}{tag} (top 10 by AMI) ---", flush=True)
            print(f"  {'Algorithm':<22} {'AMI':>7} {'NMI':>7} {'Q':>7} "
                  f"{'K':>6} {'Time':>8}", flush=True)
            print(f"  {'-'*60}", flush=True)
            for algo, r in avg.head(10).iterrows():
                print(f"  {algo:<22} {r['ami']:>7.3f} {r['nmi']:>7.3f} "
                      f"{r['modularity']:>7.3f} {r['num_communities']:>6.0f} "
                      f"{r['runtime']:>7.1f}s", flush=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Social Network Benchmark')
    parser.add_argument('--run-classical', action='store_true',
                        help='Run classical algorithm sweep')
    parser.add_argument('--run-gnn', action='store_true',
                        help='Run GNN algorithm sweep')
    parser.add_argument('--feature-ablation', action='store_true',
                        help='Run GNNs with all 3 feature types (real, structural, random)')
    parser.add_argument('--network', type=str, default=None,
                        help='Run on specific network (cora, citeseer, amazon_photo)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run')
    parser.add_argument('--epochs', type=int, default=GNN_EPOCHS,
                        help=f'GNN training epochs (default: {GNN_EPOCHS})')
    parser.add_argument('--seeds', type=int, default=GNN_SEEDS,
                        help=f'Number of random seeds (default: {GNN_SEEDS})')

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
        networks = NETWORK_ORDER

    print(f"Networks: {networks}", flush=True)
    print(f"Classical: {args.run_classical}, GNN: {args.run_gnn}", flush=True)
    if args.run_gnn:
        print(f"Epochs: {args.epochs}, Seeds: {args.seeds}, "
              f"Feature ablation: {args.feature_ablation}", flush=True)

    t0 = time.time()

    if args.run_classical:
        run_classical_benchmark(networks, resume=args.resume)

    if args.run_gnn:
        run_gnn_benchmark(networks, epochs=args.epochs, seeds=args.seeds,
                          resume=args.resume,
                          feature_ablation=args.feature_ablation)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)


if __name__ == '__main__':
    main()
