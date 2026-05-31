"""
GRN Perturbation Analysis Runner.

Applies 4 perturbation tests to the GRN network, reruns community detection,
and measures partition stability against the unperturbed baseline.

Tests:
  1. TF knockout     — remove top-5/top-10 highest out-degree TFs
  2. Edge dropout    — randomly remove 5%/10%/20%/30% of edges
  3. Hub gene removal — remove top 1%/2%/5% genes by degree
  4. TF knockout controls — remove 5/10 random genes (control for test 1)

Stability metrics (perturbed vs baseline, on the surviving-node subgraph):
  NMI, AMI, ECS, Jaccard, modularity change, FFL preservation change

Usage:
    python run_perturbation.py                           # all tests, default algos
    python run_perturbation.py --test tf_knockout         # single test
    python run_perturbation.py --algorithms louvain leiden # specific algorithms
    python run_perturbation.py --resume                   # skip completed rows
"""
import os, sys, json, time, argparse, copy
import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lfr'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithm_runner import AlgorithmRunner
from config import (
    OUTPUT_DIR, RESULTS_DIR, PERTURBATION_SEEDS, PERTURBATION_LEVELS,
    TF_KNOCKOUT_LEVELS, GNN_EPOCHS, GNN_UNSUPERVISED, GNN_BASELINES,
    GNN_SUPERVISED, GO_GAF_PATH,
)

DATA_DIR = OUTPUT_DIR
NET_ID = 'grn'

# Default algorithms to run perturbation on (fast + representative).
# Override with --algorithms flag.
DEFAULT_ALGORITHMS = [
    # Classical (top-5 by modularity + enrichment on GRN)
    'leiden', 'louvain', 'rb_pots', 'fastgreedy', 'agdl',
    # GNN unsupervised
    'dmon_gnn', 'mincut_gnn', 'dgi_gnn', 'vgae_gnn',
    # GNN baselines
    'mlp_kmeans', 'node2vec_gnn',
    # GNN supervised
    'gcn_supervised', 'gat_supervised',
]


# =============================================================================
# LOADING
# =============================================================================

def load_grn():
    """Load the GRN network, features, directed edges, and FFLs."""
    net_dir = DATA_DIR / NET_ID
    metadata = json.load(open(net_dir / 'metadata.json'))
    node_list = json.load(open(net_dir / 'node_list.json'))
    features = np.load(str(net_dir / 'features.npy'))
    edges = np.load(str(net_dir / 'edges.npy'))

    G = nx.Graph()
    G.add_nodes_from(range(len(node_list)))
    for row in edges:
        G.add_edge(int(row['src']), int(row['tgt']),
                   weight=abs(float(row['weight'])))

    # Directed edges (string IDs) — needed for TF identification
    directed_path = net_dir / 'directed_edges.json'
    directed_edges = json.load(open(str(directed_path))) if directed_path.is_file() else []

    # FFLs (string IDs)
    ffl_path = net_dir / 'ffls.json'
    ffls = json.load(open(str(ffl_path))) if ffl_path.is_file() else []

    return G, features, node_list, metadata, directed_edges, ffls


def identify_tfs(directed_edges, node_list):
    """Identify TFs and their out-degrees from directed edges.
    Returns dict: node_index -> out_degree (only for source nodes)."""
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    out_degree = Counter()
    for src, tgt, _ in directed_edges:
        if src in node_to_idx:
            out_degree[node_to_idx[src]] += 1
    return out_degree


# =============================================================================
# PERTURBATION FUNCTIONS
# =============================================================================

def perturb_edge_dropout(G, features, frac, rng):
    """Remove a random fraction of edges. Returns new graph + kept-node set."""
    edges = list(G.edges())
    n_remove = int(len(edges) * frac)
    remove_idx = rng.choice(len(edges), size=n_remove, replace=False)
    remove_set = set(remove_idx)

    G_new = G.copy()
    for i in remove_idx:
        u, v = edges[i]
        if G_new.has_edge(u, v):
            G_new.remove_edge(u, v)

    # Remove isolated nodes
    isolates = list(nx.isolates(G_new))
    G_new.remove_nodes_from(isolates)
    surviving = set(G_new.nodes())
    return G_new, surviving


def perturb_gene_removal_hub(G, features, frac):
    """Remove the top-frac% genes by degree (deterministic)."""
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    n_remove = max(1, int(len(degrees) * frac))
    remove_nodes = {node for node, _ in degrees[:n_remove]}

    G_new = G.copy()
    G_new.remove_nodes_from(remove_nodes)
    isolates = list(nx.isolates(G_new))
    G_new.remove_nodes_from(isolates)
    surviving = set(G_new.nodes())
    return G_new, surviving


def perturb_gene_removal_random(G, features, n_remove, rng):
    """Remove n_remove random genes."""
    nodes = list(G.nodes())
    remove_nodes = set(rng.choice(nodes, size=min(n_remove, len(nodes)), replace=False))

    G_new = G.copy()
    G_new.remove_nodes_from(remove_nodes)
    isolates = list(nx.isolates(G_new))
    G_new.remove_nodes_from(isolates)
    surviving = set(G_new.nodes())
    return G_new, surviving


def perturb_tf_knockout(G, features, tf_out_degrees, top_k):
    """Remove the top-K TFs by out-degree (deterministic)."""
    top_tfs = sorted(tf_out_degrees.items(), key=lambda x: x[1], reverse=True)[:top_k]
    remove_nodes = {node for node, _ in top_tfs}

    G_new = G.copy()
    G_new.remove_nodes_from(remove_nodes)
    isolates = list(nx.isolates(G_new))
    G_new.remove_nodes_from(isolates)
    surviving = set(G_new.nodes())
    return G_new, surviving, [node for node, deg in top_tfs]


# =============================================================================
# STABILITY METRICS
# =============================================================================

def partition_to_labels(communities, nodes):
    """Convert list-of-lists partition to a label array aligned to `nodes`."""
    node_to_label = {}
    for ci, comm in enumerate(communities):
        for n in comm:
            node_to_label[n] = ci
    labels = np.array([node_to_label.get(n, -1) for n in nodes])
    return labels


def compute_nmi(labels_a, labels_b):
    from sklearn.metrics import normalized_mutual_info_score
    mask = (labels_a >= 0) & (labels_b >= 0)
    if mask.sum() < 2:
        return 0.0
    return normalized_mutual_info_score(labels_a[mask], labels_b[mask])


def compute_ami(labels_a, labels_b):
    from sklearn.metrics import adjusted_mutual_info_score
    mask = (labels_a >= 0) & (labels_b >= 0)
    if mask.sum() < 2:
        return 0.0
    return adjusted_mutual_info_score(labels_a[mask], labels_b[mask])


def compute_ecs(labels_a, labels_b):
    """Element-centric similarity via clusim (falls back to NMI if unavailable)."""
    try:
        from clusim.clustering import Clustering
        import clusim.sim as sim

        def labels_to_clustering(labels):
            clusters = defaultdict(set)
            for i, l in enumerate(labels):
                if l >= 0:
                    clusters[l].add(i)
            c = Clustering()
            c.from_cluster_list([list(v) for v in clusters.values()])
            return c

        mask = (labels_a >= 0) & (labels_b >= 0)
        if mask.sum() < 2:
            return 0.0
        # Reindex to contiguous 0..N-1
        idx = np.where(mask)[0]
        remap = {old: new for new, old in enumerate(idx)}
        la = np.array([labels_a[i] for i in idx])
        lb = np.array([labels_b[i] for i in idx])

        ca = labels_to_clustering(la)
        cb = labels_to_clustering(lb)
        return sim.element_sim(ca, cb)
    except ImportError:
        return compute_nmi(labels_a, labels_b)


def compute_jaccard_top_k(comms_a, comms_b, surviving, k=10):
    """Mean Jaccard similarity between top-K communities (by size)
    from partition A matched greedily to partition B."""
    # Restrict to surviving nodes
    ca = [set(c) & surviving for c in comms_a]
    cb = [set(c) & surviving for c in comms_b]
    ca = sorted([c for c in ca if c], key=len, reverse=True)[:k]
    cb = [c for c in cb if c]

    if not ca or not cb:
        return 0.0

    jaccards = []
    used = set()
    for a in ca:
        best_j, best_idx = 0.0, -1
        for j, b in enumerate(cb):
            if j in used:
                continue
            inter = len(a & b)
            union = len(a | b)
            jac = inter / union if union > 0 else 0.0
            if jac > best_j:
                best_j, best_idx = jac, j
        jaccards.append(best_j)
        if best_idx >= 0:
            used.add(best_idx)

    return float(np.mean(jaccards))


def compute_modularity(G, communities):
    if not communities:
        return 0.0
    comm_sets = [set(c) for c in communities if len(c) > 0]
    if not comm_sets:
        return 0.0
    try:
        return nx.community.modularity(G, comm_sets)
    except Exception:
        return 0.0


def compute_ffl_preservation(ffls, node_list, communities):
    """Fraction of FFLs with all 3 nodes in the same community."""
    if not ffls:
        return 0.0, 0
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    node_to_comm = {}
    for ci, comm in enumerate(communities):
        for n in comm:
            node_to_comm[n] = ci

    preserved = 0
    counted = 0
    for tf1, tf2, gene in ffls:
        i1 = node_to_idx.get(tf1)
        i2 = node_to_idx.get(tf2)
        i3 = node_to_idx.get(gene)
        if i1 is None or i2 is None or i3 is None:
            continue
        c1 = node_to_comm.get(i1)
        c2 = node_to_comm.get(i2)
        c3 = node_to_comm.get(i3)
        if c1 is not None and c1 == c2 == c3:
            preserved += 1
        counted += 1

    return (preserved / counted if counted > 0 else 0.0), counted


# =============================================================================
# RUNNING ALGORITHMS ON PERTURBED GRAPH
# =============================================================================

def run_algo_on_graph(runner, G, algo_name, features, node_list, louvain_k=None):
    """Run a single algorithm on a (possibly perturbed) graph.
    Returns list-of-lists communities (node indices in original graph)."""
    kwargs = {}
    is_gnn = algo_name in (GNN_UNSUPERVISED + GNN_BASELINES + GNN_SUPERVISED)

    if is_gnn:
        # Remap features to the subgraph's node ordering
        sub_nodes = sorted(G.nodes())
        sub_features = features[sub_nodes] if features is not None else None
        # Relabel graph to 0..N-1 for PyG
        mapping = {old: new for new, old in enumerate(sub_nodes)}
        G_relabeled = nx.relabel_nodes(G, mapping)

        k = louvain_k
        if k is None:
            import community as community_louvain
            part = community_louvain.best_partition(G_relabeled)
            k = max(2, len(set(part.values())))

        kwargs['num_clusters'] = k
        kwargs['epochs'] = GNN_EPOCHS
        kwargs['features'] = sub_features

        if algo_name in GNN_SUPERVISED:
            labels, num_classes = _build_go_slim_labels_for_subgraph(
                sub_nodes, node_list)
            if num_classes < 2:
                return None
            kwargs['labels'] = labels
            kwargs['train_mask'] = _build_train_mask(labels, seed=0)
            kwargs['num_clusters'] = num_classes

        result = runner.run_algorithm(G_relabeled, algo_name, **kwargs)
        if not result.success or not result.communities:
            return None
        # Map communities back to original node indices
        inv_map = {new: old for old, new in mapping.items()}
        communities = [[inv_map[n] for n in comm] for comm in result.communities]
        return communities
    else:
        result = runner.run_algorithm(G, algo_name, **kwargs)
        if not result.success or not result.communities:
            return None
        return result.communities


def _build_go_slim_labels_for_subgraph(sub_nodes, full_node_list):
    """Lightweight GO Slim labels for a subgraph (subset of node indices)."""
    import gzip

    go_slim_bp = [
        'GO:0008152', 'GO:0009987', 'GO:0050896', 'GO:0065007',
        'GO:0032502', 'GO:0051179', 'GO:0023052', 'GO:0006950',
        'GO:0002376', 'GO:0007049', 'GO:0012501', 'GO:0007155',
        'GO:0048870', 'GO:0030154', 'GO:0019725',
    ]
    go_slim_set = set(go_slim_bp)

    meta = json.load(open(DATA_DIR / NET_ID / 'metadata.json'))
    id_col = 1 if meta['id_format'] == 'uniprot' else 2

    gene_to_terms = defaultdict(set)
    with gzip.open(str(GO_GAF_PATH), 'rt') as f:
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

    # Assign raw GO Slim term per node
    raw_labels = [None] * len(sub_nodes)
    for i, nidx in enumerate(sub_nodes):
        gene = full_node_list[nidx]
        terms = gene_to_terms.get(gene, set())
        for t in terms:
            if t in go_slim_set:
                raw_labels[i] = t
                break

    # Remap to contiguous 0..K-1 labels
    terms_present = sorted(set(t for t in raw_labels if t is not None))
    term_to_idx = {t: i for i, t in enumerate(terms_present)}

    labels = np.full(len(sub_nodes), -1, dtype=np.int64)
    for i, t in enumerate(raw_labels):
        if t is not None:
            labels[i] = term_to_idx[t]

    num_classes = len(terms_present)
    return labels, num_classes


def _build_train_mask(labels, seed=0):
    rng = np.random.RandomState(seed)
    labeled = np.where(labels >= 0)[0]
    train_mask = np.zeros(len(labels), dtype=bool)
    if len(labeled) > 0:
        train_size = max(1, int(0.2 * len(labeled)))
        train_idx = rng.choice(labeled, size=train_size, replace=False)
        train_mask[train_idx] = True
    return train_mask


# =============================================================================
# TEST ORCHESTRATORS
# =============================================================================

def generate_perturbations(G, features, node_list, directed_edges, tf_out_degrees):
    """Yield (test_name, level_str, seed, G_perturbed, surviving_nodes) tuples."""

    # --- Test 1: TF knockout (deterministic) ---
    for k in TF_KNOCKOUT_LEVELS:
        G_p, surv, removed_tfs = perturb_tf_knockout(G, features, tf_out_degrees, k)
        yield f'tf_knockout_top{k}', f'top{k}', 0, G_p, surv

    # --- Test 2: Edge dropout (stochastic) ---
    for frac in PERTURBATION_LEVELS['edge_dropout']:
        for seed in range(PERTURBATION_SEEDS):
            rng = np.random.RandomState(seed)
            G_p, surv = perturb_edge_dropout(G, features, frac, rng)
            yield f'edge_dropout_{frac}', f'{frac}', seed, G_p, surv

    # --- Test 3: Hub gene removal (deterministic) ---
    for frac in PERTURBATION_LEVELS['node_removal_hub']:
        G_p, surv = perturb_gene_removal_hub(G, features, frac)
        yield f'gene_removal_hub_{frac}', f'{frac}', 0, G_p, surv

    # --- Test 4: TF knockout controls — random gene removal (stochastic) ---
    for k in TF_KNOCKOUT_LEVELS:
        for seed in range(PERTURBATION_SEEDS):
            rng = np.random.RandomState(seed)
            G_p, surv = perturb_gene_removal_random(G, features, k, rng)
            yield f'tf_knockout_control_random{k}', f'random{k}', seed, G_p, surv


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GRN Perturbation Analysis')
    parser.add_argument('--test', type=str, default=None,
                        choices=['tf_knockout', 'edge_dropout',
                                 'gene_removal_hub', 'tf_knockout_control'],
                        help='Run only one perturbation test')
    parser.add_argument('--algorithms', nargs='+', default=None,
                        help='Algorithms to run (default: fast representative set)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip already-completed (test, level, seed, algo) rows')
    args = parser.parse_args()

    algos = args.algorithms or DEFAULT_ALGORITHMS

    # --- Setup ---
    results_dir = RESULTS_DIR / 'perturbation'
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / 'grn_perturbation_results.csv'
    partitions_dir = results_dir / 'partitions'
    partitions_dir.mkdir(parents=True, exist_ok=True)

    csv_cols = [
        'test', 'level', 'seed', 'algorithm',
        'nodes_original', 'edges_original',
        'nodes_perturbed', 'edges_perturbed',
        'k_original', 'k_perturbed',
        'modularity_original', 'modularity_perturbed', 'modularity_change',
        'nmi', 'ami', 'ecs', 'jaccard',
        'ffl_preserved_orig', 'ffl_preserved_pert', 'ffl_change',
        'runtime',
    ]

    existing = set()
    if args.resume and csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                existing.add((r['test'], str(r['level']), int(r['seed']), r['algorithm']))
        except Exception:
            pass
        if existing:
            print(f"Resume: {len(existing)} rows cached", flush=True)

    def _append_row(row_dict):
        row = pd.DataFrame([row_dict], columns=csv_cols)
        header = not csv_path.is_file()
        row.to_csv(csv_path, mode='a', header=header, index=False)

    # --- Load GRN ---
    print("Loading GRN network...", flush=True)
    G, features, node_list, metadata, directed_edges, ffls = load_grn()
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{len(ffls)} FFLs", flush=True)

    tf_out_degrees = identify_tfs(directed_edges, node_list)
    top_tfs = sorted(tf_out_degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top-10 TFs by out-degree: "
          f"{[(node_list[n], d) for n, d in top_tfs]}", flush=True)

    # --- Baseline partitions ---
    print("\nComputing baseline partitions...", flush=True)
    runner = AlgorithmRunner(verbose=False)
    baseline_partitions = {}
    baseline_dir = RESULTS_DIR / 'partitions'

    for algo in algos:
        # Try loading from Set A results first
        part_file = baseline_dir / f'{NET_ID}_{algo}.json'
        if not part_file.is_file():
            # Try seed 0 for GNN
            part_file = baseline_dir / f'{NET_ID}_{algo}_s0.json'
        if part_file.is_file():
            comms = json.load(open(str(part_file)))
            print(f"  {algo}: loaded baseline ({len(comms)} communities)", flush=True)
            baseline_partitions[algo] = comms
        else:
            print(f"  {algo}: running baseline...", end=' ', flush=True)
            comms = run_algo_on_graph(runner, G, algo, features, node_list)
            if comms:
                baseline_partitions[algo] = comms
                out_file = partitions_dir / f'baseline_{algo}.json'
                with open(str(out_file), 'w') as f:
                    json.dump(comms, f)
                print(f"OK ({len(comms)} communities)", flush=True)
            else:
                print("FAIL — skipping this algorithm", flush=True)

    if not baseline_partitions:
        print("No baseline partitions available. Exiting.", flush=True)
        sys.exit(1)

    # Precompute baseline metrics
    baseline_mod = {a: compute_modularity(G, c) for a, c in baseline_partitions.items()}
    baseline_ffl = {}
    for a, c in baseline_partitions.items():
        frac, _ = compute_ffl_preservation(ffls, node_list, c)
        baseline_ffl[a] = frac

    print(f"\nBaseline modularity: {baseline_mod}", flush=True)
    print(f"Baseline FFL preservation: {baseline_ffl}", flush=True)

    # --- Run perturbations ---
    n_orig = G.number_of_nodes()
    e_orig = G.number_of_edges()
    total_start = time.time()
    completed = 0

    for test_name, level_str, seed, G_pert, surviving in \
            generate_perturbations(G, features, node_list, directed_edges, tf_out_degrees):

        # Filter by --test flag
        if args.test:
            if not test_name.startswith(args.test):
                continue

        n_pert = G_pert.number_of_nodes()
        e_pert = G_pert.number_of_edges()
        pct_nodes_lost = 100 * (1 - n_pert / n_orig)
        pct_edges_lost = 100 * (1 - e_pert / e_orig)

        print(f"\n{'='*70}", flush=True)
        print(f"  {test_name}  level={level_str}  seed={seed}", flush=True)
        print(f"  Nodes: {n_orig} -> {n_pert} (-{pct_nodes_lost:.1f}%)  "
              f"Edges: {e_orig} -> {e_pert} (-{pct_edges_lost:.1f}%)", flush=True)
        print(f"{'='*70}", flush=True)

        for algo in baseline_partitions:
            key = (test_name, level_str, seed, algo)
            if key in existing:
                print(f"  {algo}: skipped (cached)", flush=True)
                continue

            print(f"  {algo}...", end=' ', flush=True)
            t0 = time.time()

            comms_pert = run_algo_on_graph(runner, G_pert, algo, features, node_list)
            runtime = time.time() - t0

            if comms_pert is None:
                print(f"FAIL ({runtime:.1f}s)", flush=True)
                continue

            # Save perturbed partition
            pf = partitions_dir / f'{test_name}_s{seed}_{algo}.json'
            with open(str(pf), 'w') as f:
                json.dump(comms_pert, f)

            # Compute stability metrics on surviving nodes
            base_comms = baseline_partitions[algo]
            surv_sorted = sorted(surviving)

            labels_base = partition_to_labels(base_comms, surv_sorted)
            labels_pert = partition_to_labels(comms_pert, surv_sorted)

            nmi = compute_nmi(labels_base, labels_pert)
            ami = compute_ami(labels_base, labels_pert)
            ecs = compute_ecs(labels_base, labels_pert)
            jaccard = compute_jaccard_top_k(base_comms, comms_pert, surviving, k=10)

            mod_pert = compute_modularity(G_pert, comms_pert)
            mod_orig = baseline_mod[algo]

            ffl_pert, _ = compute_ffl_preservation(ffls, node_list, comms_pert)
            ffl_orig = baseline_ffl[algo]

            print(f"OK  NMI={nmi:.3f} AMI={ami:.3f} ECS={ecs:.3f} "
                  f"Jac={jaccard:.3f} Q={mod_pert:.3f} "
                  f"FFL={ffl_pert:.3f} ({runtime:.1f}s)", flush=True)

            _append_row({
                'test': test_name, 'level': level_str, 'seed': seed,
                'algorithm': algo,
                'nodes_original': n_orig, 'edges_original': e_orig,
                'nodes_perturbed': n_pert, 'edges_perturbed': e_pert,
                'k_original': len(base_comms), 'k_perturbed': len(comms_pert),
                'modularity_original': round(mod_orig, 4),
                'modularity_perturbed': round(mod_pert, 4),
                'modularity_change': round(mod_pert - mod_orig, 4),
                'nmi': round(nmi, 4), 'ami': round(ami, 4),
                'ecs': round(ecs, 4), 'jaccard': round(jaccard, 4),
                'ffl_preserved_orig': round(ffl_orig, 4),
                'ffl_preserved_pert': round(ffl_pert, 4),
                'ffl_change': round(ffl_pert - ffl_orig, 4),
                'runtime': round(runtime, 2),
            })
            completed += 1

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}", flush=True)
    print(f"Done. {completed} rows saved to {csv_path}", flush=True)
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)

    # --- Print summary ---
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        print(f"\n{'='*70}", flush=True)
        print("PERTURBATION SUMMARY (mean across seeds)", flush=True)
        print(f"{'='*70}", flush=True)

        grouped = df.groupby(['test', 'algorithm']).agg({
            'nmi': 'mean', 'ami': 'mean', 'ecs': 'mean',
            'jaccard': 'mean', 'modularity_change': 'mean',
            'ffl_change': 'mean',
        }).round(3)
        print(grouped.to_string(), flush=True)


if __name__ == '__main__':
    main()
