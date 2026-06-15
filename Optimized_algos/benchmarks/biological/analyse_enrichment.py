"""
Phase 3: Biological Enrichment Analysis.

For each algorithm's partition on each network, compute:
  1. GO enrichment  — fraction of communities with >=1 significant GO term
  2. KEGG enrichment — fraction with >=1 significant KEGG pathway
  3. Sign coherence — (SIGNOR only) are intra-community edges mostly same-sign?
  4. FFL preservation — (GRN only) fraction of FFLs entirely within one community

Usage:
    python analyse_enrichment.py                          # all networks, all partitions
    python analyse_enrichment.py --network signor         # one network
    python analyse_enrichment.py --algorithm leiden        # one algorithm across all
"""
import os, sys, json, time, argparse, gzip
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from config import NETWORKS, NETWORK_ORDER, RESULTS_DIR, OUTPUT_DIR, GO_GAF_PATH

DATA_DIR = OUTPUT_DIR
PARTITIONS_DIR = RESULTS_DIR / 'partitions'
GAF_PATH = GO_GAF_PATH

MIN_COMMUNITY_SIZE = 5      # skip tiny communities for enrichment
ENRICHMENT_PVAL = 0.05      # significance threshold (Benjamini-Hochberg)
MAX_COMMUNITIES_FOR_API = 1100  # covers biogrid_cpm (1095 valid), the largest partition


# =============================================================================
# GO / KEGG ENRICHMENT via g:Profiler
# =============================================================================

GPROFILER_TIMEOUT = 60       # seconds per community query
GPROFILER_RETRIES = 3        # retries on timeout / transient error
GPROFILER_BATCH_PAUSE = 1.0  # seconds pause between queries to avoid rate limits


def _gprofiler_query_with_retry(gp, gene_names, bg_genes, retries=GPROFILER_RETRIES):
    """Single community g:Profiler call with timeout and retry."""
    import signal, platform

    for attempt in range(retries):
        try:
            if platform.system() != 'Windows':
                def _timeout_handler(signum, frame):
                    raise TimeoutError("g:Profiler query timed out")
                old = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(GPROFILER_TIMEOUT)

            result = gp.profile(
                organism='hsapiens',
                query=gene_names,
                background=bg_genes,
                sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG'],
                user_threshold=ENRICHMENT_PVAL,
                no_evidences=True,
                all_results=False,
            )

            if platform.system() != 'Windows':
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)

            return result
        except (TimeoutError, Exception) as e:
            if platform.system() != 'Windows':
                signal.alarm(0)
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"      [retry {attempt+1}/{retries} after {wait}s: {str(e)[:60]}]",
                      flush=True)
                time.sleep(wait)
            else:
                raise
    return None


def run_gprofiler_enrichment(communities, node_list, background=None, organism='hsapiens'):
    """
    Run GO + KEGG enrichment for each community using g:Profiler API.

    Returns:
        enrichment_summary: dict with per-community and aggregate stats
    """
    from gprofiler import GProfiler
    gp = GProfiler(return_dataframe=True)

    bg_genes = background if background else node_list

    # Filter to communities large enough for enrichment
    valid_comms = [(i, c) for i, c in enumerate(communities)
                   if len(c) >= MIN_COMMUNITY_SIZE]

    if len(valid_comms) > MAX_COMMUNITIES_FOR_API:
        import random
        random.seed(42)
        valid_comms = random.sample(valid_comms, MAX_COMMUNITIES_FOR_API)

    go_enriched = 0
    kegg_enriched = 0
    total_tested = len(valid_comms)
    total_go_terms = 0
    total_kegg_terms = 0
    per_community = []

    for qi, (comm_idx, node_ids) in enumerate(valid_comms):
        gene_names = [node_list[nid] for nid in node_ids]

        try:
            result = _gprofiler_query_with_retry(gp, gene_names, bg_genes)

            if isinstance(result, pd.DataFrame) and len(result) > 0:
                go_hits = result[result['source'].str.startswith('GO:')]
                kegg_hits = result[result['source'] == 'KEGG']

                has_go = len(go_hits) > 0
                has_kegg = len(kegg_hits) > 0
                if has_go:
                    go_enriched += 1
                if has_kegg:
                    kegg_enriched += 1
                total_go_terms += len(go_hits)
                total_kegg_terms += len(kegg_hits)

                per_community.append({
                    'comm_idx': comm_idx,
                    'size': len(node_ids),
                    'go_terms': len(go_hits),
                    'kegg_terms': len(kegg_hits),
                    'top_go': go_hits.iloc[0]['name'] if has_go else None,
                    'top_kegg': kegg_hits.iloc[0]['name'] if has_kegg else None,
                })
            else:
                per_community.append({
                    'comm_idx': comm_idx, 'size': len(node_ids),
                    'go_terms': 0, 'kegg_terms': 0,
                    'top_go': None, 'top_kegg': None,
                })
        except Exception as e:
            per_community.append({
                'comm_idx': comm_idx, 'size': len(node_ids),
                'go_terms': 0, 'kegg_terms': 0,
                'top_go': None, 'top_kegg': None,
                'error': str(e)[:80],
            })

        if GPROFILER_BATCH_PAUSE > 0 and qi < len(valid_comms) - 1:
            time.sleep(GPROFILER_BATCH_PAUSE)

    return {
        'communities_tested': total_tested,
        'communities_total': len(communities),
        'go_enriched_count': go_enriched,
        'go_enriched_frac': go_enriched / total_tested if total_tested > 0 else 0,
        'kegg_enriched_count': kegg_enriched,
        'kegg_enriched_frac': kegg_enriched / total_tested if total_tested > 0 else 0,
        'total_go_terms': total_go_terms,
        'total_kegg_terms': total_kegg_terms,
        'mean_go_per_community': total_go_terms / total_tested if total_tested > 0 else 0,
        'per_community': per_community,
    }


# =============================================================================
# SIGN COHERENCE (SIGNOR only)
# =============================================================================

def compute_sign_coherence(net_id, communities):
    """
    For signed networks: what fraction of intra-community edges share the
    dominant sign? A perfectly coherent community has all edges the same sign.

    Returns:
        mean_coherence: float in [0.5, 1.0]
        per_community: list of coherence values
    """
    edges = np.load(str(DATA_DIR / net_id / 'edges.npy'))

    # Build node -> community mapping
    node_to_comm = {}
    for ci, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = ci

    # Count signs per community
    comm_pos = defaultdict(int)
    comm_neg = defaultdict(int)
    comm_total = defaultdict(int)

    for row in edges:
        src, tgt, w = int(row['src']), int(row['tgt']), float(row['weight'])
        cs = node_to_comm.get(src)
        ct = node_to_comm.get(tgt)
        if cs is not None and ct is not None and cs == ct:
            comm_total[cs] += 1
            if w > 0:
                comm_pos[cs] += 1
            else:
                comm_neg[cs] += 1

    coherences = []
    for ci in range(len(communities)):
        total = comm_total.get(ci, 0)
        if total >= 2:
            dominant = max(comm_pos.get(ci, 0), comm_neg.get(ci, 0))
            coherences.append(dominant / total)

    mean_coh = np.mean(coherences) if coherences else 0.0
    return {
        'mean_sign_coherence': mean_coh,
        'communities_with_edges': len(coherences),
        'per_community_coherence': coherences,
    }


# =============================================================================
# FFL PRESERVATION (GRN only)
# =============================================================================

def compute_ffl_preservation(net_id, communities):
    """
    For directed GRN: what fraction of feed-forward loops (FFLs) have all
    three nodes in the same community?

    Returns:
        ffl_preserved_frac: float in [0, 1]
        ffl_total: int
        ffl_preserved: int
    """
    ffl_path = DATA_DIR / net_id / 'ffls.npy'
    if not ffl_path.is_file():
        return {'ffl_preserved_frac': None, 'ffl_total': 0, 'ffl_preserved': 0,
                'note': 'no FFLs file'}

    ffls = np.load(str(ffl_path))

    node_to_comm = {}
    for ci, comm in enumerate(communities):
        for node in comm:
            node_to_comm[node] = ci

    preserved = 0
    total = len(ffls)
    for ffl in ffls:
        a, b, c = int(ffl[0]), int(ffl[1]), int(ffl[2])
        ca = node_to_comm.get(a)
        cb = node_to_comm.get(b)
        cc = node_to_comm.get(c)
        if ca is not None and ca == cb == cc:
            preserved += 1

    return {
        'ffl_preserved_frac': preserved / total if total > 0 else 0,
        'ffl_total': total,
        'ffl_preserved': preserved,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyse_partition(net_id, algo_name, seed=None, part_file=None):
    """Run full enrichment analysis on one partition."""
    if part_file is None:
        suffix = f"_s{seed}" if seed is not None else ""
        part_file = PARTITIONS_DIR / f"{net_id}_{algo_name}{suffix}.json"

    if not Path(part_file).is_file():
        return None

    communities = json.load(open(str(part_file)))
    node_list = json.load(open(str(DATA_DIR / net_id / 'node_list.json')))
    metadata = json.load(open(str(DATA_DIR / net_id / 'metadata.json')))

    result = {
        'network': net_id,
        'algorithm': algo_name,
        'seed': seed if seed is not None else 0,
        'num_communities': len(communities),
    }

    # 1. GO/KEGG enrichment
    print(f"    GO/KEGG enrichment ({len(communities)} communities)...", end=' ',
          flush=True)
    t0 = time.time()
    enrich = run_gprofiler_enrichment(communities, node_list, background=node_list)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)

    result['go_enriched_frac'] = enrich['go_enriched_frac']
    result['go_enriched_count'] = enrich['go_enriched_count']
    result['kegg_enriched_frac'] = enrich['kegg_enriched_frac']
    result['kegg_enriched_count'] = enrich['kegg_enriched_count']
    result['communities_tested'] = enrich['communities_tested']
    result['mean_go_per_community'] = enrich['mean_go_per_community']

    # 2. Sign coherence (SIGNOR)
    if metadata.get('signed'):
        print(f"    Sign coherence...", end=' ', flush=True)
        sign_res = compute_sign_coherence(net_id, communities)
        result['sign_coherence'] = sign_res['mean_sign_coherence']
        print(f"{sign_res['mean_sign_coherence']:.3f}", flush=True)

    # 3. FFL preservation (GRN)
    if metadata.get('directed') and (DATA_DIR / net_id / 'ffls.npy').is_file():
        print(f"    FFL preservation...", end=' ', flush=True)
        ffl_res = compute_ffl_preservation(net_id, communities)
        result['ffl_preserved_frac'] = ffl_res['ffl_preserved_frac']
        result['ffl_total'] = ffl_res['ffl_total']
        result['ffl_preserved'] = ffl_res['ffl_preserved']
        print(f"{ffl_res['ffl_preserved']}/{ffl_res['ffl_total']} "
              f"({ffl_res['ffl_preserved_frac']:.3f})", flush=True)

    return result


def main():
    parser = argparse.ArgumentParser(description='Biological Enrichment Analysis')
    parser.add_argument('--network', type=str, default=None)
    parser.add_argument('--algorithm', type=str, default=None)
    parser.add_argument('--partitions-dir', type=str, nargs='+', default=None,
                        help='One or more directories containing partition JSON files. '
                             'Defaults to RESULTS_DIR/partitions if not specified.')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path (default: RESULTS_DIR/bio_enrichment_results.csv)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip partitions already in the output CSV')
    args = parser.parse_args()

    # Discover available partitions from one or more directories
    if args.partitions_dir:
        part_dirs = [Path(d) for d in args.partitions_dir]
    else:
        part_dirs = [PARTITIONS_DIR]

    partition_files = []
    for pd_dir in part_dirs:
        if pd_dir.is_dir():
            partition_files.extend(sorted(pd_dir.glob('*.json')))
        else:
            print(f"Warning: {pd_dir} not found, skipping", flush=True)

    if not partition_files:
        print(f"No partition files found in: {part_dirs}")
        sys.exit(1)

    print(f"Found {len(partition_files)} partition files "
          f"across {len(part_dirs)} director(ies)", flush=True)

    # Parse partition filenames -> (network, algorithm, seed, filepath)
    entries = []
    for pf in partition_files:
        name = pf.stem  # e.g. "signor_leiden" or "signor_dmon_gnn_s0"
        seed = None
        base = name
        if '_s' in name and name.rsplit('_s', 1)[1].isdigit():
            base, seed_str = name.rsplit('_s', 1)
            seed = int(seed_str)
        for net_id in NETWORK_ORDER:
            if base.startswith(net_id + '_'):
                algo = base[len(net_id) + 1:]
                entries.append((net_id, algo, seed, str(pf)))
                break

    # Apply filters
    if args.network:
        entries = [(n, a, s, f) for n, a, s, f in entries if n == args.network]
    if args.algorithm:
        entries = [(n, a, s, f) for n, a, s, f in entries if a == args.algorithm]

    print(f"Analysing {len(entries)} partitions", flush=True)

    # Resume support: skip already-completed entries
    out_path = Path(args.output) if args.output else RESULTS_DIR / 'bio_enrichment_results.csv'
    done_keys = set()
    existing_df = None
    if args.resume and out_path.is_file():
        existing_df = pd.read_csv(out_path)
        for _, row in existing_df.iterrows():
            done_keys.add((row['network'], row['algorithm'], int(row['seed'])))
        print(f"Resume: {len(done_keys)} partitions already done", flush=True)

    all_results = []
    for i, (net_id, algo, seed, fpath) in enumerate(entries):
        key = (net_id, algo, seed if seed is not None else 0)
        if key in done_keys:
            continue
        seed_str = f" (seed {seed})" if seed is not None else ""
        print(f"\n  [{i+1}/{len(entries)}] {net_id} / {algo}{seed_str}:", flush=True)
        result = analyse_partition(net_id, algo, seed, part_file=fpath)
        if result:
            all_results.append(result)
            # Incremental save after each partition
            df_new = pd.DataFrame(all_results)
            if existing_df is not None:
                df_save = pd.concat([existing_df, df_new], ignore_index=True)
            else:
                df_save = df_new
            df_save.to_csv(out_path, index=False)

    # Final summary
    if out_path.is_file():
        df = pd.read_csv(out_path)
        print(f"\nResults saved to {out_path} ({len(df)} rows)", flush=True)

        print("\n" + "=" * 80)
        print("ENRICHMENT SUMMARY")
        print("=" * 80)

        for net_id in NETWORK_ORDER:
            net_df = df[df['network'] == net_id].copy()
            if net_df.empty:
                continue
            net_df = net_df.sort_values('go_enriched_frac', ascending=False)

            meta = json.load(open(str(DATA_DIR / net_id / 'metadata.json')))
            edge_key = 'num_edges_undirected' if 'num_edges_undirected' in meta else 'num_edges'
            print(f"\n--- {net_id} ({meta['num_nodes']} nodes, "
                  f"{meta.get(edge_key, '?')} edges) ---")

            cols = ['algorithm', 'seed', 'num_communities', 'communities_tested',
                    'go_enriched_frac', 'kegg_enriched_frac']
            extra = []
            if 'sign_coherence' in net_df.columns and net_df['sign_coherence'].notna().any():
                extra.append('sign_coherence')
            if 'ffl_preserved_frac' in net_df.columns and net_df['ffl_preserved_frac'].notna().any():
                extra.append('ffl_preserved_frac')

            print(net_df[cols + extra].to_string(index=False))


if __name__ == '__main__':
    main()
