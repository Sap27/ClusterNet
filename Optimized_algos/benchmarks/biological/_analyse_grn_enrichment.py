"""
Analyse all GRN partitions: FFL preservation + GO/KEGG enrichment via g:Profiler.
Collects partitions from multiple result directories, deduplicates, and ranks.
"""
import json, glob, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ---- Paths (Windows, accessed via WSL /mnt/d/) ----
DATA_DIR = Path('/mnt/d/cluster/BIologicalNetworks/data/grn')

PARTITION_DIRS = [
    Path('/mnt/d/results/partitions'),
    Path('/mnt/d/results-biological-networks-classical2/results-biological-networks-classical/results/partitions'),
    Path('/mnt/d/cluster/BIologicalNetworks/results/partitions'),
]

# ---- Load GRN data ----
print("Loading GRN metadata...", flush=True)
node_list = json.load(open(DATA_DIR / 'node_list.json'))
metadata = json.load(open(DATA_DIR / 'metadata.json'))
node_to_idx = {n: i for i, n in enumerate(node_list)}

# FFLs
ffl_path = DATA_DIR / 'ffls.json'
if ffl_path.is_file():
    ffls = json.load(open(str(ffl_path)))
    print(f"  {len(ffls):,} FFLs loaded", flush=True)
else:
    ffl_npy = DATA_DIR / 'ffls.npy'
    if ffl_npy.is_file():
        ffl_arr = np.load(str(ffl_npy))
        ffls = [list(row) for row in ffl_arr]
        print(f"  {len(ffls):,} FFLs loaded from npy", flush=True)
    else:
        ffls = []
        print("  WARNING: No FFLs found", flush=True)

# Directed edges for TF identification
de_path = DATA_DIR / 'directed_edges.json'
directed_edges = json.load(open(str(de_path))) if de_path.is_file() else []

print(f"  {len(node_list):,} nodes, {metadata.get('num_edges_undirected', '?')} undirected edges",
      flush=True)

# ---- Collect all GRN partition files ----
partitions = {}  # algo_name -> communities (list of lists)

for pdir in PARTITION_DIRS:
    if not pdir.is_dir():
        continue
    for pf in sorted(pdir.glob('grn_*.json')):
        name = pf.stem  # e.g. grn_leiden or grn_dmon_gnn_s0
        algo = name[4:]  # strip 'grn_'

        # For GNN with seeds, just use seed 0
        if '_s1' in algo or '_s2' in algo:
            continue
        algo_clean = algo.replace('_s0', '')

        if algo_clean not in partitions:
            try:
                comms = json.load(open(str(pf)))
                partitions[algo_clean] = comms
            except Exception as e:
                print(f"  WARN: Failed to load {pf}: {e}", flush=True)

print(f"\nLoaded {len(partitions)} unique GRN partitions:", flush=True)
for a, c in sorted(partitions.items(), key=lambda x: len(x[1])):
    print(f"  {a:25s}  K={len(c)}", flush=True)

# ---- FFL Preservation ----
def compute_ffl_preservation(communities):
    if not ffls:
        return 0.0, 0
    node_to_comm = {}
    for ci, comm in enumerate(communities):
        for n in comm:
            node_to_comm[n] = ci
    preserved = 0
    counted = 0
    for tf1, tf2, gene in ffls:
        i1 = node_to_idx.get(str(tf1)) if isinstance(tf1, str) else tf1
        i2 = node_to_idx.get(str(tf2)) if isinstance(tf2, str) else tf2
        i3 = node_to_idx.get(str(gene)) if isinstance(gene, str) else gene
        if i1 is None or i2 is None or i3 is None:
            # Try as integers directly
            i1 = tf1 if isinstance(tf1, int) else i1
            i2 = tf2 if isinstance(tf2, int) else i2
            i3 = gene if isinstance(gene, int) else i3
        if i1 is None or i2 is None or i3 is None:
            continue
        c1 = node_to_comm.get(i1)
        c2 = node_to_comm.get(i2)
        c3 = node_to_comm.get(i3)
        if c1 is not None and c1 == c2 == c3:
            preserved += 1
        counted += 1
    return (preserved / counted if counted > 0 else 0.0), counted

print("\n--- FFL Preservation ---", flush=True)
ffl_results = {}
for algo, comms in sorted(partitions.items()):
    frac, total = compute_ffl_preservation(comms)
    ffl_results[algo] = frac
    print(f"  {algo:25s}  K={len(comms):>6}  FFL={frac:.4f} ({int(frac*total):,}/{total:,})",
          flush=True)

# ---- GO/KEGG Enrichment via g:Profiler ----
from gprofiler import GProfiler

MIN_COMMUNITY_SIZE = 5
MAX_COMMUNITIES_FOR_API = 100

# GRN uses UniProt IDs — g:Profiler handles these natively
print("\n--- GO/KEGG Enrichment (g:Profiler) ---", flush=True)
gp = GProfiler(return_dataframe=True)

enrich_results = {}

for algo, comms in sorted(partitions.items()):
    valid_comms = [(i, c) for i, c in enumerate(comms) if len(c) >= MIN_COMMUNITY_SIZE]
    if not valid_comms:
        print(f"  {algo:25s}  No communities >= {MIN_COMMUNITY_SIZE} nodes", flush=True)
        enrich_results[algo] = {'go_frac': 0, 'kegg_frac': 0, 'tested': 0,
                                'go_count': 0, 'kegg_count': 0, 'mean_go': 0}
        continue

    if len(valid_comms) > MAX_COMMUNITIES_FOR_API:
        import random
        random.seed(42)
        valid_comms = random.sample(valid_comms, MAX_COMMUNITIES_FOR_API)

    go_enriched = 0
    kegg_enriched = 0
    total_go_terms = 0
    total_kegg_terms = 0
    tested = len(valid_comms)

    for ci, node_ids in valid_comms:
        gene_names = [node_list[nid] for nid in node_ids]
        try:
            result = gp.profile(
                organism='hsapiens',
                query=gene_names,
                background=node_list,
                sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG'],
                user_threshold=0.05,
                no_evidences=True,
                all_results=False,
            )
            if isinstance(result, pd.DataFrame) and len(result) > 0:
                go_hits = result[result['source'].str.startswith('GO:')]
                kegg_hits = result[result['source'] == 'KEGG']
                if len(go_hits) > 0:
                    go_enriched += 1
                if len(kegg_hits) > 0:
                    kegg_enriched += 1
                total_go_terms += len(go_hits)
                total_kegg_terms += len(kegg_hits)
        except Exception as e:
            pass

    go_frac = go_enriched / tested if tested > 0 else 0
    kegg_frac = kegg_enriched / tested if tested > 0 else 0
    mean_go = total_go_terms / tested if tested > 0 else 0

    enrich_results[algo] = {
        'go_frac': go_frac, 'kegg_frac': kegg_frac, 'tested': tested,
        'go_count': go_enriched, 'kegg_count': kegg_enriched,
        'mean_go': mean_go,
    }
    print(f"  {algo:25s}  tested={tested:>4}  GO={go_frac:.2%} ({go_enriched}/{tested})  "
          f"KEGG={kegg_frac:.2%} ({kegg_enriched}/{tested})  mean_GO_terms={mean_go:.1f}",
          flush=True)

# ---- Combined Ranking Table ----
print("\n" + "=" * 110, flush=True)
print("COMBINED GRN RANKING (modularity from CSV + FFL + GO/KEGG enrichment)", flush=True)
print("=" * 110, flush=True)

# Load modularity from CSVs
mod_data = {}
for csv_path in [
    '/mnt/d/results-biological-networks-classical2/results-biological-networks-classical/results/bio_classical_results.csv',
    '/mnt/d/cluster/BIologicalNetworks/results/bio_classical_results.csv',
    '/mnt/d/results/bio_gnn_results.csv',
]:
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        grn_df = df[df['network'] == 'grn']
        for _, row in grn_df.iterrows():
            algo = row['algorithm']
            seed = int(row.get('seed', 0))
            if seed > 0:
                continue
            if algo not in mod_data or row.get('modularity', 0) > mod_data[algo]['modularity']:
                mod_data[algo] = {
                    'modularity': row.get('modularity', 0),
                    'K': int(row.get('num_communities', 0)),
                    'runtime': row.get('runtime', 0),
                    'type': row.get('algorithm_type', 'unknown'),
                }

rows = []
for algo in sorted(set(list(partitions.keys()))):
    mod_info = mod_data.get(algo, {})
    enrich = enrich_results.get(algo, {})
    rows.append({
        'algorithm': algo,
        'type': mod_info.get('type', '?'),
        'K': mod_info.get('K', len(partitions.get(algo, []))),
        'Q': mod_info.get('modularity', 0),
        'FFL': ffl_results.get(algo, 0),
        'GO%': enrich.get('go_frac', 0),
        'KEGG%': enrich.get('kegg_frac', 0),
        'mean_GO': enrich.get('mean_go', 0),
        'tested': enrich.get('tested', 0),
        'runtime': mod_info.get('runtime', 0),
    })

df_out = pd.DataFrame(rows)
df_out = df_out.sort_values(['GO%', 'FFL', 'Q'], ascending=[False, False, False])

print(f"\n{'Algorithm':25s} {'Type':20s} {'K':>6} {'Q':>8} {'FFL':>8} "
      f"{'GO%':>8} {'KEGG%':>8} {'avgGO':>8} {'tested':>7} {'time':>8}", flush=True)
print("-" * 110, flush=True)
for _, r in df_out.iterrows():
    print(f"{r['algorithm']:25s} {r['type']:20s} {r['K']:>6} {r['Q']:>8.3f} {r['FFL']:>8.4f} "
          f"{r['GO%']:>7.1%} {r['KEGG%']:>7.1%} {r['mean_GO']:>8.1f} {r['tested']:>7} "
          f"{r['runtime']:>7.1f}s", flush=True)

# Save
out_path = '/mnt/d/cluster/BIologicalNetworks/results/grn_enrichment_ranking.csv'
df_out.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}", flush=True)
