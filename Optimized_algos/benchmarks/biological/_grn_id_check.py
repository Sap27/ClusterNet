"""
Deep check of Human GRN ID resolution:
1. What IDs does the GRN use? (UniProt accessions)
2. How many map to gene symbols via GAF?
3. How many of those symbols exist in the GTEx TPM matrix?
4. What's left unmapped?
"""
import csv, gzip
from collections import defaultdict

BIO = '/mnt/d/cluster/BIologicalNetworks'

# 1. Load GRN node IDs
print("1. Loading GRN edge list...", flush=True)
grn_nodes = set()
grn_sources = set()  # potential TFs (have outgoing edges)
grn_targets = set()
with open(f'{BIO}/Human_GRN_Weighted.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        src, tgt = row[0], row[1]
        grn_nodes.add(src)
        grn_nodes.add(tgt)
        grn_sources.add(src)
        grn_targets.add(tgt)

pure_tfs = grn_sources - grn_targets
pure_targets = grn_targets - grn_sources
both = grn_sources & grn_targets

print(f"   Total nodes: {len(grn_nodes):,}")
print(f"   Pure TFs (source only): {len(pure_tfs):,}")
print(f"   Pure targets (target only): {len(pure_targets):,}")
print(f"   Both (TF and target): {len(both):,}")
print(f"   Sample IDs: {sorted(list(grn_nodes))[:15]}")

# 2. Build UniProt -> symbol mapping from GAF
print("\n2. Building UniProt -> symbol from GAF...", flush=True)
uni_to_sym = {}
sym_to_unis = defaultdict(set)
with gzip.open(f'{BIO}/goa_human.gaf.gz', 'rt') as f:
    for line in f:
        if line.startswith('!'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 11:
            continue
        uniprot = parts[1]
        symbol = parts[2]
        if uniprot not in uni_to_sym:
            uni_to_sym[uniprot] = symbol
        sym_to_unis[symbol].add(uniprot)

print(f"   GAF has {len(uni_to_sym):,} UniProt -> symbol mappings")

# 3. Map GRN nodes to symbols
print("\n3. Mapping GRN nodes to gene symbols...", flush=True)
mapped = {}
unmapped = set()
for node in grn_nodes:
    if node in uni_to_sym:
        mapped[node] = uni_to_sym[node]
    else:
        unmapped.add(node)

mapped_symbols = set(mapped.values())
print(f"   Mapped: {len(mapped):,} / {len(grn_nodes):,} ({100*len(mapped)/len(grn_nodes):.1f}%)")
print(f"   Unique symbols after mapping: {len(mapped_symbols):,}")
print(f"   Unmapped: {len(unmapped):,}")
if unmapped:
    print(f"   Sample unmapped IDs: {sorted(list(unmapped))[:20]}")

# Check for symbol collisions (multiple UniProts -> same symbol)
sym_counts = defaultdict(int)
for sym in mapped.values():
    sym_counts[sym] += 1
collisions = {s: c for s, c in sym_counts.items() if c > 1}
print(f"   Symbol collisions (multiple UniProts -> same symbol): {len(collisions)}")
if collisions:
    top_collisions = sorted(collisions.items(), key=lambda x: -x[1])[:10]
    for sym, cnt in top_collisions:
        print(f"     {sym}: {cnt} UniProt IDs")

# 4. Check mapped symbols against GTEx TPM
print("\n4. Checking mapped symbols vs GTEx TPM...", flush=True)
tpm_symbols = set()
tpm_path = f'{BIO}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm (2).gct'
with open(tpm_path, 'r') as f:
    f.readline()  # #1.2
    f.readline()  # dims
    f.readline()  # header
    for line in f:
        sym = line.split('\t', 3)[1]
        tpm_symbols.add(sym)

overlap = mapped_symbols & tpm_symbols
print(f"   Mapped GRN symbols in TPM: {len(overlap):,} / {len(mapped_symbols):,} "
      f"({100*len(overlap)/len(mapped_symbols):.1f}%)")

# Full chain: GRN node -> symbol -> TPM
full_chain = {node for node, sym in mapped.items() if sym in tpm_symbols}
print(f"\n   FULL PIPELINE: GRN nodes with expression features: "
      f"{len(full_chain):,} / {len(grn_nodes):,} "
      f"({100*len(full_chain)/len(grn_nodes):.1f}%)")

missing_from_tpm = mapped_symbols - tpm_symbols
if missing_from_tpm:
    print(f"   Symbols mapped but not in TPM ({len(missing_from_tpm)}): "
          f"{sorted(list(missing_from_tpm))[:20]}")
