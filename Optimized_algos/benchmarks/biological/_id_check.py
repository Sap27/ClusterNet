"""
Check gene/protein ID formats across all data sources to verify overlap.
"""
import csv, gzip, os

BIO = '/mnt/d/cluster/BIologicalNetworks'

def sample_ids(path, cols=(0, 1), n=5, skip_header=True):
    """Return first n unique IDs from given columns of a CSV."""
    ids = set()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader)
        for row in reader:
            for c in cols:
                ids.add(row[c])
            if len(ids) >= n:
                break
    return sorted(ids)[:n]

def get_all_ids(path, cols=(0, 1), skip_header=True):
    """Return all unique IDs from given columns."""
    ids = set()
    with open(path, 'r') as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader)
        for row in reader:
            for c in cols:
                ids.add(row[c])
    return ids

# --- 1. Network gene IDs ---
print("=" * 70)
print("NETWORK GENE ID FORMATS")
print("=" * 70)

networks = {
    'GTEx_0.8': f'{BIO}/GTEx_Benchmark_0.7.csv',
    'Human_GRN': f'{BIO}/Human_GRN_Weighted.csv',
    'BioGRID': f'{BIO}/BioGRID_Experimental.csv',
    'SIGNOR': f'{BIO}/SIGNOR_Directed.csv',
}

net_ids = {}
for name, path in networks.items():
    samples = sample_ids(path, (0, 1), n=10)
    all_ids = get_all_ids(path, (0, 1))
    net_ids[name] = all_ids
    print(f"\n{name} ({len(all_ids):,} unique IDs):")
    print(f"  Samples: {samples}")

# --- 2. GTEx TPM gene IDs ---
print("\n" + "=" * 70)
print("GTEx TPM MATRIX GENE IDs")
print("=" * 70)

tpm_path = f'{BIO}/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm (2).gct'
tpm_genes_ensembl = []
tpm_genes_symbol = []
with open(tpm_path, 'r') as f:
    f.readline()  # #1.2
    f.readline()  # dimensions
    header = f.readline()  # Name, Description, samples...
    for i, line in enumerate(f):
        parts = line.split('\t', 3)
        tpm_genes_ensembl.append(parts[0])       # ENSG...
        tpm_genes_symbol.append(parts[1])          # Description = gene symbol
        if i >= 9:
            break

print(f"\nFirst 10 Ensembl IDs:  {tpm_genes_ensembl}")
print(f"First 10 Descriptions: {tpm_genes_symbol}")

# Read all TPM gene symbols
tpm_all_ensembl = set()
tpm_all_symbols = set()
with open(tpm_path, 'r') as f:
    f.readline()
    f.readline()
    f.readline()
    for line in f:
        parts = line.split('\t', 3)
        ens = parts[0].split('.')[0]  # strip version: ENSG00000186092.7 -> ENSG00000186092
        tpm_all_ensembl.add(ens)
        tpm_all_symbols.add(parts[1])
print(f"\nTotal TPM genes: {len(tpm_all_ensembl)} Ensembl, {len(tpm_all_symbols)} symbols")

# --- 3. GO Annotation gene IDs ---
print("\n" + "=" * 70)
print("GO ANNOTATION (GAF) GENE IDs")
print("=" * 70)

gaf_path = f'{BIO}/goa_human.gaf.gz'
gaf_symbols = set()
gaf_uniprots = set()
gaf_samples_sym = []
gaf_samples_uni = []
with gzip.open(gaf_path, 'rt') as f:
    for line in f:
        if line.startswith('!'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 11:
            continue
        uniprot = parts[1]   # DB Object ID (UniProt accession)
        symbol = parts[2]    # DB Object Symbol (gene symbol)
        synonyms = parts[10] # DB Object Synonyms
        gaf_uniprots.add(uniprot)
        gaf_symbols.add(symbol)
        if len(gaf_samples_sym) < 10:
            gaf_samples_sym.append(symbol)
            gaf_samples_uni.append(uniprot)

print(f"\nTotal GAF entries: {len(gaf_uniprots)} UniProt IDs, {len(gaf_symbols)} gene symbols")
print(f"Sample UniProt IDs: {gaf_samples_uni[:10]}")
print(f"Sample gene symbols: {gaf_samples_sym[:10]}")

# --- 4. Cross-reference overlaps ---
print("\n" + "=" * 70)
print("OVERLAP ANALYSIS")
print("=" * 70)

for name, ids in net_ids.items():
    # vs GAF symbols
    overlap_sym = ids & gaf_symbols
    # vs GAF UniProt
    overlap_uni = ids & gaf_uniprots
    # vs TPM symbols
    overlap_tpm_sym = ids & tpm_all_symbols
    # vs TPM Ensembl
    overlap_tpm_ens = ids & tpm_all_ensembl

    print(f"\n{name} ({len(ids):,} IDs):")
    print(f"  vs GAF symbols:     {len(overlap_sym):>6,} / {len(ids):,}  ({100*len(overlap_sym)/len(ids):.1f}%)")
    print(f"  vs GAF UniProt:     {len(overlap_uni):>6,} / {len(ids):,}  ({100*len(overlap_uni)/len(ids):.1f}%)")
    print(f"  vs TPM symbols:     {len(overlap_tpm_sym):>6,} / {len(ids):,}  ({100*len(overlap_tpm_sym)/len(ids):.1f}%)")
    print(f"  vs TPM Ensembl:     {len(overlap_tpm_ens):>6,} / {len(ids):,}  ({100*len(overlap_tpm_ens)/len(ids):.1f}%)")
