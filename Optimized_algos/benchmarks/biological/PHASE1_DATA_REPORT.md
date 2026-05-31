# Phase 1: Data Preparation Report

## 1. Raw Data Inventory

| Network | File | Nodes | Edges | Avg Degree | Type | ID Format |
|---|---|---|---|---|---|---|
| GTEx co-expression | `GTEx_Full_Benchmark_2 (1).csv` | 24,499 | 20,324,998 | 1659.3 | Undirected, weighted (Pearson) | Gene symbols |
| Human GRN | `Human_GRN_Weighted.csv` | 18,609 | 278,660 | 29.9 | Directed, weighted (confidence) | UniProt accessions |
| BioGRID PPI | `BioGRID_Experimental.csv` | 17,473 | 944,933 | 108.2 | Undirected, unweighted | Gene symbols |
| SIGNOR signaling | `SIGNOR_Directed.csv` | 5,837 | 17,484 | 6.0 | Directed, signed (+1/-1) | Gene symbols |

**External resources:**
- GTEx TPM matrix: `GTEx_...gene_tpm.gct` (56,200 genes x 17,382 samples, 4.3 GB)
- GO annotations: `goa_human.gaf.gz` (38,919 UniProt IDs, 38,816 gene symbols)
- GO ontology: `go-basic.obo` (hierarchy for enrichment tools)

## 2. ID Cross-Reference Analysis

Three ID namespaces exist across our data sources:
- **Gene symbols** (e.g., STAT3, PIK3CD): used by GTEx network, BioGRID, SIGNOR, GTEx TPM, and GAF
- **UniProt accessions** (e.g., O00716, P05412): used by Human GRN and GAF
- **Ensembl IDs** (e.g., ENSG00000186092): used by GTEx TPM (column 1), not used by any network

The GAF file serves as the bridge: it contains both UniProt accessions and gene symbols per entry, allowing UniProt-based GRN nodes to be mapped to gene symbols for TPM feature lookup.

### Overlap Matrix

| Network (ID format) | vs GO/GAF symbols | vs GO/GAF UniProt | vs GTEx TPM symbols |
|---|---|---|---|
| **GTEx** (symbols) | 14,183 / 20,669 (68.6%) | 0% | **20,669 / 20,669 (100%)** |
| **Human GRN** (UniProt) | 0% | **18,196 / 18,609 (97.8%)** | via mapping: **17,349 / 18,609 (93.2%)** |
| **BioGRID** (symbols) | **15,896 / 17,473 (91.0%)** | 0% | 17,130 / 17,473 (98.0%) |
| **SIGNOR** (symbols) | **5,818 / 5,837 (99.7%)** | 0% | 5,833 / 5,837 (99.9%) |

Bold values indicate the primary match path used for each network's features.

### GRN ID Mapping Detail

The Human GRN uses UniProt accessions. The mapping chain is:

```
GRN UniProt ID  -->  GAF (UniProt->symbol)  -->  GTEx TPM (symbol->expression)
  18,609 nodes       18,196 mapped (97.8%)       17,349 with expression (93.2%)
```

- **413 unmapped** UniProt IDs: likely deprecated or secondary accessions not in the current GAF release.
- **846 mapped but absent from TPM**: mostly due to HGNC gene symbol updates (e.g., AARS renamed to AARS1).
- **36 symbol collisions**: multiple UniProt isoforms map to the same gene symbol (e.g., GNAS has 4 isoforms). Each UniProt node keeps its own copy of the expression vector.

### GRN Node Roles

| Role | Count | Description |
|---|---|---|
| Pure TFs (source only) | 1 | Only regulate, never regulated |
| Pure targets (target only) | 17,964 | Only regulated, never regulate |
| Both TF and target | 644 | Regulate others and are themselves regulated |

The network is highly asymmetric: 645 TFs regulate 18,608 targets, with most TFs also being targets of other TFs (feed-forward architecture).

## 3. Feature Strategy

| Network | Feature source | Feature type | Expected dimension |
|---|---|---|---|
| **GTEx 0.8** | GTEx TPM matrix | PCA-reduced log2(TPM+1) expression profiles | 128 |
| **Human GRN** | GTEx TPM via UniProt->symbol mapping | PCA-reduced log2(TPM+1) expression + TF/target binary | ~130 |
| **BioGRID** | GO annotations (GAF) | Binary GO term vectors (filtered 10-1000 genes/term) | ~2,000-3,000 |
| **SIGNOR** | GO annotations + edge signs | Binary GO term vectors + sign-aware in-degree (activator/inhibitor) | ~2,000-3,000 + 2 |

## 4. Preprocessing Steps

| Step | GTEx 0.8 | Human GRN | BioGRID | SIGNOR |
|---|---|---|---|---|
| Threshold edges | Pearson >= 0.8 | — | — | — |
| Map IDs | — | UniProt -> symbol (GAF) | — | — |
| Build features | TPM -> PCA(128) | TPM -> PCA(128) + TF role | GAF -> GO binary | GAF -> GO binary + sign degree |
| Filter to featured nodes | Drop genes not in TPM | Drop unmapped UniProts | Drop genes not in GAF | Drop genes not in GAF |
| Symmetrize | — (already undirected) | Directed -> undirected | — (already undirected) | Directed -> undirected |
| Save directed copy | — | Yes (for FFL evaluation) | — | Yes (for sign coherence) |
| Enumerate FFLs | — | Yes | — | — |

## 5. Expected Output

```
data/
├── gtex_0.8/
│   ├── edges.npy          # undirected edge list with weights
│   ├── features.npy       # (N, 128) expression features
│   ├── node_list.json     # gene symbol -> index mapping
│   └── metadata.json
├── grn/
│   ├── edges.npy          # symmetrized undirected edges
│   ├── features.npy       # (N, ~130) expression + TF role
│   ├── node_list.json     # UniProt ID -> index mapping
│   ├── directed_edges.json # original directed edges for FFL eval
│   ├── ffls.json          # pre-computed feed-forward loop triples
│   └── metadata.json
├── biogrid/
│   ├── edges.npy
│   ├── features.npy       # (N, ~2500) GO term binary
│   ├── go_terms.npy       # GO term ID list
│   ├── node_list.json
│   └── metadata.json
├── signor/
│   ├── edges.npy          # symmetrized undirected edges
│   ├── features.npy       # (N, ~2382) GO terms + sign degree
│   ├── go_terms.npy
│   ├── node_list.json
│   ├── directed_edges.json
│   └── metadata.json
└── metadata.json           # combined metadata for all networks
```
