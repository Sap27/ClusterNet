# ClusterNet GNN Integration Design

## Overview

This document outlines the integration of Graph Neural Networks into ClusterNet for community detection, with a focus on:
1. **Diverse GNN architectures** (mix of existing + novel implementations)
2. **Multi-scale benchmarking** (synthetic, rewired-real, biological)
3. **Robustness evaluation** (perturbation studies)

---

## 🧠 GNN Architecture Selection

### From RobustGNN (Adapted, not copied):
| Model | Type | Why Include |
|-------|------|-------------|
| **GCN** | Message Passing | Foundation model, Kipf & Welling 2017 |
| **DMoN** | Unsupervised Pooling | Best for modularity optimization |
| **MinCutPool** | Spectral Pooling | Graph-cut based clustering |

### New Implementations (Not in RobustGNN):
| Model | Type | Source/Paper | Why Include |
|-------|------|--------------|-------------|
| **GIN** | Message Passing | Xu et al. 2019 | Most expressive GNN |
| **GraphTransformer** | Attention | Dwivedi 2021 | State-of-the-art attention |
| **VGAE** | Generative | Kipf 2016 | Unsupervised embeddings (TODO) |
| **GRACE** | Contrastive | Zhu 2020 | Self-supervised clustering (TODO) |
| **BGRL** | Self-supervised | Thakoor 2021 | No negative samples (TODO) |

### Total: 8 GNN Methods
- 3 adapted from RobustGNN (GCN, DMoN, MinCut)
- 2 new implementations (GIN, GraphTransformer)
- 3 planned (VGAE, GRACE, BGRL)

---

## 📊 Benchmark Design

### Benchmark 1: Synthetic (LFR + SBM + Hierarchical)

```
synthetic/
├── lfr/
│   ├── varying_mu/          # μ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}
│   ├── varying_size/        # n ∈ {1K, 5K, 10K, 50K}
│   └── varying_degree/      # avg_degree ∈ {10, 25, 50}
├── sbm/
│   ├── assortative/         # p_in > p_out
│   ├── disassortative/      # p_in < p_out
│   └── degree_corrected/    # DC-SBM with heterogeneity
└── hierarchical/
    ├── nested_sbm/          # Multi-level communities
    └── hierarchical_lfr/    # Recursive community structure
```

### Benchmark 2: Rewired Real Networks

```
rewired_real/
├── social/
│   ├── karate_rewired/
│   ├── football_rewired/
│   └── polbooks_rewired/
├── citation/
│   ├── cora_rewired/
│   ├── citeseer_rewired/
│   └── pubmed_rewired/
└── biological/
    ├── ppi_rewired/
    └── yeast_rewired/

Rewiring levels: {0%, 25%, 50%, 75%, 100%}
```

### Benchmark 3: Biological Networks (Motif Enrichment)

```
biological/
├── grn/                     # Gene Regulatory Networks
│   ├── ecoli_grn/
│   ├── yeast_grn/
│   └── human_grn/
├── ppi/                     # Protein-Protein Interaction
│   ├── string_ppi/
│   ├── biogrid_ppi/
│   └── dip_ppi/
└── single_cell/             # Single-cell RNA-seq
    ├── pbmc_3k/
    ├── mouse_brain/
    └── covid_atlas/

Motifs analyzed:
- Triangles (3-cliques)
- Feed-forward loops
- Bi-fan motifs
- Feedback loops
- 4-cliques
```

---

## 🔬 Perturbation Studies

### Feature Perturbations:
1. **Mean shift**: Add constant to features
2. **Variance scaling**: Scale feature variance
3. **Feature dropout**: Randomly mask features
4. **Feature noise**: Add Gaussian noise

### Structural Perturbations:
1. **Random edge removal**: Uniform dropout
2. **Targeted edge removal**: High-betweenness edges
3. **Edge addition**: Random spurious edges
4. **Community mixing**: Cross-community edges

### Biological-Specific:
1. **Hub node perturbation**: Target high-degree proteins
2. **Pathway disruption**: Remove pathway-specific edges
3. **Expression noise**: Simulate measurement error

---

## 📁 Directory Structure

```
clusternet/
├── gnn/
│   ├── __init__.py
│   ├── base_gnn.py              # Base class for GNN clustering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dmon.py              # DMoN (adapted)
│   │   ├── mincut.py            # MinCut pooling (adapted)
│   │   ├── gin_cluster.py       # GIN for clustering (NEW)
│   │   ├── graph_transformer.py # Transformer (NEW)
│   │   ├── vgae_cluster.py      # VGAE clustering (NEW)
│   │   ├── grace_cluster.py     # Contrastive (NEW)
│   │   ├── bgrl_cluster.py      # Self-supervised (NEW)
│   │   └── cluster_gcn.py       # Scalable GCN (NEW)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Unified training loop
│   │   ├── losses.py            # Custom losses (modularity, etc.)
│   │   └── schedulers.py        # Learning rate schedules
│   └── utils/
│       ├── __init__.py
│       ├── pyg_conversion.py    # NetworkX <-> PyG
│       └── feature_generation.py # Synthetic features
│
├── benchmarks/
│   ├── __init__.py
│   ├── synthetic/
│   │   ├── lfr_generator.py
│   │   ├── sbm_generator.py
│   │   └── hierarchical_generator.py
│   ├── rewired/
│   │   ├── rewiring.py          # Edge rewiring algorithms
│   │   └── real_datasets.py     # Load real networks
│   └── biological/
│       ├── grn_loader.py        # Gene regulatory networks
│       ├── ppi_loader.py        # PPI networks
│       ├── single_cell.py       # scRNA-seq networks
│       └── motif_analysis.py    # Motif enrichment
│
├── robustness/
│   ├── __init__.py
│   ├── perturbations.py         # Perturbation functions
│   ├── attacks.py               # Adversarial attacks
│   └── evaluator.py             # Robustness metrics
│
└── evaluation/
    ├── __init__.py
    ├── partition_metrics.py     # NMI, AMI, ARI
    ├── motif_metrics.py         # Motif enrichment scores
    ├── topology_metrics.py      # Topology preservation
    └── coherence.py             # Multi-scale coherence
```

---

## 🎯 Implementation Priority

### Phase 1: Core GNN Infrastructure
1. Base GNN class with ClusterNet interface
2. DMoN adaptation (primary unsupervised method)
3. GIN implementation (most expressive)
4. PyG <-> NetworkX conversion utilities

### Phase 2: Benchmarks
1. LFR generator with perturbations
2. Rewired real network suite
3. Basic motif analysis

### Phase 3: Advanced GNNs
1. Graph Transformer
2. VGAE clustering
3. Contrastive methods (GRACE, BGRL)

### Phase 4: Biological Focus
1. GRN/PPI loaders
2. Single-cell network construction
3. Full motif enrichment analysis

---

## 📈 Evaluation Protocol

### Metrics per Benchmark:

| Benchmark | Primary Metrics | Secondary Metrics |
|-----------|----------------|-------------------|
| LFR/SBM | NMI, AMI, ARI | Modularity, Runtime |
| Rewired | NMI vs rewiring % | Topology preservation |
| Biological | Motif enrichment | GO enrichment, Pathway overlap |
| Robustness | Δ Performance | Recovery rate |

### Multi-Scale Coherence Score:

```
Coherence = α × Motif_Score + β × Partition_Score + γ × Topology_Score

Where:
- α, β, γ are weights (default: 0.33 each)
- Motif_Score: Average motif enrichment z-score
- Partition_Score: AMI with ground truth
- Topology_Score: Correlation of network statistics
```

