# ClusterNet GNN Integration Overview

## 🎯 Design Philosophy

The GNN integration follows these principles:
1. **Diversity**: Mix of architectures from RobustGNN (adapted) + new implementations
2. **Non-duplication**: Original code, not direct copies
3. **Unified interface**: All GNNs follow ClusterNet's `BaseAlgorithm` pattern
4. **Benchmark-focused**: Built-in support for the 3 benchmark categories

---

## 🧠 GNN Architectures (8 Total)

### From RobustGNN (Adapted - 2):
| Model | Original Use | Our Modification |
|-------|-------------|------------------|
| **DMoN** | Node classification | Pure clustering with modularity loss |
| **MinCutPool** | Graph classification | Adapted for node clustering |

### New Implementations (6):
| Model | Paper | Key Innovation |
|-------|-------|----------------|
| **GIN** | Xu et al. 2019 | Most expressive (WL-equivalent) |
| **Graph Transformer** | Dwivedi 2021 | Attention + Laplacian positional encoding |
| **VGAE** | Kipf 2016 | Variational embeddings (TODO) |
| **GRACE** | Zhu 2020 | Contrastive learning (TODO) |
| **BGRL** | Thakoor 2021 | Bootstrap without negatives (TODO) |
| **ClusterGCN** | Chiang 2019 | Scalable mini-batching (TODO) |

### Implementation Status:
- ✅ DMoN - Complete
- ✅ MinCut - Complete  
- ✅ GIN - Complete
- ✅ Graph Transformer - Complete
- 🔲 VGAE - Planned
- 🔲 GRACE - Planned
- 🔲 BGRL - Planned
- 🔲 ClusterGCN - Planned

---

## 📊 Benchmark Framework

### Benchmark 1: Synthetic (LFR + SBM + Hierarchical)

```python
from clusternet.benchmarks import LFRBenchmark, SBMBenchmark

# LFR with varying mixing parameter
lfr = LFRBenchmark(n=1000, mu=0.3)
G, communities, labels = lfr.generate()

# SBM with degree correction
sbm = SBMBenchmark(n=1000, k=10, degree_corrected=True)
G, communities, labels = sbm.generate()

# Hierarchical SBM
from clusternet.benchmarks.synthetic.sbm_generator import HierarchicalSBM
hier = HierarchicalSBM(n=1000, levels=2, branching=3)
```

**Perturbation Support:**
- Edge removal (random, targeted)
- Community mixing
- Feature noise

### Benchmark 2: Rewired Real Networks

```python
from clusternet.benchmarks import RewiredBenchmark

# Load real network
G = nx.karate_club_graph()

# Create rewired versions
benchmark = RewiredBenchmark(G)
results = benchmark.rewire_sweep(fractions=[0.0, 0.25, 0.5, 0.75, 1.0])

# Each result has known ground truth + topology preservation metrics
```

**Key insight**: Bridges synthetic-real gap by gradually imposing known structure.

### Benchmark 3: Biological Networks (Motif Enrichment)

```python
from clusternet.benchmarks.biological import compute_motif_enrichment

# Evaluate communities by motif preservation
scores = compute_motif_enrichment(G, communities)
# Returns: triangle_mean_z, feedforward_mean_z, etc.
```

**Supported motifs:**
- Triangles (3-cliques)
- 4-cycles (squares)
- Feed-forward loops (directed)
- Bi-fan motifs (directed)

---

## 🔬 Robustness Evaluation

```python
from clusternet.robustness import RobustnessEvaluator

evaluator = RobustnessEvaluator(G, ground_truth, features)

# Single evaluation
result = evaluator.evaluate_single(algorithm, 'edge_random', level=0.2)

# Sweep over levels
results = evaluator.evaluate_sweep(algorithm, 'edge_targeted', levels=[0.1, 0.2, 0.3])

# Compare algorithms
comparison = evaluator.compare_algorithms(
    {'Louvain': louvain_func, 'DMoN': dmon_func},
    'edge_random'
)
```

**Perturbation Types:**
- `edge_random`: Random edge removal
- `edge_targeted`: High-betweenness edge removal
- `feature_mean`: Mean shift in features
- `feature_variance`: Variance scaling
- `feature_dropout`: Random feature masking

---

## 📁 Directory Structure

```
clusternet/
├── gnn/
│   ├── __init__.py
│   ├── base_gnn.py              # BaseGNNClustering class
│   ├── models/
│   │   ├── dmon.py              # ✅ DMoN (adapted)
│   │   ├── mincut.py            # ✅ MinCut (adapted)
│   │   ├── gin_cluster.py       # ✅ GIN (NEW)
│   │   └── graph_transformer.py # ✅ Graph Transformer (NEW)
│   ├── training/
│   │   └── trainer.py           # Unified training loop
│   └── utils/
│       └── pyg_conversion.py    # NetworkX <-> PyG
│
├── benchmarks/
│   ├── synthetic/
│   │   ├── lfr_generator.py     # ✅ LFR with perturbations
│   │   └── sbm_generator.py     # ✅ SBM + Hierarchical
│   ├── rewired/
│   │   └── rewiring.py          # ✅ Real network rewiring
│   └── biological/
│       └── motif_analysis.py    # ✅ Motif enrichment
│
├── robustness/
│   ├── perturbations.py         # ✅ All perturbation functions
│   └── evaluator.py             # ✅ Robustness framework
│
└── evaluation/
    └── (partition metrics)
```

---

## 🎯 Research Paper Framework

### Three-Pillar Evaluation:

1. **Synthetic Benchmarks (Controlled)**
   - LFR varying μ: 0.1 → 0.5
   - SBM varying SNR: 1 → 8
   - Hierarchical: 2-3 levels
   
2. **Rewired Real Networks (Semi-Synthetic)**
   - Social: Karate, Football, PolBooks
   - Citation: Cora, CiteSeer
   - Biological: PPI, Yeast
   - Rewiring: 0%, 25%, 50%, 75%, 100%

3. **Biological Networks (Real + Motifs)**
   - Gene Regulatory Networks
   - Protein-Protein Interaction
   - Single-cell derived networks
   - Motif enrichment scoring

### Proposed Experiments:

| Experiment | Networks | Methods | Metrics |
|------------|----------|---------|---------|
| Synthetic accuracy | LFR, SBM | All 8 GNN + 5 traditional | NMI, AMI, ARI |
| Scalability | LFR (1K-100K) | All methods | Runtime, Memory |
| Robustness | LFR, Rewired | All methods | Recovery rate |
| Motif preservation | PPI, GRN | All methods | Z-scores |
| Hierarchy detection | H-SBM | All methods | Multi-level NMI |

---

## 🚀 Quick Start

```python
# Traditional + GNN comparison
import networkx as nx
from clusternet.algorithms import louvain_algorithm
from clusternet.gnn.models import dmon_clustering, gin_clustering
from clusternet.benchmarks import LFRBenchmark

# Generate benchmark
lfr = LFRBenchmark(n=500, mu=0.3)
G, communities, labels = lfr.generate()
features = lfr.features

# Run methods
louvain_result, _ = louvain_algorithm(G)
dmon_result, _ = dmon_clustering(G, num_clusters=len(communities), features=features)
gin_result, _ = gin_clustering(G, num_clusters=len(communities), features=features)

# Evaluate
from sklearn.metrics import normalized_mutual_info_score
# ... compute NMI for each
```

---

## Dependencies

**Core:**
- `networkx >= 2.6`
- `numpy >= 1.20`
- `scipy >= 1.7`
- `scikit-learn >= 1.0`

**GNN (Optional):**
- `torch >= 1.10`
- `torch-geometric >= 2.0`

**Visualization (Optional):**
- `matplotlib >= 3.4`
- `seaborn >= 0.11`

---

## Next Steps

1. **Complete GNN implementations**: VGAE, GRACE, BGRL, ClusterGCN
2. **Add biological data loaders**: STRING PPI, BioGRID, scRNA-seq
3. **Implement adversarial attacks**: Nettack, Metattack
4. **Build experiment runner**: Automated benchmark suite
5. **Create visualization dashboard**: Results comparison

---

## Key Innovation Claims

1. **Unified Framework**: First to combine traditional + GNN methods with consistent interface
2. **Multi-Scale Evaluation**: Synthetic → Rewired → Real with motif analysis
3. **Robustness Focus**: Systematic perturbation studies across method classes
4. **Biological Relevance**: Motif enrichment as community quality metric









