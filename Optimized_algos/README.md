# ClusterNet

**Disentangling When and Why Graph Neural Networks Surpass Classical Community Detection on Biological Networks**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-2.0+-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

ClusterNet is a unified benchmarking framework that evaluates **30 classical** and **8 GNN-based** community detection algorithms under identical experimental conditions across three tiers of increasing biological realism. It accompanies the paper submitted to the [HICSS-60 Minitrack on AI Analysis in Network Biology](https://hicss.hawaii.edu/).

---

## Key Findings

| Finding | Evidence |
|---------|----------|
| GNNs are **conditionally** better | They require informative node features + sparse topology |
| Biological features boost GNN performance | GO annotations improve modularity by 56–100% on sparse regulatory networks |
| GAT learns regulatory specificity | Attention correlates negatively with TF out-degree (ρ = −0.92, p < 10⁻²⁶⁹) |
| DGI is the most perturbation-robust GNN | NMI = 0.51 under TF knockout; supervised GNNs collapse (NMI < 0.14) |
| Dense networks nullify GNN advantage | On GTEx (1.4M edges), Leiden matches or exceeds GNN performance |

---

## Three-Tier Evaluation

```
Tier 1: LFR Synthetic       →  Controlled baseline (no features, known ground truth)
Tier 2: Social Networks      →  Feature ablation (Cora, CiteSeer, Amazon Photo)
Tier 3: Biological Networks  →  Real test (SIGNOR, GRN, BioGRID, GTEx)
```

**Tier 1** establishes that GNNs underperform classical methods without informative features.  
**Tier 2** quantifies the feature contribution via systematic ablation.  
**Tier 3** evaluates biological coherence (GO/KEGG enrichment, FFL preservation, sign coherence, perturbation robustness) and demonstrates GNN interpretability through attention weight and embedding analysis.

---

## Repository Structure

```
ClusterNet/
├── paper/                          # Manuscript, figures, and tables
│   ├── main.tex                    # Main paper (LaTeX)
│   ├── supplementary.tex           # Supplementary material
│   ├── references.bib              # Bibliography
│   ├── figures/                    # All PDF figures
│   ├── tables/                     # All LaTeX tables
│   ├── plot_lfr.py                 # LFR figure generation
│   ├── plot_social.py              # Social figure generation
│   ├── plot_bio.py                 # Biological figure generation
│   └── validate_*.py              # Manuscript number validation scripts
├── Optimized_algos/
│   ├── clusternet/                 # Core package
│   │   ├── algorithms/             # Classical algorithm implementations
│   │   │   ├── cdlib_wrappers/     # CDlib integrations
│   │   │   └── ...                 # Custom implementations
│   │   ├── gnn/                    # GNN implementations
│   │   │   ├── base_gnn.py         # Base class with embedding export
│   │   │   └── models/             # DMoN, MinCut, VGAE, DGI, GAT, GCN, ...
│   │   └── utils/                  # Evaluation metrics, graph I/O
│   ├── benchmarks/
│   │   ├── lfr/                    # Tier 1: LFR benchmarks
│   │   ├── social/                 # Tier 2: Social network benchmarks
│   │   └── biological/             # Tier 3: Biological network benchmarks
│   │       ├── run_benchmark.py    # Main biological benchmark runner
│   │       ├── run_perturbation.py # Perturbation analysis
│   │       ├── run_hicss_experiments.sh  # GPU experiment orchestrator
│   │       ├── plot_bio_feature_ablation.py
│   │       ├── plot_bio_embeddings.py
│   │       ├── analyse_attention.py
│   │       └── config.py           # Data paths configuration
│   ├── requirements-clusternet.txt
│   └── setup.py
```

---

## Algorithms (38 Total)

### Classical (30)

| Family | Algorithms |
|--------|-----------|
| Modularity | Louvain, Leiden, FastGreedy, Walktrap |
| Information-theoretic | Infomap |
| Spectral | Spectral Clustering, Leading Eigenvector |
| Statistical inference | SBM, Nested SBM, EM |
| Label propagation / local | Label Propagation, Spinglass, SCAN, Async Fluid, Surprise |
| Resolution-based | CPM, RB-Pots, RBer-Pots |
| Overlapping | Angel, Demon, k-Clique |
| DREAM-derived | SCORE, TeamCS, BigS2, CSBIO-IITM2 |
| Other heuristics | AGDL, DER, GDMP2, SimNet, SVT, Tusk |

### GNN-Based (8)

| Category | Models | Features | Labels |
|----------|--------|----------|--------|
| Differentiable pooling | DMoN, MinCutPool | Yes | No |
| Representation learning | VGAE+KM, DGI+KM | Yes | No |
| Semi-supervised | GCN, GAT | Yes | Yes |
| Baselines | MLP+KM (features only), Node2Vec+KM (topology only) | Varies | No |

All GNN models: 2-layer encoder, hidden dim 128, 600 epochs, Adam (lr = 10⁻³), PyTorch Geometric.

---

## Installation

```bash
cd Optimized_algos

# Core dependencies
pip install -r requirements-clusternet.txt

# Install package in editable mode
pip install -e .

# GNN support (GPU)
pip install torch torchvision torchaudio
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# SBM algorithms (requires conda)
conda install -c conda-forge graph-tool
```

---

## Reproducing Experiments

### Tier 1: LFR Benchmarks

```bash
cd benchmarks/lfr
python run_accuracy_benchmark.py        # Accuracy vs mixing parameter
python run_scalability_benchmark.py     # Scalability (N=500 to 25,000)
python run_hierarchical_benchmark.py    # Hierarchical community detection
```

### Tier 2: Social Networks

```bash
cd benchmarks/social
python data_preparation.py              # Download Cora, CiteSeer, Amazon Photo
python run_benchmark.py                 # Run all algorithms + feature ablation
```

### Tier 3: Biological Networks

```bash
cd benchmarks/biological

# Configure data paths
# Edit config.py: BIO_DATA_DIR = Path('/path/to/BIologicalNetworks')

# Run full experiment suite (GPU recommended)
bash run_hicss_experiments.sh

# Or run individual experiments:
python run_benchmark.py --experiment ablation      # Feature ablation
python run_benchmark.py --experiment embeddings    # Embedding export
python run_benchmark.py --experiment attention     # GAT attention export
python run_perturbation.py                         # Perturbation analysis
```

### Post-Processing and Figure Generation

```bash
# Biological post-processing
python plot_bio_feature_ablation.py --results-dir /path/to/results
python plot_bio_embeddings.py --results-dir /path/to/results
python analyse_attention.py --results-dir /path/to/results

# Paper figures
cd ../../paper
python plot_lfr.py
python plot_social.py
python plot_bio.py
```

---

## Quick Start (Python API)

```python
from clusternet import CommunityDetector
import networkx as nx

G = nx.karate_club_graph()

# Classical
detector = CommunityDetector(algorithm='leiden', resolution=1.0)
communities = detector.detect(G)

# GNN (requires node features)
from clusternet.gnn.models import DMoNCluster
model = DMoNCluster(n_clusters=4, device='cuda')
communities = model.fit_predict(G, features=X)
```

---

## Evaluation Metrics

| Metric | Type | Used In |
|--------|------|---------|
| AMI / NMI | Partition similarity | Tiers 1, 2, 3 (perturbation) |
| Modularity | Topological quality | All tiers |
| GO/KEGG enrichment | Biological coherence | Tier 3 |
| FFL preservation | Regulatory motif | Tier 3 (SIGNOR, GRN) |
| Sign coherence | Regulatory consistency | Tier 3 (SIGNOR) |
| Disease module overlap | Clinical relevance | Tier 3 (BioGRID) |
| Perturbation NMI | Stability | Tier 3 (GRN) |

---

## Biological Networks

| Network | Nodes | Edges | Type | Features |
|---------|-------|-------|------|----------|
| SIGNOR | 4,060 | 34,104 | Signed signaling | GO binary |
| Human GRN | 17,348 | 264,393 | Directed regulatory | Expression PCA-128 |
| BioGRID | 15,289 | 296,797 | Undirected PPI | GO binary |
| GTEx (τ=0.9) | 5,694 | 1,436,482 | Weighted co-expression | Expression PCA-128 |

---

## Citation

If you use ClusterNet in your research, please cite:

```bibtex
@inproceedings{sapna2026clusternet,
  title     = {ClusterNet: Disentangling When and Why Graph Neural Networks 
               Surpass Classical Community Detection on Biological Networks},
  author    = {Sapna, R. and Karthik, Harikeshav and Raman, Karthik},
  booktitle = {Proceedings of the 60th Hawaii International Conference 
               on System Sciences (HICSS)},
  year      = {2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

- Karthik Raman — kraman@iitm.ac.in
- [RamanLab @ IIT Madras](https://github.com/RamanLab)
