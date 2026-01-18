# Null Model Benchmarks for Community Detection

## Overview

This module provides **property-preserving network rewiring** benchmarks to understand algorithm behavior beyond simple accuracy metrics.

## The Central Question

> **"Why do algorithms that perform well on synthetic benchmarks (LFR) sometimes fail on real networks?"**

**Answer**: Different algorithms rely on different network properties. Null models reveal these hidden dependencies.

---

## Two Complementary Benchmarks

### 1. WEAKEN BENCHMARK (`weaken_benchmark.py`)

**Purpose**: Progressively destroy community structure while preserving network properties.

```
Original Network (with communities)
         │
         ▼
    ┌─────────┐
    │   1k    │  Preserve: degree distribution
    │  weaken │  Destroy: community edges → random edges
    └────┬────┘
         │
    ┌─────────┐
    │   2k    │  Preserve: + degree correlations
    │  weaken │  Harder to weaken (more constraints)
    └────┬────┘
         │
    ┌─────────┐
    │   3k    │  Preserve: + clustering coefficient
    │  weaken │  Hardest to weaken
    └────┬────┘
         ▼
    Randomized Network
```

**Key Outputs**:
- **Critical Threshold**: Weakening level where AMI < 0.5
- **Property Fingerprint**: How much each algorithm needs each property

**Interpretation**:
```
Algorithm A: threshold_1k=0.3, threshold_3k=0.6
→ Algorithm A NEEDS clustering coefficient (3k) to work well

Algorithm B: threshold_1k=0.5, threshold_3k=0.5  
→ Algorithm B is robust to different properties
```

### 2. MESOSCALE BENCHMARK (`mesoscale_benchmark.py`)

**Purpose**: Shuffle edges while PRESERVING community structure.

```
Original Network (with communities)
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│ inner │ │ inter │
│shuffle│ │shuffle│  
└───┬───┘ └───┬───┘
    │         │
    ▼         ▼
  Same      Same
community  community
structure  structure
but edges  but edges
 moved      moved
```

**Key Outputs**:
- **Stability Score**: Performance after shuffling vs. before
- **Mesoscale vs Edge-Specific**: Does algorithm detect communities or memorize edges?

**Interpretation**:
```
Algorithm with high stability (>0.8) → Truly detects mesoscale structure
Algorithm with low stability (<0.5) → Relies on specific edge patterns
```

---

## Research Applications

### 1. Algorithm Property Fingerprinting

```python
from clusternet.benchmarks.null_model import run_weaken_benchmark

# Run benchmark
results = run_weaken_benchmark(G, communities, algorithms,
    preserve_orders=['1k', '2k', '3k'],
    weaken_levels=[0.0, 0.2, 0.4, 0.6, 0.8]
)

# Extract fingerprints
for algo in algorithms:
    fingerprint = {
        'needs_degree': threshold_1k[algo] > 0.3,
        'needs_correlation': threshold_2k[algo] - threshold_1k[algo] > 0.1,
        'needs_clustering': threshold_3k[algo] - threshold_2k[algo] > 0.1
    }
```

### 2. Understanding Algorithm Failures

```
Real Network: High clustering, low assortativity

Algorithm rankings on LFR (synthetic):
  1. Louvain (AMI=0.95)
  2. Infomap (AMI=0.92)
  3. Label Prop (AMI=0.88)

But on real network:
  1. Infomap (AMI=0.78)  ← Better because it needs correlation, not clustering
  2. Louvain (AMI=0.65)  ← Worse because clustering is different
  3. Label Prop (AMI=0.70)

WHY? Null model reveals:
  - Louvain has high clustering dependence
  - Real network clustering differs from LFR
  - Infomap is more robust to clustering variations
```

### 3. Algorithm Selection Guidelines

| Network Property | Best Algorithm Type |
|------------------|---------------------|
| High clustering | Modularity-based (Louvain, Leiden) |
| Flow-like structure | Flow-based (Infomap) |
| High degree heterogeneity | Label Propagation, Spectral |
| Unknown properties | GNN (learned), Ensemble |

---

## Usage

### Quick Start

```python
import networkx as nx
from clusternet.benchmarks.null_model import (
    run_weaken_benchmark,
    run_mesoscale_benchmark,
    analyze_weaken_results
)

# Load network
G = nx.karate_club_graph()
communities = [[0,1,2,...], [8,9,10,...]]  # Ground truth

# Define algorithms
algorithms = {
    'Louvain': louvain_func,
    'Infomap': infomap_func,
}

# Run benchmarks
weaken_results = run_weaken_benchmark(G, communities, algorithms)
mesoscale_results = run_mesoscale_benchmark(G, communities, algorithms)

# Analyze
analysis = analyze_weaken_results(weaken_results)
print(analysis['critical_thresholds'])
print(analysis['property_fingerprints'])
```

### Full Research Pipeline

```bash
cd ClusterNet/Optimized_algos/benchmarks/null_model
python research_integration.py --network karate --output results/
python research_integration.py --network cora --cora_path ../../../../CoRA_Raw
```

---

## Connection to Multi-Scale Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTI-SCALE ROBUSTNESS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  LFR/ABCD              NULL MODEL              BIOLOGICAL           │
│  ────────              ──────────              ──────────           │
│     │                      │                       │                │
│     ▼                      ▼                       ▼                │
│  "Can it               "What does              "Does it find        │
│   detect?"             it need?"               meaningful           │
│                                                communities?"        │
│     │                      │                       │                │
│     └──────────────────────┼───────────────────────┘                │
│                            │                                        │
│                            ▼                                        │
│              MULTI-SCALE ROBUSTNESS SCORE                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `weaken_benchmark.py` | Progressive community destruction |
| `mesoscale_benchmark.py` | Edge shuffling with structure preservation |
| `research_integration.py` | Full pipeline with Cora integration |
| `__init__.py` | Module exports |

---

## References

Based on:
- Community-nullmodel project
- Orsini et al. (2015). "Quantifying randomness in real networks"
- Fosdick et al. (2018). "Configuring random graph models with fixed degree sequences"






