#!/bin/bash
# =============================================================================
# Biological Network HICSS-60 Experiments
# Run on RTX 5090 GPU machine (Vast.ai)
#
# Experiments:
#   1. Feature ablation (real vs structural vs random) — highest priority
#   2. Embedding export (DGI, GAT, DMoN, VGAE) — for t-SNE visualization
#   3. Attention export (GAT) — for biological edge interpretation
#   4. Complete perturbation coverage (add missing GNN methods)
#
# Estimated runtime: ~15-19 hours total on RTX 5090
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo " HICSS-60 Biological Experiments"
echo " $(date)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "========================================"

# --- Experiment 1: Feature Ablation (highest priority, ~10-14h) ---
echo ""
echo "=== Experiment 1: Feature Ablation ==="
echo "    8 GNNs x 4 networks x 3 feature types x 3 seeds = 288 runs"
echo ""
python run_benchmark.py \
    --run-gnn \
    --feature-ablation \
    --epochs 600 \
    --seeds 3 \
    --resume

echo ""
echo "=== Experiment 1 COMPLETE ==="
echo ""

# --- Experiment 2: Embedding Export (~1h) ---
echo ""
echo "=== Experiment 2: Embedding Export ==="
echo "    DGI, DMoN, VGAE, GAT on 4 networks x 1 seed"
echo ""
python run_benchmark.py \
    --save-embeddings \
    --epochs 600 \
    --seeds 1

echo ""
echo "=== Experiment 2 COMPLETE ==="
echo ""

# --- Experiment 3: Attention Export (~36min) ---
echo ""
echo "=== Experiment 3: GAT Attention Export ==="
echo "    GAT on 4 networks x 3 seeds"
echo ""
python run_benchmark.py \
    --save-attention \
    --epochs 600 \
    --seeds 3

echo ""
echo "=== Experiment 3 COMPLETE ==="
echo ""

# --- Experiment 4: Complete Perturbation (~3h) ---
echo ""
echo "=== Experiment 4: Perturbation (all GNN methods) ==="
echo "    Adding GAT, VGAE, Node2Vec to perturbation study"
echo ""
python run_perturbation.py --resume

echo ""
echo "=== Experiment 4 COMPLETE ==="
echo ""

echo "========================================"
echo " ALL EXPERIMENTS COMPLETE"
echo " $(date)"
echo "========================================"
echo ""
echo "Results:"
echo "  Feature ablation:  results/bio_gnn_ablation_results.csv"
echo "  Embeddings:        results/embeddings/*.npy"
echo "  Attention:         results/attention/*.json"
echo "  Perturbation:      results/perturbation/grn_perturbation_results.csv"
