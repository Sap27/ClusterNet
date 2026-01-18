"""
LFR Benchmark Suite for Community Detection
============================================

This package provides a comprehensive benchmarking framework for community detection
algorithms using LFR (Lancichinetti-Fortunato-Radicchi) synthetic networks.

Benchmarks:
-----------
1. **Accuracy**: AMI vs mixing parameter (μ)
2. **Scalability**: Runtime vs network size (N)
3. **Hierarchical**: Detection of nested communities
4. **Overlapping**: Detection of overlapping communities

Usage:
------
    # Quick test
    python run_all_benchmarks.py --minimal

    # Full benchmark
    python run_all_benchmarks.py --full

    # Individual benchmarks
    python run_accuracy_benchmark.py --all
    python run_scalability_benchmark.py --all
    python run_hierarchical_benchmark.py --all
    python run_overlapping_benchmark.py --all

Modules:
--------
- config: Benchmark configuration
- lfr_generator: LFR network generation
- algorithm_runner: Unified algorithm interface
- metrics: Evaluation metrics (AMI, NMI, ONMI, etc.)
"""

__version__ = '1.0.0'




