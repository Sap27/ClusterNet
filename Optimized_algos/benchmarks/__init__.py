"""
ClusterNet Benchmarks

Comprehensive benchmarking framework for community detection algorithms.

Three Scales of Evaluation:
    1. SYNTHETIC (LFR/ABCD)
       - Standard accuracy tests
       - Scalability analysis
       - Attack resilience
    
    2. STRUCTURAL (Null Model)
       - Property dependence analysis
       - Mesoscale preservation tests
       - Community weakening studies
    
    3. BIOLOGICAL (Real Networks)
       - Motif enrichment
       - Functional validation
       - Perturbation studies
"""

from . import null_model

__all__ = ['null_model']






