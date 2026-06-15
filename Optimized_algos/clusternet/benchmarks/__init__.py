"""
ClusterNet Benchmarks Module

Provides standardized benchmarks for evaluating community detection:
1. Synthetic: LFR, SBM, Hierarchical
2. Rewired Real Networks (Simple + Null Model)
3. Biological Networks with Motif Analysis
"""

from .synthetic.lfr_generator import LFRBenchmark
from .synthetic.sbm_generator import SBMBenchmark, HierarchicalSBM
from .rewired.rewiring import RewiredBenchmark
from .rewired.null_model_rewiring import (
    weaken_community_structure,
    strengthen_community_structure,
    generate_null_model_sweep,
    compute_mixing_parameter,
    Q_decrease_1k,
    Q_decrease_2k,
    Q_decrease_3k,
)

__all__ = [
    # Synthetic
    'LFRBenchmark',
    'SBMBenchmark',
    'HierarchicalSBM',
    # Rewired
    'RewiredBenchmark',
    'weaken_community_structure',
    'strengthen_community_structure',
    'generate_null_model_sweep',
    'compute_mixing_parameter',
    'Q_decrease_1k',
    'Q_decrease_2k',
    'Q_decrease_3k',
]

