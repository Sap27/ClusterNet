"""
Rewired Real Network Benchmarks

Two approaches:
1. RewiredBenchmark: Simple rewiring to impose community structure
2. Null Model Rewiring: Sophisticated rewiring preserving network properties
"""

from .rewiring import RewiredBenchmark
from .null_model_rewiring import (
    # Community structure manipulation
    Q_decrease_1k,
    Q_decrease_2k,
    Q_decrease_3k,
    Q_increase_1k,
    weaken_community_structure,
    strengthen_community_structure,
    # Edge shuffling
    inner_random_1k,
    inter_random_1k,
    # High-level functions
    generate_null_model_sweep,
    compute_mixing_parameter,
)

__all__ = [
    # Simple rewiring
    'RewiredBenchmark',
    # Null model functions
    'Q_decrease_1k',
    'Q_decrease_2k', 
    'Q_decrease_3k',
    'Q_increase_1k',
    'weaken_community_structure',
    'strengthen_community_structure',
    'inner_random_1k',
    'inter_random_1k',
    'generate_null_model_sweep',
    'compute_mixing_parameter',
]

