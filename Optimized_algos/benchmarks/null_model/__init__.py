"""
Null Model Benchmarks for ClusterNet

Two complementary benchmarks for understanding algorithm behavior:

1. MESOSCALE BENCHMARK (mesoscale_benchmark.py)
   - Shuffles edges while PRESERVING community structure
   - Tests if algorithms detect mesoscale patterns vs edge-specific patterns
   - Key insight: Algorithms that work after shuffling are truly detecting communities

2. WEAKEN BENCHMARK (weaken_benchmark.py)  
   - Progressively DESTROYS community structure
   - Preserves network properties at different orders (1k, 2k, 3k)
   - Key insight: Reveals which properties each algorithm depends on

Property Preservation Orders:
    0k: Preserves edge count only
    1k: Preserves degree distribution
    2k: Preserves degree correlation (assortativity)
    2.5k: Preserves degree-dependent clustering
    3k: Preserves clustering coefficient

Based on: Community-nullmodel project
"""

from .mesoscale_benchmark import (
    run_mesoscale_benchmark,
    analyze_mesoscale_results,
    inner_random_1k,
    inter_random_1k,
    edge_in_community,
    compute_mixing_parameter
)

from .weaken_benchmark import (
    run_weaken_benchmark,
    analyze_weaken_results,
    plot_weaken_results,
    Q_decrease_1k,
    Q_decrease_2k,
    Q_decrease_3k
)

__all__ = [
    # Mesoscale
    'run_mesoscale_benchmark',
    'analyze_mesoscale_results',
    'inner_random_1k',
    'inter_random_1k',
    
    # Weaken
    'run_weaken_benchmark',
    'analyze_weaken_results',
    'plot_weaken_results',
    'Q_decrease_1k',
    'Q_decrease_2k',
    'Q_decrease_3k',
    
    # Shared
    'edge_in_community',
    'compute_mixing_parameter'
]






