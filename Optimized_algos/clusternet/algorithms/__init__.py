"""
Community detection algorithms for ClusterNet.

This package contains implementations of various community detection algorithms,
all with a unified interface.
"""

from clusternet.base import BaseAlgorithm
from clusternet.registry import register_algorithm

# Import all algorithm implementations
# These imports will trigger the @register_algorithm decorators

try:
    from clusternet.algorithms.louvain_wrapper import LouvainWrapper
except ImportError as e:
    print(f"Warning: Could not load Louvain: {e}")

try:
    from clusternet.algorithms.leiden_wrapper import LeidenWrapper
except ImportError as e:
    print(f"Warning: Could not load Leiden: {e}")

try:
    from clusternet.algorithms.svt_wrapper import SVTWrapper
except ImportError as e:
    print(f"Warning: Could not load SVT: {e}")

try:
    from clusternet.algorithms.score_wrapper import SCOREWrapper
except ImportError as e:
    print(f"Warning: Could not load SCORE: {e}")

try:
    from clusternet.algorithms.teamcs_wrapper import TeamCSWrapper
except ImportError as e:
    print(f"Warning: Could not load TeamCS: {e}")

try:
    from clusternet.algorithms.simnet_wrapper import SimNetWrapper
except ImportError as e:
    print(f"Warning: Could not load SimNet: {e}")

try:
    from clusternet.algorithms.tusk_wrapper import TuskWrapper
except ImportError as e:
    print(f"Warning: Could not load Tusk: {e}")

try:
    from clusternet.algorithms.bigs2_wrapper import BiGS2Wrapper
except ImportError as e:
    print(f"Warning: Could not load BiGS2: {e}")

try:
    from clusternet.algorithms.csbio_iitm2_wrapper import CSBIOWrapper
except ImportError as e:
    print(f"Warning: Could not load CSBIO-IITM2: {e}")

__all__ = [
    'BaseAlgorithm',
    'LouvainWrapper',
    'LeidenWrapper',
    'SVTWrapper',
    'SCOREWrapper',
    'TeamCSWrapper',
    'SimNetWrapper',
    'TuskWrapper',
    'BiGS2Wrapper',
    'CSBIOWrapper',
]

