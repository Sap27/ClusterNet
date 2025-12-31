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

# Import cdlib wrappers
try:
    from clusternet.algorithms.cdlib_wrappers import *
except ImportError as e:
    print(f"Warning: Could not load cdlib wrappers: {e}")

# Import new classic community detection algorithms
try:
    from clusternet.algorithms.walktrap_wrapper import WalktrapWrapper
except ImportError as e:
    print(f"Warning: Could not load Walktrap: {e}")

try:
    from clusternet.algorithms.spinglass_wrapper import SpinGlassWrapper
except ImportError as e:
    print(f"Warning: Could not load SpinGlass: {e}")

try:
    from clusternet.algorithms.spectral_wrapper import SpectralWrapper
except ImportError as e:
    print(f"Warning: Could not load Spectral Clustering: {e}")

try:
    from clusternet.algorithms.label_propagation_wrapper import LabelPropagationWrapper
except ImportError as e:
    print(f"Warning: Could not load Label Propagation: {e}")

try:
    from clusternet.algorithms.fastgreedy_wrapper import FastGreedyWrapper
except ImportError as e:
    print(f"Warning: Could not load FastGreedy: {e}")

try:
    from clusternet.algorithms.leading_eigen_wrapper import LeadingEigenWrapper
except ImportError as e:
    print(f"Warning: Could not load Leading Eigenvector: {e}")

try:
    from clusternet.algorithms.girvan_newman_wrapper import GirvanNewmanWrapper
except ImportError as e:
    print(f"Warning: Could not load Girvan-Newman: {e}")

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
    # New classic algorithms
    'WalktrapWrapper',
    'SpinGlassWrapper',
    'SpectralWrapper',
    'LabelPropagationWrapper',
    'FastGreedyWrapper',
    'LeadingEigenWrapper',
    'GirvanNewmanWrapper',
]

