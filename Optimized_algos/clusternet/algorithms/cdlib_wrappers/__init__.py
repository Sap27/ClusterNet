"""
CDlib algorithm wrappers for ClusterNet.

This module provides wrappers for community detection algorithms from the cdlib library.
"""

from .statistical_wrappers import *
from .physics_wrappers import *
from .diffusion_wrappers import *
from .structural_wrappers import *
from .overlapping_wrappers import *

__all__ = [
    # Statistical
    'EMWrapper',
    'SBMWrapper',
    'NestedSBMWrapper',
    
    # Physics-based
    'CPMWrapper',
    'RBPotsWrapper',
    'RBERPotsWrapper',
    
    # Diffusion-based
    'DERWrapper',
    'AsyncFluidWrapper',
    
    # Structural
    'SCANWrapper',
    'AGDLWrapper',
    'GDMP2Wrapper',
    
    # Overlapping
    'AngelWrapper',
    'SurpriseCommunitiesWrapper',
]



