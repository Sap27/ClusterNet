"""
Algorithm registry for ClusterNet.

This module maintains a registry of all available community detection algorithms.
"""

from typing import Dict, Type, Optional, List
import warnings

# Registry will be populated by algorithm imports
ALGORITHM_REGISTRY: Dict[str, Type] = {}


def register_algorithm(name: str, aliases: Optional[List[str]] = None):
    """
    Decorator to register a community detection algorithm.
    
    Args:
        name: Primary name for the algorithm
        aliases: List of alternative names
        
    Example:
        @register_algorithm('louvain', aliases=['louvain_method'])
        class LouvainAlgorithm(BaseAlgorithm):
            pass
    """
    def decorator(cls):
        # Register primary name
        ALGORITHM_REGISTRY[name.lower()] = cls
        
        # Register aliases
        if aliases:
            for alias in aliases:
                ALGORITHM_REGISTRY[alias.lower()] = cls
        
        # Store metadata on class
        cls.ALGORITHM_NAME = name
        cls.ALGORITHM_ALIASES = aliases or []
        
        return cls
    
    return decorator


def get_algorithm_class(name: str) -> Optional[Type]:
    """
    Get algorithm class by name.
    
    Args:
        name: Algorithm name (case-insensitive)
        
    Returns:
        Algorithm class or None if not found
    """
    return ALGORITHM_REGISTRY.get(name.lower())


def list_algorithms(verbose: bool = False) -> List[str]:
    """
    List all available algorithms.
    
    Args:
        verbose: If True, show detailed information
        
    Returns:
        List of algorithm names
    """
    # Get unique algorithms (remove aliases)
    unique_algos = {}
    for name, cls in ALGORITHM_REGISTRY.items():
        primary_name = getattr(cls, 'ALGORITHM_NAME', name)
        if primary_name not in unique_algos:
            unique_algos[primary_name] = cls
    
    if verbose:
        print(f"\nAvailable Community Detection Algorithms ({len(unique_algos)}):")
        print("=" * 70)
        
        for name, cls in sorted(unique_algos.items()):
            aliases = getattr(cls, 'ALGORITHM_ALIASES', [])
            directed = getattr(cls, 'SUPPORTS_DIRECTED', '?')
            weighted = getattr(cls, 'SUPPORTS_WEIGHTED', '?')
            
            print(f"\n{name}")
            print(f"  Class: {cls.__name__}")
            if aliases:
                print(f"  Aliases: {', '.join(aliases)}")
            print(f"  Directed: {'✓' if directed else '✗'}")
            print(f"  Weighted: {'✓' if weighted else '✗'}")
            
            # Get docstring first line
            if cls.__doc__:
                first_line = cls.__doc__.strip().split('\n')[0]
                print(f"  Description: {first_line}")
        
        print("\n" + "=" * 70)
    
    return sorted(unique_algos.keys())


def algorithm_exists(name: str) -> bool:
    """Check if an algorithm is registered."""
    return name.lower() in ALGORITHM_REGISTRY


# Algorithm metadata
ALGORITHM_CATEGORIES = {
    'modularity': ['louvain', 'leiden'],
    'ensemble': ['csbio_iitm2', 'ensemble_louvain'],
    'spectral': ['score', 'svt'],
    'hierarchical': ['simnet', 'teamcs'],
    'infomap': ['infomap', 'tusk'],
    'hybrid': ['bigs2'],
}


def get_algorithms_by_category(category: str) -> List[str]:
    """
    Get algorithms in a specific category.
    
    Args:
        category: Category name ('modularity', 'spectral', etc.)
        
    Returns:
        List of algorithm names in that category
    """
    return ALGORITHM_CATEGORIES.get(category.lower(), [])


def get_algorithm_category(name: str) -> Optional[str]:
    """Get the category of an algorithm."""
    name = name.lower()
    for category, algos in ALGORITHM_CATEGORIES.items():
        if name in algos:
            return category
    return None

