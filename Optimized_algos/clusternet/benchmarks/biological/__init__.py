"""
Biological Network Benchmarks

Includes:
- Gene Regulatory Networks (GRN)
- Protein-Protein Interaction (PPI)
- Single-Cell RNA-seq derived networks
- Motif analysis utilities
"""

from .motif_analysis import MotifAnalyzer, compute_motif_enrichment

__all__ = ['MotifAnalyzer', 'compute_motif_enrichment']

