"""
I/O helpers for Tier-3 (biological) benchmark data.
All paths and loaders in one place.
"""
from __future__ import annotations

import hashlib
import pathlib
from typing import Any

import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================
import platform as _platform

def _p(win: str) -> pathlib.Path:
    if _platform.system() == "Windows":
        return pathlib.Path(win)
    return pathlib.Path(win.replace("D:/", "/mnt/d/").replace("D:\\", "/mnt/d/"))

PATHS = {
    "enrichment": _p("D:/cluster/BIologicalNetworks/results/bio_enrichment_results.csv"),
    "classical": _p("D:/results-biological-networks-classical2/results-biological-networks-classical/results/bio_classical_results.csv"),
    "gnn": _p("D:/results-perturb-gnn-biological/results/bio_gnn_results.csv"),
    "perturbation": _p("D:/results-perturb-gnn-biological/results/perturbation/grn_perturbation_results.csv"),
}

NETWORK_ORDER = ["signor", "grn", "biogrid", "gtex_0.9"]
NETWORK_DISPLAY = {
    "signor": "SIGNOR",
    "grn": "Human GRN",
    "biogrid": "BioGRID",
    "gtex_0.9": "GTEx",
}

GNN_ALGOS = [
    "dgi_gnn", "dmon_gnn", "gat_supervised", "gcn_supervised",
    "mincut_gnn", "mlp_kmeans", "node2vec_gnn", "vgae_gnn",
]

ALGO_DISPLAY = {
    "leiden": "Leiden", "louvain": "Louvain", "fastgreedy": "FastGreedy",
    "label_propagation": "Label Prop.", "infomap": "Infomap",
    "spectral": "Spectral", "spinglass": "Spinglass", "rb_pots": "RB-Pots",
    "rber_pots": "RBer-Pots", "cpm": "CPM", "surprise": "Surprise",
    "nested_sbm": "Nested SBM", "nested_sbm_coarse": "Nested SBM (coarse)",
    "sbm": "SBM", "agdl": "AGDL", "scan": "SCAN", "walktrap": "Walktrap",
    "demon": "DEMON", "angel": "ANGEL", "teamcs": "TeamCS",
    "csbio_iitm2": "CSBIO-IITM2", "bigs2": "BiGS2", "score": "SCORE",
    "dmon_gnn": "DMoN", "mincut_gnn": "MinCutPool", "vgae_gnn": "VGAE+KM",
    "dgi_gnn": "DGI+KM", "node2vec_gnn": "Node2Vec+KM",
    "mlp_kmeans": "MLP+KM", "gcn_supervised": "GCN (sup.)",
    "gat_supervised": "GAT (sup.)", "leading_eigen": "Leading Eigen",
    "em": "EM", "der": "DER", "svt": "SVT", "kclique": "K-Clique",
    "async_fluid": "Async Fluid",
}


def file_fingerprint(path: pathlib.Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    data = path.read_bytes()
    return {
        "exists": True,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


# =============================================================================
# LOADERS
# =============================================================================

def load_enrichment() -> pd.DataFrame:
    df = pd.read_csv(PATHS["enrichment"])
    df["type"] = df["algorithm"].apply(lambda x: "GNN" if x in GNN_ALGOS else "Classical")
    return df


def load_enrichment_averaged() -> pd.DataFrame:
    """Average over seeds for GNN; classical has seed=0 only."""
    df = load_enrichment()
    return df.groupby(["network", "algorithm", "type"]).agg({
        "num_communities": "mean",
        "go_enriched_frac": "mean",
        "kegg_enriched_frac": "mean",
        "communities_tested": "mean",
        "mean_go_per_community": "mean",
        "sign_coherence": "mean",
    }).reset_index()


def load_classical() -> pd.DataFrame:
    return pd.read_csv(PATHS["classical"])


def load_gnn() -> pd.DataFrame:
    return pd.read_csv(PATHS["gnn"])


def load_perturbation() -> pd.DataFrame:
    return pd.read_csv(PATHS["perturbation"])


def perturbation_summary() -> pd.DataFrame:
    """Mean NMI, AMI, FFL change per (test, algorithm) across seeds."""
    df = load_perturbation()
    return df.groupby(["test", "level", "algorithm"]).agg({
        "nmi": "mean",
        "ami": "mean",
        "ffl_preserved_orig": "mean",
        "ffl_preserved_pert": "mean",
        "ffl_change": "mean",
        "modularity_change": "mean",
    }).reset_index()
