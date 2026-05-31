"""
Paths and loaders for Tier~2 (social) benchmark CSVs.

Primary layout: results under ``D:/social/...`` (mirrored cluster output).
"""

from __future__ import annotations

import json
import pathlib
import platform

import pandas as pd

# Graph-tool SBM variants used Louvain fallback in these runs; excluded from reporting.
EXCLUDED_ALGORITHMS_CLASSICAL = frozenset({"sbm", "nested_sbm", "nested_sbm_coarse"})

GNN_ORDER = [
    "dmon_gnn",
    "mincut_gnn",
    "vgae_gnn",
    "dgi_gnn",
    "gcn_supervised",
    "gat_supervised",
    "mlp_kmeans",
    "node2vec_gnn",
]

FEATURE_ORDER = ["real", "structural", "random"]

NETWORK_ORDER = ["cora", "citeseer", "amazon_photo"]


def _p(win_path: str) -> pathlib.Path:
    if platform.system() != "Windows" and len(win_path) >= 3 and win_path[1:3] == ":/":
        drive = win_path[0].lower()
        return pathlib.Path(f"/mnt/{drive}/{win_path[3:]}")
    return pathlib.Path(win_path)


def _candidate_dirs() -> list[pathlib.Path]:
    """Try D:/social mirror first, then direct ClusterNet path."""
    return [
        _p("D:/social/mnt/d/cluster/SocialNetworks/results"),
        _p("D:/cluster/SocialNetworks/results"),
    ]


def results_dir() -> pathlib.Path:
    for d in _candidate_dirs():
        p1 = d / "social_classical_results.csv"
        p2 = d / "social_gnn_results.csv"
        if p1.is_file() and p2.is_file():
            return d
    # Default for fingerprinting / clear errors
    return _candidate_dirs()[0]


def path_classical() -> pathlib.Path:
    return results_dir() / "social_classical_results.csv"


def path_gnn() -> pathlib.Path:
    return results_dir() / "social_gnn_results.csv"


def bundle_root() -> pathlib.Path:
    """Parent of ``results/`` (contains ``data/`` and ``results/``)."""
    return results_dir().parent


def metadata_path(net_id: str) -> pathlib.Path:
    return bundle_root() / "data" / net_id / "metadata.json"


def load_dataset_metadata(net_id: str) -> dict:
    with open(metadata_path(net_id), encoding="utf-8") as f:
        return json.load(f)


def _success_mask(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(("true", "1", "yes"))


def load_classical_filtered() -> pd.DataFrame:
    df = pd.read_csv(path_classical())
    df = df[~df["algorithm"].isin(EXCLUDED_ALGORITHMS_CLASSICAL)].copy()
    df["success"] = _success_mask(df["success"])
    df["ami"] = pd.to_numeric(df["ami"], errors="coerce")
    return df


def load_gnn() -> pd.DataFrame:
    df = pd.read_csv(path_gnn())
    df["success"] = _success_mask(df["success"])
    df["ami"] = pd.to_numeric(df["ami"], errors="coerce")
    df["feature_type"] = df["feature_type"].astype(str)
    return df


def file_fingerprint(path: pathlib.Path) -> dict:
    import hashlib

    p = pathlib.Path(path)
    out: dict = {"path": str(p), "exists": p.is_file()}
    if not out["exists"]:
        return out
    out["size_bytes"] = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    out["sha256"] = h.hexdigest()
    return out
