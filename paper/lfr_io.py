"""
Single source of truth for LFR benchmark CSV paths and merged loaders.

GPU folders (Vast / remote runs): accuracy, scalability, hierarchical classical+GNN.
Local repo (this machine): fills classical algorithms missing on GPU (graph-tool / extras).
"""

from __future__ import annotations

import pathlib
import platform

import numpy as np
import pandas as pd


def _p(win_path: str) -> pathlib.Path:
    if platform.system() != "Windows" and len(win_path) >= 3 and win_path[1:3] == ":/":
        drive = win_path[0].lower()
        return pathlib.Path(f"/mnt/{drive}/{win_path[3:]}")
    return pathlib.Path(win_path)


# --- Explicit paths (user-indicated result folders) ---
PATHS = {
    "gpu_accuracy_classical": _p(
        "D:/results-clusternet-classical-gpu-new/results/classical_accuracy_results.csv"
    ),
    "gpu_accuracy_gnn": _p(
        "D:/results-clusternet-classical-gpu-3/results/gnn_accuracy_results.csv"
    ),
    "gpu_scalability_classical": _p(
        "D:/results-clusternet-classical-sca-gpu-new4/results/classical_accuracy_results.csv"
    ),
    "gpu_scalability_gnn": _p(
        "D:/results-clusternet-classical-sca-gpu-3/results-clusternet-classical-sca-gpu-3/results/gnn_scalability_results.csv"
    ),
    "gpu_hierarchical_classical": _p(
        "D:/results-clusternet-classical-hier-gpu-new/results/hierarchical_results.csv"
    ),
    "gpu_hierarchical_gnn": _p(
        "D:/results-clusternet-classical-hierarchical-gnn-gpu/results-clusternet-classical-hier-gpu/results/gnn_hierarchical_results.csv"
    ),
    "local_hierarchical_classical": _p(
        "D:/cluster/ClusterNet/Optimized_algos/benchmarks/lfr/hierarchical/results/hierarchical_results.csv"
    ),
    "local_accuracy_classical": _p(
        "D:/cluster/ClusterNet/Optimized_algos/benchmarks/lfr/accuracy/results/accuracy_results.csv"
    ),
    "local_scalability_classical": _p(
        "D:/cluster/ClusterNet/Optimized_algos/benchmarks/lfr/scalability/results/scalability_results.csv"
    ),
}

GNN_BASE_ORDER = [
    "dmon_gnn",
    "mincut_gnn",
    "vgae_gnn",
    "dgi_gnn",
    "gcn_supervised",
    "gat_supervised",
    "mlp_kmeans",
    "node2vec_gnn",
]

# Use local (this machine) AMI for these classical methods where local has matching μ or N;
# keep GPU CSV rows for runtime, fine-μ grid coverage, and all other columns.
AMI_OVERLAY_FROM_LOCAL = frozenset({"sbm", "nested_sbm"})

# Hierarchical benchmark has no plain ``sbm`` row (only nested variants on GPU/local).
AMI_OVERLAY_HIERARCHICAL_CLASSICAL = frozenset({"nested_sbm", "nested_sbm_coarse"})


def _overlay_local_ami_by_mu(
    merged: pd.DataFrame, local: pd.DataFrame, algorithms: frozenset, *, atol: float = 0.005
) -> pd.DataFrame:
    out = merged.copy()
    for algo in algorithms:
        sel_a = out["algorithm"] == algo
        if not sel_a.any():
            continue
        for mu_g in out.loc[sel_a, "mu"].unique():
            loc_sub = local[
                (local["algorithm"] == algo) & (np.isclose(local["mu"], float(mu_g), atol=atol))
            ]
            if len(loc_sub) == 0:
                continue
            ami_mean = float(loc_sub["ami"].mean())
            row_mask = sel_a & np.isclose(out["mu"], float(mu_g), atol=atol)
            out.loc[row_mask, "ami"] = ami_mean
    return out


def _overlay_local_ami_by_n(
    merged: pd.DataFrame, local: pd.DataFrame, algorithms: frozenset
) -> pd.DataFrame:
    out = merged.copy()
    for algo in algorithms:
        sel_a = out["algorithm"] == algo
        if not sel_a.any():
            continue
        for n_val in out.loc[sel_a, "N"].unique():
            n_int = int(n_val)
            loc_sub = local[(local["algorithm"] == algo) & (local["N"] == n_int)]
            if len(loc_sub) == 0:
                continue
            ami_mean = float(loc_sub["ami"].mean())
            row_mask = sel_a & (out["N"] == n_int)
            out.loc[row_mask, "ami"] = ami_mean
    return out


def load_gpu_accuracy_classical() -> pd.DataFrame:
    df = pd.read_csv(PATHS["gpu_accuracy_classical"])
    df["mu"] = df["mu"].astype(float)
    return df


def load_local_accuracy_classical() -> pd.DataFrame:
    df = pd.read_csv(PATHS["local_accuracy_classical"])
    df["mu"] = df["mu"].astype(float)
    df = df[~df["algorithm"].isin(["dmon", "mincut"])]
    return df


def load_merged_accuracy_classical() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (merged_df, provenance_df one row per algorithm)."""
    gpu = load_gpu_accuracy_classical()
    local = load_local_accuracy_classical()
    gpu_algos = set(gpu["algorithm"].unique())
    missing = set(local["algorithm"].unique()) - gpu_algos
    supplement = local[local["algorithm"].isin(missing)]
    merged = pd.concat([gpu, supplement], ignore_index=True)
    merged = _overlay_local_ami_by_mu(merged, local, AMI_OVERLAY_FROM_LOCAL)

    prov_rows = []
    for a in sorted(merged["algorithm"].unique()):
        if a not in gpu_algos:
            src = "local_accuracy_supplement (full rows)"
        elif a in AMI_OVERLAY_FROM_LOCAL:
            src = (
                "gpu_accuracy_classical rows; AMI replaced from local_accuracy_classical "
                "when same mu is present else GPU AMI; runtime from GPU"
            )
        else:
            src = "gpu_accuracy_classical"
        prov_rows.append({"algorithm": a, "accuracy_source": src})
    provenance = pd.DataFrame(prov_rows)
    return merged, provenance


def load_merged_accuracy_gnn() -> pd.DataFrame:
    df = pd.read_csv(PATHS["gpu_accuracy_gnn"])
    df["mu"] = df["mu"].astype(float)
    return df[df["algorithm"].isin(GNN_BASE_ORDER)]


def _overlay_hierarchical_ami_from_local(
    gpu: pd.DataFrame, local: pd.DataFrame, algorithms: frozenset
) -> pd.DataFrame:
    """Replace ami_micro / ami_macro from local CSV for matching (mu1, mu2, realization); keep GPU runtime."""
    out = gpu.copy()
    for algo in algorithms:
        sel = out["algorithm"] == algo
        if not sel.any():
            continue
        for i in out.index[sel]:
            m1 = float(out.at[i, "mu1"])
            m2 = float(out.at[i, "mu2"])
            r = int(out.at[i, "realization"])
            loc = local[
                (local["algorithm"] == algo)
                & (np.isclose(local["mu1"], m1))
                & (np.isclose(local["mu2"], m2))
                & (local["realization"].astype(int) == r)
            ]
            if len(loc) == 0:
                continue
            out.at[i, "ami_micro"] = float(loc["ami_micro"].mean())
            out.at[i, "ami_macro"] = float(loc["ami_macro"].mean())
    return out


def load_gpu_hierarchical_classical() -> pd.DataFrame:
    df = pd.read_csv(PATHS["gpu_hierarchical_classical"])
    df["mu1"] = df["mu1"].astype(float)
    df["mu2"] = df["mu2"].astype(float)
    df["realization"] = df["realization"].astype(int)
    return df


def load_local_hierarchical_classical() -> pd.DataFrame:
    df = pd.read_csv(PATHS["local_hierarchical_classical"])
    df["mu1"] = df["mu1"].astype(float)
    df["mu2"] = df["mu2"].astype(float)
    df["realization"] = df["realization"].astype(int)
    return df


def load_merged_hierarchical_classical() -> tuple[pd.DataFrame, pd.DataFrame]:
    """GPU hierarchical rows; Nested SBM AMI from local graph-tool fits when keys align; runtime from GPU."""
    gpu = load_gpu_hierarchical_classical()
    local = load_local_hierarchical_classical()
    merged = _overlay_hierarchical_ami_from_local(gpu, local, AMI_OVERLAY_HIERARCHICAL_CLASSICAL)
    prov_rows = []
    for a in sorted(merged["algorithm"].unique()):
        if a in AMI_OVERLAY_HIERARCHICAL_CLASSICAL:
            src = (
                "gpu_hierarchical_classical; ami_micro/ami_macro from local_hierarchical_classical "
                "when same (mu1, mu2, realization); runtime and row grid from GPU"
            )
        else:
            src = "gpu_hierarchical_classical"
        prov_rows.append({"algorithm": a, "hierarchical_source": src})
    return merged, pd.DataFrame(prov_rows)


def load_gpu_hierarchical_gnn() -> pd.DataFrame:
    df = pd.read_csv(PATHS["gpu_hierarchical_gnn"])
    df["mu1"] = df["mu1"].astype(float)
    df["mu2"] = df["mu2"].astype(float)
    df["realization"] = df["realization"].astype(int)
    return df


def load_gpu_accuracy_gnn_full() -> pd.DataFrame:
    """All GNN accuracy rows (label-fraction variants: gat_supervised_40, etc.)."""
    df = pd.read_csv(PATHS["gpu_accuracy_gnn"])
    df["mu"] = df["mu"].astype(float)
    return df


def load_merged_scalability_classical() -> tuple[pd.DataFrame, pd.DataFrame]:
    gpu = pd.read_csv(PATHS["gpu_scalability_classical"])
    gpu = gpu.rename(columns={"size": "N"})
    gpu["N"] = gpu["N"].astype(float).astype(int)

    local = pd.read_csv(PATHS["local_scalability_classical"])
    local["N"] = local["N"].astype(float).astype(int)
    local = local[~local["algorithm"].isin(["dmon", "mincut"])]

    gpu_algos = set(gpu["algorithm"].unique())
    missing = set(local["algorithm"].unique()) - gpu_algos
    supplement = local[local["algorithm"].isin(missing)]
    merged = pd.concat([gpu, supplement], ignore_index=True)
    merged = _overlay_local_ami_by_n(merged, local, AMI_OVERLAY_FROM_LOCAL)

    prov_rows = []
    for a in sorted(merged["algorithm"].unique()):
        if a not in gpu_algos:
            src = "local_scalability_supplement (full rows)"
        elif a in AMI_OVERLAY_FROM_LOCAL:
            src = (
                "gpu_scalability_classical rows; AMI replaced from local_scalability_classical "
                "when same N is present else GPU AMI; runtime from GPU"
            )
        else:
            src = "gpu_scalability_classical"
        prov_rows.append({"algorithm": a, "scalability_source": src})
    provenance = pd.DataFrame(prov_rows)
    return merged, provenance


def load_merged_scalability_gnn() -> pd.DataFrame:
    df = pd.read_csv(PATHS["gpu_scalability_gnn"])
    df = df.rename(columns={"size": "N"})
    df["N"] = df["N"].astype(float).astype(int)
    return df[df["algorithm"].isin(GNN_BASE_ORDER)]


def ami_mean(cl_or_gn: pd.DataFrame, algo: str, mu: float, atol: float = 0.005) -> float:
    sub = cl_or_gn[(cl_or_gn["algorithm"] == algo) & (np.isclose(cl_or_gn["mu"], mu, atol=atol))]
    return float(sub["ami"].mean()) if len(sub) else float("nan")


def ami_mean_scalability(
    df: pd.DataFrame, algo: str, n: int, success_only: bool = True
) -> tuple[float, float]:
    sub = df[(df["algorithm"] == algo) & (df["N"] == n)]
    if success_only and "success" in sub.columns:
        sub = sub[sub["success"] == True]
    if len(sub) == 0:
        return float("nan"), float("nan")
    return float(sub["ami"].mean()), float(sub["runtime"].mean())


def file_fingerprint(path: pathlib.Path) -> dict:
    import hashlib

    p = pathlib.Path(path)
    out = {"path": str(p), "exists": p.is_file()}
    if not out["exists"]:
        return out
    out["size_bytes"] = p.stat().st_size
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    out["sha256"] = h.hexdigest()
    return out
