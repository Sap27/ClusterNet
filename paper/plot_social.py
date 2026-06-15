"""
Figures for Tier~2 (social) benchmarks from social_io CSV paths.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from social_io import (
    FEATURE_ORDER,
    GNN_ORDER,
    NETWORK_ORDER,
    load_classical_filtered,
    load_gnn,
)

OUT = pathlib.Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

DISPLAY = {
    "dmon_gnn": "DMoN",
    "mincut_gnn": "MinCut",
    "vgae_gnn": "VGAE+KM",
    "dgi_gnn": "DGI+KM",
    "gcn_supervised": "GCN",
    "gat_supervised": "GAT",
    "mlp_kmeans": "MLP+KM",
    "node2vec_gnn": "N2V+KM",
}


def _ami_rankings():
    cl = load_classical_filtered()
    gn = load_gnn()
    cl_ok = cl[cl["success"] & cl["ami"].notna()]
    gn_real = gn[gn["success"] & gn["ami"].notna() & (gn["feature_type"] == "real")]
    out = {}
    for net in NETWORK_ORDER:
        rows = []
        for algo, ser in cl_ok[cl_ok["network"] == net].groupby("algorithm")["ami"].mean().items():
            rows.append((algo, float(ser), "class."))
        for algo, ser in gn_real[gn_real["network"] == net].groupby("algorithm")["ami"].mean().items():
            rows.append((algo, float(ser), "GNN"))
        rows.sort(key=lambda x: -x[1])
        out[net] = rows
    return out


def plot_social_top_ami(*, top_n: int = 15) -> None:
    ranks = _ami_rankings()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    colors = plt.cm.tab20(np.linspace(0, 1, top_n))
    for ax, net in zip(axes, NETWORK_ORDER):
        chunk = ranks[net][:top_n]
        y = np.arange(len(chunk))
        vals = [c[1] for c in chunk]
        labs = [f"{c[0]} ({c[2]})" for c in chunk]
        ax.barh(y, vals, color=colors[: len(chunk)])
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("AMI (mean)")
        ax.set_title(net.replace("_", " ").title())
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_ylabel("Method")
    fig.suptitle(
        "Attributed social networks: top methods by AMI\n"
        "(classical excludes SBM / nested SBM; GNN uses real node features)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_social_top_ami.{ext}")
    plt.close(fig)


def plot_social_feature_ablation() -> None:
    gn = load_gnn()
    ok = gn[gn["success"] & gn["ami"].notna()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    x = np.arange(len(GNN_ORDER))
    w = 0.25
    ft_colors = {"real": "#1f77b4", "structural": "#ff7f0e", "random": "#7f7f7f"}
    for ax, net in zip(axes, NETWORK_ORDER):
        for i, ft in enumerate(FEATURE_ORDER):
            means = []
            for algo in GNN_ORDER:
                cell = ok[
                    (ok["network"] == net)
                    & (ok["algorithm"] == algo)
                    & (ok["feature_type"] == ft)
                ]["ami"]
                means.append(float(cell.mean()) if len(cell) else np.nan)
            ax.bar(
                x + (i - 1) * w,
                means,
                width=w,
                label=ft,
                color=ft_colors.get(ft, "#333"),
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY.get(a, a) for a in GNN_ORDER], rotation=35, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("AMI (mean over seeds)")
        ax.set_title(net.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle("GNN feature ablation (real vs structural vs random)", fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_social_feature_ablation.{ext}")
    plt.close(fig)


def main() -> None:
    plot_social_top_ami(top_n=15)
    plot_social_feature_ablation()
    print(f"Wrote {OUT / 'fig_social_top_ami.pdf'}")
    print(f"Wrote {OUT / 'fig_social_feature_ablation.pdf'}")


if __name__ == "__main__":
    main()
