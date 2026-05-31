#!/usr/bin/env python3
"""
Generate biological benchmark figures for the paper.

Figures:
  fig_bio_enrichment    - GO/KEGG enrichment bar chart (4 networks, top algorithms)
  fig_bio_perturbation  - Perturbation stability (NMI + FFL change)
"""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bio_io import (
    ALGO_DISPLAY,
    GNN_ALGOS,
    NETWORK_DISPLAY,
    NETWORK_ORDER,
    load_enrichment_averaged,
    perturbation_summary,
)

OUT = pathlib.Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def fig_bio_enrichment():
    """Bar chart: GO enrichment fraction per algorithm on each network."""
    df = load_enrichment_averaged()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, net in enumerate(NETWORK_ORDER):
        ax = axes[idx]
        sub = df[df["network"] == net].sort_values("go_enriched_frac", ascending=False).head(15)

        colors = ["#2196F3" if a in GNN_ALGOS else "#4CAF50" for a in sub["algorithm"]]
        labels = [ALGO_DISPLAY.get(a, a) for a in sub["algorithm"]]

        x = np.arange(len(sub))
        bars_go = ax.bar(x - 0.2, sub["go_enriched_frac"], 0.4,
                         color=colors, alpha=0.8, label="GO")
        bars_kegg = ax.bar(x + 0.2, sub["kegg_enriched_frac"], 0.4,
                           color=colors, alpha=0.4, label="KEGG")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Enrichment fraction")
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{NETWORK_DISPLAY[net]} ({net})")
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("GO and KEGG Enrichment Fraction (top 15 algorithms per network)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_bio_enrichment.pdf")
    fig.savefig(OUT / "fig_bio_enrichment.png")
    plt.close(fig)
    print("  -> saved fig_bio_enrichment.pdf/.png")


def fig_bio_perturbation():
    """Perturbation stability: NMI and FFL change."""
    ps = perturbation_summary()

    algo_order = ["leiden", "louvain", "fastgreedy", "rb_pots",
                  "dmon_gnn", "mincut_gnn", "dgi_gnn", "gcn_supervised", "mlp_kmeans"]
    algo_labels = [ALGO_DISPLAY.get(a, a) for a in algo_order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: NMI by perturbation type
    test_groups = {
        "TF KO top-5": "tf_knockout_top5",
        "TF KO top-10": "tf_knockout_top10",
        "Random-5 ctrl": "tf_knockout_control_random5",
        "Random-10 ctrl": "tf_knockout_control_random10",
        "Edge drop 5%": "edge_dropout_0.05",
        "Edge drop 10%": "edge_dropout_0.1",
        "Edge drop 20%": "edge_dropout_0.2",
        "Edge drop 30%": "edge_dropout_0.3",
        "Hub removal 1%": "gene_removal_hub_0.01",
    }

    x = np.arange(len(algo_order))
    width = 0.8 / len(test_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_groups)))

    for i, (label, test) in enumerate(test_groups.items()):
        tsub = ps[ps["test"] == test]
        nmis = []
        for algo in algo_order:
            row = tsub[tsub["algorithm"] == algo]
            nmis.append(float(row["nmi"].iloc[0]) if len(row) > 0 else 0)
        offset = (i - len(test_groups) / 2 + 0.5) * width
        ax1.bar(x + offset, nmis, width, label=label, color=colors[i], alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(algo_labels, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("NMI (vs. baseline)")
    ax1.set_title("Partition Stability Under Perturbation (Human GRN)")
    ax1.legend(fontsize=6, ncol=2, loc="upper right")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.2)

    # Panel 2: FFL change (TF knockout vs control)
    ffl_tests = {
        "TF KO top-5": "tf_knockout_top5",
        "TF KO top-10": "tf_knockout_top10",
        "Random-5 ctrl": "tf_knockout_control_random5",
        "Random-10 ctrl": "tf_knockout_control_random10",
    }
    width2 = 0.8 / len(ffl_tests)
    ffl_colors = ["#d32f2f", "#f57c00", "#388e3c", "#1976d2"]

    for i, (label, test) in enumerate(ffl_tests.items()):
        tsub = ps[ps["test"] == test]
        ffls = []
        for algo in algo_order:
            row = tsub[tsub["algorithm"] == algo]
            ffls.append(float(row["ffl_change"].iloc[0]) if len(row) > 0 else 0)
        offset = (i - len(ffl_tests) / 2 + 0.5) * width2
        ax2.bar(x + offset, ffls, width2, label=label, color=ffl_colors[i], alpha=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels(algo_labels, rotation=35, ha="right", fontsize=8)
    ax2.set_ylabel("FFL preservation change")
    ax2.set_title("FFL Change: TF Knockout vs Random Control")
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(OUT / "fig_bio_perturbation.pdf")
    fig.savefig(OUT / "fig_bio_perturbation.png")
    plt.close(fig)
    print("  -> saved fig_bio_perturbation.pdf/.png")


def main():
    print("Generating biological benchmark figures...")
    fig_bio_enrichment()
    fig_bio_perturbation()
    print(f"All biological figures in: {OUT}")


if __name__ == "__main__":
    main()
