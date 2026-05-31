#!/usr/bin/env python3
"""
Generate supplementary tables for Tier-3 (biological) benchmark.

Tables:
  S6: Full enrichment table (all networks × algorithms)
  S7: Perturbation stability summary (GRN)
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from bio_io import (
    ALGO_DISPLAY,
    GNN_ALGOS,
    NETWORK_DISPLAY,
    NETWORK_ORDER,
    load_enrichment_averaged,
    load_perturbation,
    perturbation_summary,
)

TABLES_DIR = pathlib.Path(__file__).parent / "tables"
TABLES_DIR.mkdir(exist_ok=True)


def _esc(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def enrichment_full_table() -> str:
    """Full enrichment table: all networks × algorithms."""
    df = load_enrichment_averaged()
    lines = []
    lines.append(r"\begin{longtable}{llrrrrr}")
    lines.append(r"\caption{GO and KEGG enrichment fractions for all algorithms on four biological networks. "
                 r"Values are fraction of tested communities ($\geq$5 nodes) with at least one significant term "
                 r"(BH-adjusted $p < 0.05$). Sign coherence reported for SIGNOR only.}")
    lines.append(r"\label{tab:bio_enrichment_full} \\")
    lines.append(r"\toprule")
    lines.append(r"Network & Algorithm & $K$ & Tested & GO frac & KEGG frac & Sign coh. \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Network & Algorithm & $K$ & Tested & GO frac & KEGG frac & Sign coh. \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    for net in NETWORK_ORDER:
        sub = df[df["network"] == net].sort_values("go_enriched_frac", ascending=False)
        for _, r in sub.iterrows():
            algo_d = ALGO_DISPLAY.get(r["algorithm"], r["algorithm"])
            tag = r" \textit{(GNN)}" if r["type"] == "GNN" else ""
            sc = f"{r['sign_coherence']:.3f}" if not np.isnan(r.get("sign_coherence", np.nan)) else "---"
            lines.append(
                rf"{NETWORK_DISPLAY[net]} & {_esc(algo_d)}{tag} & "
                rf"{int(r['num_communities'])} & {int(r['communities_tested'])} & "
                rf"{r['go_enriched_frac']:.3f} & {r['kegg_enriched_frac']:.3f} & {sc} \\"
            )
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def perturbation_table() -> str:
    """Perturbation stability table for GRN."""
    ps = perturbation_summary()
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Perturbation stability on Human GRN. Mean NMI and FFL change across 3 seeds. "
                 r"TF knockout removes top-$k$ transcription factors by out-degree; controls remove $k$ random genes.}")
    lines.append(r"\label{tab:bio_perturbation}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Test & Algorithm & NMI & AMI & FFL orig & $\Delta$FFL \\")
    lines.append(r"\midrule")

    test_order = [
        "tf_knockout_top5", "tf_knockout_top10",
        "tf_knockout_control_random5", "tf_knockout_control_random10",
        "edge_dropout_0.05", "edge_dropout_0.1", "edge_dropout_0.2", "edge_dropout_0.3",
        "gene_removal_hub_0.01", "gene_removal_hub_0.02", "gene_removal_hub_0.05",
    ]
    test_display = {
        "tf_knockout_top5": "TF KO top-5",
        "tf_knockout_top10": "TF KO top-10",
        "tf_knockout_control_random5": "Random-5 ctrl",
        "tf_knockout_control_random10": "Random-10 ctrl",
        "edge_dropout_0.05": "Edge drop 5\\%",
        "edge_dropout_0.1": "Edge drop 10\\%",
        "edge_dropout_0.2": "Edge drop 20\\%",
        "edge_dropout_0.3": "Edge drop 30\\%",
        "gene_removal_hub_0.01": "Hub removal 1\\%",
        "gene_removal_hub_0.02": "Hub removal 2\\%",
        "gene_removal_hub_0.05": "Hub removal 5\\%",
    }
    algo_order = ["leiden", "louvain", "rb_pots", "fastgreedy",
                  "dmon_gnn", "mincut_gnn", "dgi_gnn", "gcn_supervised", "mlp_kmeans"]

    for test in test_order:
        tsub = ps[ps["test"] == test]
        if tsub.empty:
            continue
        first = True
        for algo in algo_order:
            row = tsub[tsub["algorithm"] == algo]
            if row.empty:
                continue
            r = row.iloc[0]
            td = test_display.get(test, test) if first else ""
            first = False
            ad = ALGO_DISPLAY.get(algo, algo)
            lines.append(
                rf"{_esc(td)} & {_esc(ad)} & {r['nmi']:.3f} & {r['ami']:.3f} & "
                rf"{r['ffl_preserved_orig']:.3f} & {r['ffl_change']:+.3f} \\"
            )
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    e = enrichment_full_table()
    (TABLES_DIR / "bio_enrichment_full.tex").write_text(e, encoding="utf-8")
    print(f"Wrote tables/bio_enrichment_full.tex")

    p = perturbation_table()
    (TABLES_DIR / "bio_perturbation.tex").write_text(p, encoding="utf-8")
    print(f"Wrote tables/bio_perturbation.tex")


if __name__ == "__main__":
    main()
