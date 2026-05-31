"""
Generate supplementary LaTeX tables for the social benchmark (Tier~2).

Classical rows exclude sbm / nested_sbm / nested_sbm_coarse (unreliable fallback runs).
"""

from __future__ import annotations

import pathlib

import pandas as pd

from social_io import (
    GNN_ORDER,
    NETWORK_ORDER,
    FEATURE_ORDER,
    load_classical_filtered,
    load_gnn,
)


def _tex_escape(s: str) -> str:
    return str(s).replace("\\", "/").replace("_", r"\_").replace("%", r"\%")


def _fmt(v: float | None) -> str:
    if v is None or (isinstance(v, float) and (v != v)):  # nan
        return "---"
    x = float(v)
    if abs(x) < 5e-4:
        x = 0.0
    return f"{x:.3f}"


def classical_wide_table() -> str:
    df = load_classical_filtered()
    ok = df[df["success"] & df["ami"].notna()]
    agg = (
        ok.groupby(["algorithm", "network"])["ami"]
        .mean()
        .unstack()
        .reindex(columns=NETWORK_ORDER)
    )
    algos = sorted(agg.index.tolist())
    lines = [
        r"\small",
        r"\begin{tabular}{l" + "r" * len(NETWORK_ORDER) + "}",
        r"\toprule",
        r"\textbf{Algorithm} "
        + "".join(
            rf"& \textbf{{{n.replace('_', ' ').title()}}} " for n in NETWORK_ORDER
        )
        + r"\\",
        r"\midrule",
    ]
    for a in algos:
        row = [_tex_escape(a)]
        for net in NETWORK_ORDER:
            val = agg.loc[a, net] if a in agg.index else float("nan")
            row.append(_fmt(float(val)) if pd.notna(val) else "---")
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\normalsize"])
    return "\n".join(lines)


def gnn_ablation_table() -> str:
    """One row per (GNN, feature type); columns are mean AMI on each network."""
    gn = load_gnn()
    ok = gn[gn["success"] & gn["ami"].notna()]
    hdr = (
        r"\textbf{GNN} & \textbf{Features} "
        + "".join(rf"& \textbf{{{net.replace('_', ' ').title()}}} " for net in NETWORK_ORDER)
        + r"\\"
    )
    lines = [
        r"\footnotesize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        hdr,
        r"\midrule",
    ]
    for algo in GNN_ORDER:
        sub = ok[ok["algorithm"] == algo]
        if sub.empty:
            continue
        for i, ft in enumerate(FEATURE_ORDER):
            row = [_tex_escape(algo) if i == 0 else "", _tex_escape(ft)]
            for net in NETWORK_ORDER:
                cell = sub[(sub["network"] == net) & (sub["feature_type"] == ft)]["ami"]
                row.append(_fmt(float(cell.mean())) if len(cell) else "---")
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\addlinespace[2pt]")
    while lines and lines[-1] == r"\addlinespace[2pt]":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\normalsize"])
    return "\n".join(lines)


def top_per_network_tex(*, k: int = 10) -> str:
    """Rank classical + GNN (real features only) by mean AMI."""
    cl = load_classical_filtered()
    gn = load_gnn()
    cl_ok = cl[cl["success"] & cl["ami"].notna()]
    gn_ok = gn[gn["success"] & gn["ami"].notna() & (gn["feature_type"] == "real")]

    lines = [
        r"\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Network} & \textbf{Method} & \textbf{Type} & \textbf{AMI} \\",
        r"\midrule",
    ]
    for net in NETWORK_ORDER:
        rows = []
        for algo, ami in (
            cl_ok[cl_ok["network"] == net]
            .groupby("algorithm")["ami"]
            .mean()
            .sort_values(ascending=False)
            .items()
        ):
            rows.append((algo, "classical", float(ami)))
        for algo, ami in (
            gn_ok[gn_ok["network"] == net]
            .groupby("algorithm")["ami"]
            .mean()
            .sort_values(ascending=False)
            .items()
        ):
            rows.append((algo, "GNN (real)", float(ami)))
        rows.sort(key=lambda x: -x[2])
        shown = rows[:k]
        net_disp = net.replace("_", " ").title()
        for j, (algo, typ, ami) in enumerate(shown):
            pref = net_disp if j == 0 else ""
            lines.append(
                rf"{_tex_escape(pref)} & \texttt{{{_tex_escape(algo)}}} & {_tex_escape(typ)} & {_fmt(ami)} \\"
            )
        lines.append(r"\addlinespace[4pt]")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\normalsize")
    return "\n".join(lines)


def main() -> None:
    root = pathlib.Path(__file__).parent
    td = root / "tables"
    td.mkdir(exist_ok=True)
    (td / "social_classical_full.tex").write_text(classical_wide_table(), encoding="utf-8")
    (td / "social_gnn_ablation_full.tex").write_text(gnn_ablation_table(), encoding="utf-8")
    (td / "social_top_per_network.tex").write_text(top_per_network_tex(k=10), encoding="utf-8")
    (td / "social_main_top5.tex").write_text(top_per_network_tex(k=5), encoding="utf-8")
    print("Wrote tables/social_classical_full.tex")
    print("Wrote tables/social_gnn_ablation_full.tex")
    print("Wrote tables/social_top_per_network.tex")
    print("Wrote tables/social_main_top5.tex")


if __name__ == "__main__":
    main()
