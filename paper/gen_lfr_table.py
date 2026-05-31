"""
Generate supplementary LaTeX tables for the LFR benchmark (Tables S1--S2).

Uses merged loaders from lfr_io.py (GPU CSVs + local supplement for missing classical algos).
"""

import pathlib

import numpy as np
import pandas as pd

from lfr_io import (
    GNN_BASE_ORDER,
    load_merged_accuracy_classical,
    load_merged_accuracy_gnn,
    load_merged_scalability_classical,
    load_merged_scalability_gnn,
)

GNN_ORDER = GNN_BASE_ORDER

ALGO_DISPLAY = {
    "louvain": "Louvain",
    "leiden": "Leiden",
    "fastgreedy": "FastGreedy",
    "label_propagation": "Label Prop.",
    "walktrap": "Walktrap",
    "spinglass": "Spinglass",
    "spectral": "Spectral",
    "leading_eigen": "Leading Eigen.",
    "sbm": "SBM",
    "nested_sbm": "Nested SBM",
    "rb_pots": "RB-Pots",
    "rber_pots": "RBer-Pots",
    "surprise": "Surprise",
    "em": "EM",
    "scan": "SCAN",
    "async_fluid": "Async Fluid",
    "cpm": "CPM",
    "angel": "Angel",
    "demon": "Demon",
    "kclique": "$k$-Clique",
    "score": "SCORE",
    "teamcs": "TeamCS",
    "bigs2": "BigS2",
    "csbio_iitm2": "CSBIO-IITM2",
    "agdl": "AGDL",
    "der": "DER",
    "gdmp2": "GDMP2",
    "simnet": "SimNet",
    "svt": "SVT",
    "tusk": "Tusk",
    "infomap": "Infomap",
    "dmon_gnn": "DMoN",
    "mincut_gnn": "MinCutPool",
    "vgae_gnn": "VGAE+KM",
    "dgi_gnn": "DGI+KM",
    "gcn_supervised": "GCN (20\\%)",
    "gat_supervised": "GAT (20\\%)",
    "mlp_kmeans": "MLP+KM",
    "node2vec_gnn": "Node2Vec+KM",
}

ALGO_FAMILY = {
    "louvain": "Modularity",
    "leiden": "Modularity",
    "fastgreedy": "Modularity",
    "walktrap": "Modularity",
    "label_propagation": "Propagation",
    "spinglass": "Propagation",
    "scan": "Propagation",
    "spectral": "Spectral",
    "leading_eigen": "Spectral",
    "sbm": "Statistical",
    "nested_sbm": "Statistical",
    "em": "Statistical",
    "rb_pots": "Resolution",
    "rber_pots": "Resolution",
    "cpm": "Resolution",
    "async_fluid": "Density",
    "surprise": "Density",
    "angel": "Overlapping",
    "demon": "Overlapping",
    "kclique": "Overlapping",
    "score": "DREAM",
    "teamcs": "DREAM",
    "bigs2": "DREAM",
    "csbio_iitm2": "DREAM",
    "agdl": "Other",
    "der": "Other",
    "gdmp2": "Other",
    "simnet": "Other",
    "svt": "Other",
    "tusk": "Other",
    "dmon_gnn": "Diff.\\ pool",
    "mincut_gnn": "Diff.\\ pool",
    "vgae_gnn": "Repr.\\ learn.",
    "dgi_gnn": "Repr.\\ learn.",
    "gcn_supervised": "Semi-sup.",
    "gat_supervised": "Semi-sup.",
    "mlp_kmeans": "Baseline",
    "node2vec_gnn": "Baseline",
}

CLASSICAL_ORDER = [
    "leiden",
    "louvain",
    "fastgreedy",
    "walktrap",
    "label_propagation",
    "spinglass",
    "scan",
    "spectral",
    "leading_eigen",
    "sbm",
    "nested_sbm",
    "em",
    "rb_pots",
    "rber_pots",
    "cpm",
    "async_fluid",
    "surprise",
    "angel",
    "demon",
    "kclique",
    "score",
    "teamcs",
    "bigs2",
    "csbio_iitm2",
    "agdl",
    "der",
    "gdmp2",
    "simnet",
    "svt",
    "tusk",
]


def load_accuracy():
    cl, _ = load_merged_accuracy_classical()
    gn = load_merged_accuracy_gnn()
    return cl, gn


def load_scalability():
    cl, _ = load_merged_scalability_classical()
    gn = load_merged_scalability_gnn()
    return cl, gn


def fmt(val, bold=False):
    if pd.isna(val):
        return "--"
    s = f"{val:.3f}"
    if bold:
        return f"\\textbf{{{s}}}"
    return s


def fmt_rt(val):
    if pd.isna(val):
        return "--"
    if val < 0.01:
        return f"{val:.4f}"
    if val < 10:
        return f"{val:.2f}"
    if val < 100:
        return f"{val:.1f}"
    return f"{val:.0f}"


def gen_accuracy_table():
    cl, gn = load_accuracy()
    mu_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{\textbf{Table S1.} Complete LFR accuracy results: mean AMI across 3 realizations "
        r"at selected mixing parameters $\mu$ ($N{=}1{,}000$). \textbf{Bold}: best in column per paradigm "
        r"(classical / GNN). Dash: algorithm not run at that $\mu$. "
        r"\textbf{SBM} and \textbf{Nested SBM}: AMI from local \texttt{accuracy\_results.csv} under \texttt{lfr/accuracy/results/} "
        r"when the same $\mu$ appears there; otherwise AMI stays GPU; runtime always GPU. "
        r"Other classical rows: GPU CSV (\texttt{results-clusternet-classical-gpu-new}) plus full-row local supplement where missing on GPU.}"
    )
    lines.append(r"\label{tab:supp_lfr_accuracy_full}")
    lines.append(r"\small")
    mu_headers = " & ".join(["$" + str(m) + "$" for m in mu_vals])
    lines.append(r"\begin{tabular}{ll" + "r" * len(mu_vals) + "}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Algorithm} & \textbf{Family} & \multicolumn{7}{c}{\textbf{AMI at mixing parameter} $\mu$} \\"
    )
    lines.append(r" & & " + mu_headers + r" \\")
    lines.append(r"\midrule")

    cl_means = {}
    for algo in CLASSICAL_ORDER:
        cl_means[algo] = {}
        for mu in mu_vals:
            sub = cl[(cl["algorithm"] == algo) & (np.isclose(cl["mu"], mu, atol=0.005))]
            cl_means[algo][mu] = sub["ami"].mean() if len(sub) > 0 else np.nan

    cl_best = {}
    for mu in mu_vals:
        vals = {a: cl_means[a][mu] for a in CLASSICAL_ORDER if not pd.isna(cl_means[a][mu])}
        cl_best[mu] = max(vals.values()) if vals else np.nan

    for algo in CLASSICAL_ORDER:
        name = ALGO_DISPLAY.get(algo, algo)
        family = ALGO_FAMILY.get(algo, "")
        cells = []
        for mu in mu_vals:
            v = cl_means[algo][mu]
            is_best = (not pd.isna(v)) and (not pd.isna(cl_best[mu])) and abs(v - cl_best[mu]) < 0.002
            cells.append(fmt(v, bold=is_best))
        lines.append(f"{name} & {family} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    gn_means = {}
    for algo in GNN_ORDER:
        gn_means[algo] = {}
        for mu in mu_vals:
            sub = gn[(gn["algorithm"] == algo) & (np.isclose(gn["mu"], mu, atol=0.005))]
            gn_means[algo][mu] = sub["ami"].mean() if len(sub) > 0 else np.nan

    gn_best = {}
    for mu in mu_vals:
        vals = {a: gn_means[a][mu] for a in GNN_ORDER if not pd.isna(gn_means[a][mu])}
        gn_best[mu] = max(vals.values()) if vals else np.nan

    for algo in GNN_ORDER:
        name = ALGO_DISPLAY.get(algo, algo)
        family = ALGO_FAMILY.get(algo, "")
        cells = []
        for mu in mu_vals:
            v = gn_means[algo][mu]
            is_best = (not pd.isna(v)) and (not pd.isna(gn_best[mu])) and abs(v - gn_best[mu]) < 0.002
            cells.append(fmt(v, bold=is_best))
        lines.append(f"{name} & {family} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def gen_scalability_table():
    cl, gn = load_scalability()
    n_show = [500, 1000, 2500, 5000, 25000]

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{\textbf{Table S2.} LFR scalability ($\mu{=}0.15$): mean AMI and runtime (seconds). "
        r"\textbf{Bold}: best AMI per column per paradigm. Dash: not run or failed. "
        r"\textbf{SBM} and \textbf{Nested SBM}: AMI from local scalability CSV when the same $N$ exists; "
        r"otherwise AMI stays GPU; \textbf{Time} always from GPU. Other classical rows: GPU scalability CSV plus local supplement for algorithms missing on GPU. GNN: nested path under \texttt{results-clusternet-classical-sca-gpu-3}.}"
    )
    lines.append(r"\label{tab:supp_lfr_scalability_full}")
    lines.append(r"\footnotesize")

    n_headers = []
    for n in n_show:
        nk = str(n // 1000) + "K" if n >= 1000 else str(n)
        n_headers.append("\\multicolumn{2}{c}{$N\\!=" + nk + "$}")
    header_str = " & ".join(n_headers)

    lines.append(r"\begin{tabular}{ll" + "rr" * len(n_show) + "}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Algorithm} & \textbf{Family} & " + header_str + r" \\")
    sub_headers = " & ".join(["AMI & Time"] * len(n_show))
    lines.append(r" & & " + sub_headers + r" \\")
    lines.append(r"\midrule")

    all_algos_cl = [a for a in CLASSICAL_ORDER if a in cl["algorithm"].unique()]
    all_algos_gn = [a for a in GNN_ORDER if a in gn["algorithm"].unique()]

    cl_best_ami = {}
    for n in n_show:
        vals = {}
        for algo in all_algos_cl:
            sub = cl[(cl["algorithm"] == algo) & (cl["N"] == n) & (cl["success"] == True)]
            if len(sub) > 0:
                vals[algo] = sub["ami"].mean()
        cl_best_ami[n] = max(vals.values()) if vals else np.nan

    for algo in all_algos_cl:
        name = ALGO_DISPLAY.get(algo, algo)
        family = ALGO_FAMILY.get(algo, "")
        cells = []
        for n in n_show:
            sub = cl[(cl["algorithm"] == algo) & (cl["N"] == n) & (cl["success"] == True)]
            if len(sub) > 0:
                ami_v = sub["ami"].mean()
                rt_v = sub["runtime"].mean()
                is_best = abs(ami_v - cl_best_ami[n]) < 0.002 if not pd.isna(cl_best_ami[n]) else False
                cells.append(f"{fmt(ami_v, bold=is_best)} & {fmt_rt(rt_v)}")
            else:
                cells.append("-- & --")
        lines.append(f"{name} & {family} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")

    gn_best_ami = {}
    for n in n_show:
        vals = {}
        for algo in all_algos_gn:
            sub = gn[(gn["algorithm"] == algo) & (gn["N"] == n) & (gn["success"] == True)]
            if len(sub) > 0:
                vals[algo] = sub["ami"].mean()
        gn_best_ami[n] = max(vals.values()) if vals else np.nan

    for algo in all_algos_gn:
        name = ALGO_DISPLAY.get(algo, algo)
        family = ALGO_FAMILY.get(algo, "")
        cells = []
        for n in n_show:
            sub = gn[(gn["algorithm"] == algo) & (gn["N"] == n) & (gn["success"] == True)]
            if len(sub) > 0:
                ami_v = sub["ami"].mean()
                rt_v = sub["runtime"].mean()
                is_best = abs(ami_v - gn_best_ami[n]) < 0.002 if not pd.isna(gn_best_ami[n]) else False
                cells.append(f"{fmt(ami_v, bold=is_best)} & {fmt_rt(rt_v)}")
            else:
                cells.append("-- & --")
        lines.append(f"{name} & {family} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = pathlib.Path(__file__).parent / "tables"
    out_dir.mkdir(exist_ok=True)

    print("Generating Supplementary Table S1 (accuracy)...")
    out_acc = out_dir / "lfr_accuracy_full.tex"
    out_acc.write_text(gen_accuracy_table())
    print(f"  -> {out_acc}")

    print("Generating Supplementary Table S2 (scalability)...")
    out_sca = out_dir / "lfr_scalability_full.tex"
    out_sca.write_text(gen_scalability_table())
    print(f"  -> {out_sca}")
