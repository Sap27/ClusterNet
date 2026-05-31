"""
Disease Module Overlap Analysis.

Tests whether communities detected on BioGRID (PPI network) significantly overlap
with known disease gene sets from OMIM. Uses Fisher's exact test with BH correction.

Inputs:
  - OMIM gene-disease TSV (macarthur-lab format)
  - BioGRID partition JSONs (all algorithms)
  - BioGRID node_list.json

Output:
  - tables/disease_module_overlap.csv
  - figures/fig_bio_disease_overlap.pdf
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from collections import defaultdict
from pathlib import Path

BASE = "/mnt/d/cluster/BIologicalNetworks"
PARTITION_DIRS = [
    f"{BASE}/results/partitions",
    "/mnt/d/results-biological-networks-classical2/results-biological-networks-classical/results/partitions",
    "/mnt/d/results-perturb-gnn-biological/results/partitions",
]
DATA_DIR = f"{BASE}/data/biogrid"
OMIM_PATH = "/mnt/d/cluster/ClusterNet/paper/data/omim_full.tsv"
OUT_CSV = "/mnt/d/cluster/ClusterNet/paper/tables/disease_module_overlap.csv"
OUT_FIG = "/mnt/d/cluster/ClusterNet/paper/figures/fig_bio_disease_overlap.pdf"

MIN_DISEASE_GENES = 10
MIN_COMMUNITY_SIZE = 5
PVAL_THRESHOLD = 0.05

GNN_ALGORITHMS = {
    "dmon_gnn", "mincut_gnn", "vgae_gnn", "dgi_gnn",
    "node2vec_gnn", "gat_supervised", "gcn_supervised", "mlp_kmeans"
}


def load_omim_disease_sets():
    """Parse OMIM TSV into disease -> gene set mapping."""
    df = pd.read_csv(OMIM_PATH, sep="\t")
    disease_genes = defaultdict(set)

    for _, row in df.iterrows():
        phenotype = str(row.get("phenotype", ""))
        if phenotype == "nan" or not phenotype.strip():
            continue
        genes_col = str(row.get("hgnc_genes", ""))
        if genes_col == "nan" or not genes_col.strip():
            continue
        genes = [g.strip() for g in genes_col.split(",") if g.strip()]
        disease_name = phenotype.strip().rstrip(",").split(",")[0].strip()
        if disease_name:
            for g in genes:
                disease_genes[disease_name].add(g)

    filtered = {d: gs for d, gs in disease_genes.items() if len(gs) >= MIN_DISEASE_GENES}
    print(f"OMIM: {len(disease_genes)} diseases total, {len(filtered)} with >= {MIN_DISEASE_GENES} genes")
    return filtered


def load_biogrid_node_list():
    with open(f"{DATA_DIR}/node_list.json") as f:
        return json.load(f)


def find_partition(algo):
    """Find a biogrid partition across all partition directories. For GNNs with seeds, use seed 0."""
    candidates = [
        f"biogrid_{algo}.json",
        f"biogrid_{algo}_s0.json",
    ]
    for pdir in PARTITION_DIRS:
        for fname in candidates:
            path = os.path.join(pdir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    return json.load(f)
    return None


def compute_overlap(partition, node_list, disease_sets, background_size):
    """
    For each community, test overlap with each disease gene set (Fisher exact).
    Returns the fraction of disease classes with at least one significant community.
    """
    node_set = set(node_list)
    results = []

    valid_communities = [c for c in partition if len(c) >= MIN_COMMUNITY_SIZE]
    if not valid_communities:
        return 0.0, 0, []

    all_pvalues = []
    all_labels = []

    for disease_name, disease_gene_set in disease_sets.items():
        disease_in_network = disease_gene_set & node_set
        if len(disease_in_network) < 3:
            continue

        best_pval = 1.0
        best_comm_idx = -1

        for ci, comm in enumerate(valid_communities):
            comm_genes = set(node_list[idx] for idx in comm if idx < len(node_list))
            overlap = len(comm_genes & disease_in_network)
            comm_size = len(comm_genes)
            disease_size = len(disease_in_network)

            # 2x2 contingency: overlap, comm-only, disease-only, neither
            a = overlap
            b = comm_size - overlap
            c = disease_size - overlap
            d = background_size - comm_size - disease_size + overlap

            if a == 0:
                continue

            _, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
            if pval < best_pval:
                best_pval = pval
                best_comm_idx = ci

        all_pvalues.append(best_pval)
        all_labels.append(disease_name)

    if not all_pvalues:
        return 0.0, 0, []

    # BH correction
    reject, corrected_pvals, _, _ = multipletests(all_pvalues, alpha=PVAL_THRESHOLD, method="fdr_bh")
    recovered = sum(reject)
    recovery_rate = recovered / len(reject)

    sig_diseases = [all_labels[i] for i in range(len(reject)) if reject[i]]

    return recovery_rate, recovered, sig_diseases


def main():
    print("Loading OMIM disease gene sets...")
    disease_sets = load_omim_disease_sets()

    print("Loading BioGRID node list...")
    node_list = load_biogrid_node_list()
    background_size = len(node_list)
    print(f"BioGRID: {background_size} genes")

    # Collect all biogrid partition files from all directories
    import glob
    algo_set = set()
    for pdir in PARTITION_DIRS:
        for f in glob.glob(f"{pdir}/biogrid_*.json"):
            name = os.path.basename(f).replace("biogrid_", "").replace(".json", "")
            # Strip seed suffixes like _s0, _s1, _s2
            if name.endswith("_s0") or name.endswith("_s1") or name.endswith("_s2"):
                name = name[:-3]
            algo_set.add(name)
    algorithms = sorted(algo_set)
    print(f"Found {len(algorithms)} algorithms with BioGRID partitions")

    rows = []
    for algo in algorithms:
        partition = find_partition(algo)
        if partition is None:
            continue
        algo_type = "GNN" if algo in GNN_ALGORITHMS else "Classical"
        n_communities = len([c for c in partition if len(c) >= MIN_COMMUNITY_SIZE])

        recovery_rate, n_recovered, sig_diseases = compute_overlap(
            partition, node_list, disease_sets, background_size
        )
        rows.append({
            "algorithm": algo,
            "type": algo_type,
            "n_communities": n_communities,
            "diseases_tested": len(disease_sets),
            "diseases_recovered": n_recovered,
            "recovery_rate": recovery_rate,
            "top_diseases": "; ".join(sig_diseases[:10]),
        })
        print(f"  {algo} ({algo_type}): {n_recovered}/{len(disease_sets)} diseases recovered ({recovery_rate:.2%})")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved results to {OUT_CSV}")

    # Generate figure
    plot_results(df)


def plot_results(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df_sorted = df.sort_values("recovery_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 10))
    colors = ["#2196F3" if t == "GNN" else "#4CAF50" for t in df_sorted["type"]]
    bars = ax.barh(range(len(df_sorted)), df_sorted["recovery_rate"], color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["algorithm"], fontsize=8)
    ax.set_xlabel("Disease Module Recovery Rate\n(fraction of OMIM disease classes with at least one significant community overlap)")
    ax.set_title("Disease Module Overlap: BioGRID PPI Communities vs OMIM Disease Gene Sets")
    ax.axvline(x=df_sorted["recovery_rate"].median(), color="gray", linestyle="--", alpha=0.5, label="Median")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#4CAF50", label="Classical"),
                       Patch(facecolor="#2196F3", label="GNN")]
    ax.legend(handles=legend_elements, loc="lower right")

    ax.set_xlim(0, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {OUT_FIG}")
    plt.close()


if __name__ == "__main__":
    main()
