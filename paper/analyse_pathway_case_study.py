"""
Pathway Case Study: Extract named GO terms and KEGG pathways for top communities.

For the best-performing partitions on SIGNOR and BioGRID, queries g:Profiler
for the 5 largest communities and reports the top enriched pathway per community.

Output: tables/pathway_case_study.csv
"""
import json, time, sys, os
import pandas as pd
from gprofiler import GProfiler

BASE = "/mnt/d/cluster/BIologicalNetworks"
PARTITION_DIR = f"{BASE}/results/partitions"
DATA_DIR = f"{BASE}/data"
OUT_CSV = "/mnt/d/cluster/ClusterNet/paper/tables/pathway_case_study.csv"

MIN_COMMUNITY_SIZE = 10
TOP_K_COMMUNITIES = 5
PAUSE_BETWEEN_QUERIES = 1.5

PARTITIONS_TO_STUDY = [
    ("signor", "leiden"),
    ("signor", "sbm"),
    ("signor", "louvain"),
    ("biogrid", "leiden"),
    ("biogrid", "louvain"),
    ("biogrid", "infomap"),
    ("grn", "leiden"),
    ("grn", "infomap"),
    ("grn", "fastgreedy"),
]

gp = GProfiler(return_dataframe=True)


def load_partition(network, algo):
    path = f"{PARTITION_DIR}/{network}_{algo}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_node_list(network):
    path = f"{DATA_DIR}/{network}/node_list.json"
    with open(path) as f:
        return json.load(f)


def query_gprofiler(gene_list, organism="hsapiens"):
    """Query g:Profiler with retry logic."""
    for attempt in range(3):
        try:
            result = gp.profile(
                organism=organism,
                query=gene_list,
                sources=["GO:BP", "GO:MF", "GO:CC", "KEGG"],
                no_evidences=True,
            )
            return result
        except Exception as e:
            print(f"  g:Profiler attempt {attempt+1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def main():
    rows = []
    for network, algo in PARTITIONS_TO_STUDY:
        print(f"\n=== {network} / {algo} ===")
        partition = load_partition(network, algo)
        if partition is None:
            print(f"  Partition not found, skipping.")
            continue
        node_list = load_node_list(network)

        sorted_comms = sorted(enumerate(partition), key=lambda x: -len(x[1]))
        top_comms = [(i, c) for i, c in sorted_comms if len(c) >= MIN_COMMUNITY_SIZE][:TOP_K_COMMUNITIES]

        for comm_idx, comm_nodes in top_comms:
            genes = [node_list[idx] for idx in comm_nodes if idx < len(node_list)]
            print(f"  Community {comm_idx} ({len(genes)} genes) ...")
            result = query_gprofiler(genes)
            time.sleep(PAUSE_BETWEEN_QUERIES)

            if result.empty:
                rows.append({
                    "network": network, "algorithm": algo,
                    "community_id": comm_idx, "community_size": len(genes),
                    "top_go_term": "N/A", "go_name": "N/A", "go_pvalue": None,
                    "top_kegg": "N/A", "kegg_name": "N/A", "kegg_pvalue": None,
                })
                continue

            go_results = result[result["source"].str.startswith("GO:")]
            kegg_results = result[result["source"] == "KEGG"]

            top_go = "N/A"
            go_name = "N/A"
            go_pval = None
            if not go_results.empty:
                best_go = go_results.sort_values("p_value").iloc[0]
                top_go = best_go["native"]
                go_name = best_go["name"]
                go_pval = best_go["p_value"]

            top_kegg = "N/A"
            kegg_name = "N/A"
            kegg_pval = None
            if not kegg_results.empty:
                best_kegg = kegg_results.sort_values("p_value").iloc[0]
                top_kegg = best_kegg["native"]
                kegg_name = best_kegg["name"]
                kegg_pval = best_kegg["p_value"]

            rows.append({
                "network": network, "algorithm": algo,
                "community_id": comm_idx, "community_size": len(genes),
                "top_go_term": top_go, "go_name": go_name, "go_pvalue": go_pval,
                "top_kegg": top_kegg, "kegg_name": kegg_name, "kegg_pvalue": kegg_pval,
            })
            print(f"    GO: {go_name} (p={go_pval:.2e})" if go_pval else "    GO: none")
            print(f"    KEGG: {kegg_name} (p={kegg_pval:.2e})" if kegg_pval else "    KEGG: none")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ Saved {len(df)} rows to {OUT_CSV}")

    print("\n=== Summary ===")
    for net in df["network"].unique():
        sub = df[df["network"] == net]
        go_hit = sub["go_pvalue"].notna().sum()
        kegg_hit = sub["kegg_pvalue"].notna().sum()
        print(f"  {net}: {go_hit}/{len(sub)} communities have GO enrichment, {kegg_hit}/{len(sub)} have KEGG")


if __name__ == "__main__":
    main()
