"""
Remove dmon_gnn, gcn_supervised, gat_supervised (+ ablation variants) results
from LFR benchmark CSV files and partition directories so they can be rerun
with higher epochs.
"""
import pandas as pd
from pathlib import Path

ALGOS_TO_CLEAR = [
    "dmon_gnn", "gcn_supervised", "gat_supervised",
    "gcn_supervised_40", "gat_supervised_40",
    "gcn_supervised_60", "gat_supervised_60",
    "gcn_supervised_80", "gat_supervised_80",
]

HIER_ALGOS = ["dmon_gnn"]


def clean_csv(csv_path, algo_col="algorithm", algos=None):
    algos = algos or ALGOS_TO_CLEAR
    if not csv_path.is_file():
        print(f"  [skip] {csv_path} does not exist")
        return
    df = pd.read_csv(csv_path)
    before = len(df)
    mask = df[algo_col].isin(algos)
    removed = mask.sum()
    df = df[~mask]
    df.to_csv(csv_path, index=False)
    print(f"  [csv]  {csv_path}: {before} -> {len(df)} rows ({removed} removed)")


def clean_partitions(partition_dir, algos=None):
    algos = algos or ALGOS_TO_CLEAR
    if not partition_dir.is_dir():
        print(f"  [skip] {partition_dir} does not exist")
        return
    count = 0
    for algo in algos:
        for f in partition_dir.glob(f"*_{algo}.json"):
            f.unlink()
            count += 1
    print(f"  [part] {partition_dir}: {count} files deleted")


# Accuracy — real results at /mnt/d/results-clusternet-classical/
ACC = Path("/mnt/d/results-clusternet-classical")
print("=== ACCURACY ===")
clean_csv(ACC / "results" / "gnn_accuracy_results.csv")
clean_partitions(ACC / "results" / "partitions" / "gnn")

# Scalability — real results at /mnt/d/results-clusternet-classical-sca/
SCA = Path("/mnt/d/results-clusternet-classical-sca")
print("\n=== SCALABILITY ===")
clean_csv(SCA / "results" / "gnn_scalability_results.csv")
clean_partitions(SCA / "results" / "partitions" / "gnn")

# Hierarchical — results at lfr/hierarchical/ (already cleaned, but redo for safety)
HIER = Path("/mnt/d/cluster/ClusterNet/Optimized_algos/benchmarks/lfr/hierarchical")
print("\n=== HIERARCHICAL ===")
hier_algos_expanded = [f"{a}_{lvl}" for a in HIER_ALGOS for lvl in ("micro", "macro")]
clean_csv(HIER / "results" / "gnn_hierarchical_results.csv", algos=hier_algos_expanded)

print("\nDone. Now rerun with --resume --epochs 600 --run-gnn pointing to these dirs.")
