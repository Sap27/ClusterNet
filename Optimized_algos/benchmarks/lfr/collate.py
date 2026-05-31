#!/usr/bin/env python3
"""
Rebuild benchmark CSVs from saved partition JSON files.

This script is useful when result CSVs were overwritten but partition files exist.
It computes AMI/NMI directly from each partition's `partition` and `ground_truth`.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score


def parse_network_name(network_name: str):
    """Parse network name like mu0.1_r3 into (mu, realization)."""
    match = re.match(r"^mu([0-9]*\.?[0-9]+)_r(\d+)$", str(network_name))
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), int(match.group(2))


def infer_benchmark_fields(partition_group: str, algorithm: str, extra_info: dict):
    """Infer algorithm_type/training and optional feature column fields."""
    row = {
        "algorithm_type": "unknown",
        "training": "unknown",
        "feature_distance": np.nan,
        "feature_type": np.nan,
    }

    if partition_group == "classical":
        row["algorithm_type"] = "classical"
        row["training"] = "none"
        return row

    if partition_group == "gnn_homophily":
        row["feature_distance"] = extra_info.get("fd", np.nan)
        if algorithm in {"GCN_semisup", "GAT_semisup"}:
            row["algorithm_type"] = "gnn_semisupervised"
            row["training"] = "per_network"
        elif algorithm == "kmeans_homophily":
            row["algorithm_type"] = "baseline"
            row["training"] = "none"
        else:
            row["algorithm_type"] = "gnn_unsupervised"
            row["training"] = "global_zeroshot"
        return row

    if partition_group == "gnn_features":
        row["feature_type"] = extra_info.get("ft", np.nan)
        if algorithm in {"GCN_semisup", "GAT_semisup"}:
            row["algorithm_type"] = "gnn_semisupervised"
            row["training"] = "per_network"
        elif algorithm == "KMeans":
            row["algorithm_type"] = "baseline"
            row["training"] = "none"
        else:
            row["algorithm_type"] = "gnn_unsupervised"
            row["training"] = "global_zeroshot"
        return row

    # Fallback for unknown partition groups
    if "fd" in extra_info:
        row["feature_distance"] = extra_info["fd"]
    if "ft" in extra_info:
        row["feature_type"] = extra_info["ft"]
    return row


def build_rows_from_partitions(partitions_dir: Path):
    """Load all partition JSONs and compute recoverable benchmark rows."""
    rows = []
    partition_files = sorted(partitions_dir.glob("*/*.json"))

    for part_file in partition_files:
        partition_group = part_file.parent.name
        try:
            with open(part_file, "r") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Warning: cannot read {part_file}: {exc}")
            continue

        network = data.get("network", "")
        algorithm = data.get("algorithm", "")
        extra_info = data.get("extra_info", {}) or {}
        pred = np.array(data.get("partition", []))
        gt = np.array(data.get("ground_truth", []))

        mu, realization = parse_network_name(network)
        base_fields = infer_benchmark_fields(partition_group, algorithm, extra_info)

        if len(pred) == 0:
            print(f"Warning: empty partition in {part_file}")
            continue

        if len(gt) == len(pred) and len(gt) > 0:
            ami = adjusted_mutual_info_score(gt, pred)
            nmi = normalized_mutual_info_score(gt, pred)
            success = True
            error = ""
            num_true = int(len(np.unique(gt)))
        else:
            ami = np.nan
            nmi = np.nan
            success = False
            error = "ground_truth missing or length mismatch"
            num_true = np.nan

        row = {
            "partition_group": partition_group,
            "network": network,
            "mu": mu,
            "realization": realization,
            "algorithm": algorithm,
            "success": success,
            "runtime": np.nan,  # Not recoverable from partition files
            "ami": ami,
            "nmi": nmi,
            "num_detected": int(len(np.unique(pred))),
            "num_true": num_true,
            "error": error,
        }
        row.update(base_fields)
        rows.append(row)

    return rows


def dedup_results(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows based on logical run keys."""
    key_cols = [
        "partition_group",
        "network",
        "mu",
        "realization",
        "algorithm",
        "algorithm_type",
        "training",
        "feature_distance",
        "feature_type",
    ]
    available_keys = [c for c in key_cols if c in df.columns]
    return df.drop_duplicates(subset=available_keys, keep="last")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary grouped by available benchmark dimensions."""
    df_ok = df[df["success"] == True].copy()
    if len(df_ok) == 0:
        return pd.DataFrame()

    group_cols = ["partition_group", "mu", "algorithm"]
    if "feature_distance" in df_ok.columns and df_ok["feature_distance"].notna().any():
        group_cols.append("feature_distance")
    if "feature_type" in df_ok.columns and df_ok["feature_type"].notna().any():
        group_cols.append("feature_type")

    summary = (
        df_ok.groupby(group_cols)
        .agg(
            ami_mean=("ami", "mean"),
            ami_std=("ami", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            runs=("network", "count"),
        )
        .reset_index()
        .round(4)
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Rebuild benchmark CSVs from partition JSONs")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Path to results directory (contains partitions/). "
             "Default: benchmarks/lfr/accuracy/results relative to this script.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="recovered_from_partitions",
        help="Prefix for output CSV filenames.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    default_results = script_dir / "accuracy" / "results"
    results_dir = Path(args.results_dir) if args.results_dir else default_results
    partitions_dir = results_dir / "partitions"

    if not partitions_dir.exists():
        raise FileNotFoundError(f"Partitions directory not found: {partitions_dir}")

    rows = build_rows_from_partitions(partitions_dir)
    if len(rows) == 0:
        print(f"No partition JSON files found in: {partitions_dir}")
        return

    results_df = dedup_results(pd.DataFrame(rows))
    summary_df = build_summary(results_df)

    results_csv = results_dir / f"{args.output_prefix}_results.csv"
    summary_csv = results_dir / f"{args.output_prefix}_summary.csv"

    results_df.to_csv(results_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"Recovered rows: {len(results_df)}")
    print(f"Results CSV: {results_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
