#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# aggregate_coverage.py
import glob
import pandas as pd


def load(map_name: str) -> pd.DataFrame:
    paths = sorted(glob.glob(f"result/coverage/{map_name}_s*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No metrics.csv found for map={map_name} under result/coverage/{map_name}_s*/")

    dfs = []
    for p in paths:
        try:
            seed = int(p.split(f"{map_name}_s")[1].split("/")[0])
        except Exception:
            raise ValueError(f"Could not parse seed from path: {p}")

        df = pd.read_csv(p)
        df["seed"] = seed
        df["map"] = map_name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


df = pd.concat([load("RBM"), load("RDM")], ignore_index=True)

# Seed-level summaries (avoid pseudo-replication)
seed_summary = (
    df.groupby(["map", "seed"])
      .agg(
          victims_mean=("victims_found", "mean"),
          victims_std=("victims_found", "std"),
          tfirst_median=("time_to_first_detection", "median"),
          tfirst_mean=("time_to_first_detection", "mean"),
          efirst_median=("energy_to_first_detection", "median"),
          energy_mean=("energy_used", "mean"),
          completion_mean=("completion_ratio", "mean"),
      )
      .reset_index()
)

# Ensure stable ordering for display/files
seed_summary = seed_summary.sort_values(["map", "seed"]).reset_index(drop=True)

# Final table: mean ± std across seeds
final = (
    seed_summary.groupby("map")
    .agg(
        victims_mean=("victims_mean", "mean"),
        victims_std=("victims_mean", "std"),
        tfirst_mean=("tfirst_mean", "mean"),
        tfirst_std=("tfirst_mean", "std"),
        tfirst_median_mean=("tfirst_median", "mean"),
        efirst_median_mean=("efirst_median", "mean"),
        energy_mean=("energy_mean", "mean"),
        completion_mean=("completion_mean", "mean"),
    )
    .reset_index()
)

final = final.sort_values(["map"]).reset_index(drop=True)

# ---- Balanced preview: 5 RBM + 5 RDM ----
rbm5 = seed_summary[seed_summary["map"] == "RBM"].head(5)
rdm5 = seed_summary[seed_summary["map"] == "RDM"].head(5)
preview = pd.concat([rbm5, rdm5], ignore_index=True)

print("\n=== Seed-level summary (5 RBM + 5 RDM) ===")
print(preview.to_string(index=False))

print("\n=== Final (mean across seeds ± std across seeds) ===")
print(final.to_string(index=False))

# Optional: export for paper table
seed_summary.to_csv("result/coverage/seed_summary.csv", index=False)
final.to_csv("result/coverage/final_summary.csv", index=False)
print("\nSaved: result/coverage/seed_summary.csv and final_summary.csv")