#!/usr/bin/env python
import os
import glob
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


RESULT_ROOT = "result"


def _load_all_eval_metrics(result_root=RESULT_ROOT):
    """Load all eval_metrics.csv into a single DataFrame."""
    pattern = os.path.join(result_root, "*", "*", "eval_metrics.csv")
    paths = glob.glob(pattern)
    if not paths:
        raise RuntimeError(f"No eval_metrics.csv files found under {result_root}")

    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df["run_dir"] = os.path.dirname(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _select_last_eval_per_run(df):
    """For each run (map, alg, tag, seed, model, federated, run_dir),
    keep only the last evaluation row."""
    run_keys = ["map", "alg", "tag", "seed", "model", "federated", "run_dir"]
    idxs = []
    for _, sub in df.groupby(run_keys):
        idxs.append(sub["eval_index"].idxmax())
    return df.loc[idxs].reset_index(drop=True)


def _mean_ci(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n <= 1:
        return mean, 0.0
    std = float(arr.std(ddof=1))
    ci = 1.96 * std / np.sqrt(n)
    return mean, ci


def build_summary(df_last):
    """Build method-level summary with mean ± 95% CI for each metric."""
    method_keys = ["map", "alg", "model", "federated"]
    metrics = [
        "victims_found_mean",
        "time_to_first_detection_mean",
        "energy_used_mean",
        "energy_per_victim_mean",
    ]

    rows = []
    for key, sub in df_last.groupby(method_keys):
        row = dict(zip(method_keys, key))
        row["n_seeds"] = int(sub["seed"].nunique())

        for m in metrics:
            mean, ci = _mean_ci(sub[m].values)
            label = m.replace("_mean", "")
            row[label] = f"{mean:.2f} $\\pm$ {ci:.2f}"

        # bytes_per_round should be constant within a method; just take first
        row["bytes_per_round"] = int(sub["bytes_per_round"].iloc[0])
        rows.append(row)

    return pd.DataFrame(rows)


def maybe_wilcoxon(df_last, metric="victims_found_mean"):
    """Optional: print Wilcoxon signed-rank tests between methods per map."""
    if wilcoxon is None:
        print("[compute_stats] SciPy not installed; skipping Wilcoxon tests.")
        return

    method_id_cols = ["alg", "model", "federated"]
    for map_name, df_map in df_last.groupby("map"):
        # collect per-method series aligned by seed
        groups = {}
        for key, sub in df_map.groupby(method_id_cols):
            label = f"{key[0]}_m{int(key[1])}_f{int(key[2])}"
            # sort by seed for deterministic pairing
            groups[label] = sub.sort_values("seed")[metric].values

        labels = list(groups.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                x, y = groups[a], groups[b]
                if len(x) != len(y) or len(x) < 2:
                    continue
                stat, p = wilcoxon(x, y)
                print(f"[Wilcoxon] map={map_name}, {metric}: {a} vs {b} → p={p:.4f}")


def main():
    df = _load_all_eval_metrics()
    df_last = _select_last_eval_per_run(df)
    summary = build_summary(df_last)

    os.makedirs(RESULT_ROOT, exist_ok=True)
    out_csv = os.path.join(RESULT_ROOT, "metrics_summary_ci.csv")
    out_tex = os.path.join(RESULT_ROOT, "metrics_summary_ci.tex")

    summary.to_csv(out_csv, index=False)
    with open(out_tex, "w") as f:
        f.write(summary.to_latex(index=False, escape=False))

    print("[compute_stats] Saved CI summary to:", out_csv)
    print("[compute_stats] Saved LaTeX table to:", out_tex)

    # Optional significance tests
    maybe_wilcoxon(df_last, metric="victims_found_mean")


if __name__ == "__main__":
    main()