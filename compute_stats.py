#!/usr/bin/env python
import os
import glob
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon, mannwhitneyu
except Exception:
    wilcoxon = None
    mannwhitneyu = None


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


METRICS = [
    "victims_found_mean",
    "time_to_first_detection_mean",
    "energy_used_mean",
    "energy_per_victim_mean",
]

METHOD_ID_COLS = ["alg", "model", "federated"]

_SIG_STARS = {0.001: "***", 0.01: "**", 0.05: "*", 1.0: "ns"}


def _stars(p):
    for threshold, label in _SIG_STARS.items():
        if p < threshold:
            return label
    return "ns"


def significance_table(df_last):
    """Run Wilcoxon signed-rank and Mann-Whitney U tests across all metrics
    for every pair of methods, per map.  Returns a DataFrame of results."""
    if wilcoxon is None or mannwhitneyu is None:
        print("[compute_stats] SciPy not installed; skipping significance tests.")
        return pd.DataFrame()

    rows = []
    for map_name, df_map in df_last.groupby("map"):
        groups = {}
        for key, sub in df_map.groupby(METHOD_ID_COLS):
            label = f"{key[0]}_m{int(key[1])}_f{int(key[2])}"
            groups[label] = sub.sort_values("seed")

        labels = list(groups.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                sub_a, sub_b = groups[a], groups[b]

                for metric in METRICS:
                    x = sub_a[metric].values.astype(float)
                    y = sub_b[metric].values.astype(float)

                    if len(x) < 2 or len(y) < 2:
                        continue

                    # Wilcoxon (paired — requires same length)
                    if len(x) == len(y):
                        try:
                            _, p_wx = wilcoxon(x, y)
                        except Exception:
                            p_wx = float("nan")
                    else:
                        p_wx = float("nan")

                    # Mann-Whitney U (unpaired — works for any lengths)
                    try:
                        _, p_mw = mannwhitneyu(x, y, alternative="two-sided")
                    except Exception:
                        p_mw = float("nan")

                    rows.append({
                        "map": map_name,
                        "method_a": a,
                        "method_b": b,
                        "metric": metric,
                        "n_a": len(x),
                        "n_b": len(y),
                        "mean_a": float(x.mean()),
                        "mean_b": float(y.mean()),
                        "p_wilcoxon": round(p_wx, 4),
                        "sig_wilcoxon": _stars(p_wx),
                        "p_mannwhitney": round(p_mw, 4),
                        "sig_mannwhitney": _stars(p_mw),
                    })

    return pd.DataFrame(rows)


def maybe_wilcoxon(df_last, metric="victims_found_mean"):
    """Legacy helper: print Wilcoxon results for a single metric."""
    tbl = significance_table(df_last)
    if tbl.empty:
        return
    sub = tbl[tbl["metric"] == metric]
    for _, row in sub.iterrows():
        print(
            f"[Wilcoxon] map={row['map']}, {metric}: "
            f"{row['method_a']} vs {row['method_b']} → "
            f"p={row['p_wilcoxon']:.4f} {row['sig_wilcoxon']}"
        )


def main():
    df = _load_all_eval_metrics()
    df_last = _select_last_eval_per_run(df)
    summary = build_summary(df_last)

    os.makedirs(RESULT_ROOT, exist_ok=True)

    # --- Mean ± CI summary ---
    out_csv = os.path.join(RESULT_ROOT, "metrics_summary_ci.csv")
    out_tex = os.path.join(RESULT_ROOT, "metrics_summary_ci.tex")
    summary.to_csv(out_csv, index=False)
    with open(out_tex, "w") as f:
        f.write(summary.to_latex(index=False, escape=False))
    print("[compute_stats] Saved CI summary to:", out_csv)
    print("[compute_stats] Saved LaTeX table to:", out_tex)

    # --- Significance tests (all metrics) ---
    sig_tbl = significance_table(df_last)
    if not sig_tbl.empty:
        sig_csv = os.path.join(RESULT_ROOT, "significance_tests.csv")
        sig_tex = os.path.join(RESULT_ROOT, "significance_tests.tex")
        sig_tbl.to_csv(sig_csv, index=False)
        with open(sig_tex, "w") as f:
            f.write(sig_tbl.to_latex(index=False, escape=False))
        print("[compute_stats] Saved significance table to:", sig_csv)
        print("[compute_stats] Saved significance LaTeX to:", sig_tex)

        # Print summary to console
        print("\n=== Significance Tests (p < 0.05 results) ===")
        sig_only = sig_tbl[
            (sig_tbl["p_mannwhitney"] < 0.05) | (sig_tbl["p_wilcoxon"] < 0.05)
        ]
        if sig_only.empty:
            print("  No significant differences found (may need more seeds).")
        else:
            for _, row in sig_only.iterrows():
                print(
                    f"  map={row['map']} | {row['metric']:35s} | "
                    f"{row['method_a']} vs {row['method_b']} | "
                    f"MW p={row['p_mannwhitney']:.4f}{row['sig_mannwhitney']:>4s} | "
                    f"Wilcoxon p={row['p_wilcoxon']:.4f}{row['sig_wilcoxon']:>4s}"
                )


if __name__ == "__main__":
    main()