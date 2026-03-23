#!/usr/bin/env python3
"""
plot_learning_curves.py
Generate Fig.2 (RDM) and Fig.3 (RBM) style learning curves.
Usage: python3 plot_learning_curves.py
Outputs: result/fig2_RDM.png, result/fig3_RBM.png (+ .pdf)
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RESULT_ROOT = "result"

# ---------- visual style matching the reference figures ----------
METHOD_CFG = {
    # key        label                    color      ls     lw   zorder
    "FED":  ("Model-aided FedQMIX", "tab:red",     "-",   2.0, 5),
    "MOD":  ("Model-aided QMIX",    "#1E90FF",     "--",  2.0, 4),
    "QMIX": ("QMIX (model-free)",   "tab:green",   "-.",  1.8, 3),
    "IQL":  ("IQL (model-free)",    "tab:orange",  ":",   1.8, 2),
    "IPPO": ("IPPO",                "tab:purple",  "-",   1.8, 1),
}

OLD_EVAL_EPOCH = 19   # approximation: 30000 eps / 1580 evals ≈ 19
SMOOTH_WIN     = 20   # rolling-average window (in eval steps)


# ----------------------------------------------------------------
def _method_from_old(row):
    alg = str(row.get("alg", "")).lower()
    model     = int(float(row.get("model",     0)))
    federated = int(float(row.get("federated", 0)))
    if alg == "iql":
        return "IQL"
    if alg == "qmix":
        if federated:
            return "FED"
        if model:
            return "MOD"
        return "QMIX"
    return None


def _is_seed_dir(path):
    """Accept only paths whose immediate parent matches MAP_sN pattern."""
    parent = os.path.basename(os.path.dirname(path))
    return bool(re.search(r"_s\d+$", parent))


def load_all(map_name):
    paths = glob.glob(
        os.path.join(RESULT_ROOT, "**", "eval_metrics.csv"), recursive=True
    )

    records = []
    for p in paths:
        if not _is_seed_dir(p):
            continue
        try:
            df = pd.read_csv(p, on_bad_lines="skip")
        except Exception:
            continue

        # ---- new format (has "method" column) ----
        if "method" in df.columns and "train_episode" in df.columns:
            map_col = "map_name" if "map_name" in df.columns else "map"
            sub = df[df[map_col] == map_name].copy()
            if sub.empty:
                continue
            sub = sub[["method", "train_episode", "victims_localised_mean", "seed"]].copy()
            sub.columns = ["method", "episode", "victims", "seed"]
            records.append(sub.dropna(subset=["victims"]))

        # ---- old format (has "alg" and "eval_index") ----
        elif "alg" in df.columns and "eval_index" in df.columns:
            if "map" not in df.columns:
                continue
            sub = df[df["map"] == map_name].copy()
            if sub.empty:
                continue
            # Derive episode from eval_index
            max_idx = sub["eval_index"].max()
            step = 30000 / max_idx if max_idx > 0 else OLD_EVAL_EPOCH
            sub["episode"] = (sub["eval_index"] * step).astype(int)
            sub["method"]  = sub.apply(_method_from_old, axis=1)
            sub["victims"] = sub["victims_found_mean"]
            sub = sub[["method", "episode", "victims", "seed"]].dropna(subset=["method", "victims"])
            records.append(sub)

    if not records:
        return {}

    all_df = pd.concat(records, ignore_index=True)

    result = {}
    for m in METHOD_CFG:
        chunk = all_df[all_df["method"] == m]
        if not chunk.empty:
            result[m] = chunk
    return result


def smooth(series, window=SMOOTH_WIN):
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_map(map_name, fig_num):
    data = load_all(map_name)
    if not data:
        print(f"  No data for {map_name} — skipping.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for method, (label, color, ls, lw, zorder) in METHOD_CFG.items():
        if method not in data:
            print(f"  [{map_name}] Missing method: {method}")
            continue

        df = data[method]
        # Mean across seeds at each episode checkpoint
        grouped = (
            df.groupby("episode")["victims"]
            .mean()
            .reset_index()
            .sort_values("episode")
        )
        grouped["smoothed"] = smooth(grouped["victims"])

        # Shift episode 0 → 1 so log scale works
        x = grouped["episode"].values.copy()
        x[x < 1] = 1

        ax.plot(x, grouped["smoothed"].values,
                color=color, linestyle=ls, linewidth=lw,
                label=label, zorder=zorder)

    ax.set_xscale("log")
    ax.set_xlabel("Episode [log scale]", fontsize=12)
    ax.set_ylabel("Total collected data", fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_title(
        f"Fig. {fig_num}: Learning curves on the "
        + ("Return-Base Map (RBM)" if map_name == "RBM" else "Reach-Destination Map (RDM)"),
        fontsize=11,
    )

    os.makedirs(RESULT_ROOT, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(RESULT_ROOT, f"fig{fig_num}_{map_name}.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close()


def main():
    print("Generating Fig. 3 (RBM)...")
    plot_map("RBM", 3)
    print("Generating Fig. 2 (RDM)...")
    plot_map("RDM", 2)


if __name__ == "__main__":
    main()
