#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_coverage_figure.py
Creates ONE reviewer-friendly figure with TWO panels:
  (A) victims_found distribution (RBM vs RDM)
  (B) time_to_first_detection distribution (RBM vs RDM)

Reads:
  result/coverage/<MAP>_s*/metrics.csv

Writes:
  result/coverage/figs/coverage_boxplots.png
  result/coverage/figs/coverage_boxplots.pdf
"""

import glob
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _seed_from_path(p: str, map_name: str) -> int:
    # p like: result/coverage/RBM_s3/metrics.csv
    return int(p.split(f"{map_name}_s")[1].split("/")[0])


def load_metrics(map_name: str) -> pd.DataFrame:
    paths = sorted(glob.glob(f"result/coverage/{map_name}_s*/metrics.csv"))
    if not paths:
        raise FileNotFoundError(f"No metrics.csv found for map={map_name} under result/coverage/{map_name}_s*/")
    dfs = []
    for p in paths:
        seed = _seed_from_path(p, map_name)
        df = pd.read_csv(p)
        df["seed"] = seed
        df["map"] = map_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def pick_seeds(df: pd.DataFrame, map_name: str, n_seeds: int) -> list[int]:
    seeds = sorted(df.loc[df["map"] == map_name, "seed"].unique().tolist())
    return seeds[:min(n_seeds, len(seeds))]


def summarize_for_annotation(df: pd.DataFrame, metric: str):
    x = df[metric].to_numpy(dtype=float)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)) if len(x) else float("nan"),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)) if len(x) else float("nan"),
    }


def add_boxplot(ax, data_rbm, data_rdm, ylabel, title):
    # Matplotlib boxplot (no seaborn)
    bp = ax.boxplot(
        [data_rbm, data_rdm],
        labels=["RBM", "RDM"],
        showmeans=False,
        showfliers=True,
        whis=1.5,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    return bp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5, help="Use first N seeds per map (default: 5).")
    parser.add_argument("--out_dir", type=str, default="result/coverage/figs")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    df = pd.concat([load_metrics("RBM"), load_metrics("RDM")], ignore_index=True)

    # keep only first N seeds per map (smallest seed IDs)
    rbm_seeds = pick_seeds(df, "RBM", args.n_seeds)
    rdm_seeds = pick_seeds(df, "RDM", args.n_seeds)

    df_use = df[
        ((df["map"] == "RBM") & (df["seed"].isin(rbm_seeds))) |
        ((df["map"] == "RDM") & (df["seed"].isin(rdm_seeds)))
    ].copy()

    # Distributions across ALL episodes pooled (this is what reviewers “see” as variability)
    rbm_v = df_use.query("map == 'RBM'")["victims_found"].to_numpy(dtype=float)
    rdm_v = df_use.query("map == 'RDM'")["victims_found"].to_numpy(dtype=float)

    rbm_t = df_use.query("map == 'RBM'")["time_to_first_detection"].to_numpy(dtype=float)
    rdm_t = df_use.query("map == 'RDM'")["time_to_first_detection"].to_numpy(dtype=float)

    # For annotation
    s_rbm_v = summarize_for_annotation(df_use.query("map=='RBM'"), "victims_found")
    s_rdm_v = summarize_for_annotation(df_use.query("map=='RDM'"), "victims_found")
    s_rbm_t = summarize_for_annotation(df_use.query("map=='RBM'"), "time_to_first_detection")
    s_rdm_t = summarize_for_annotation(df_use.query("map=='RDM'"), "time_to_first_detection")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "coverage_boxplots.png"
    out_pdf = out_dir / "coverage_boxplots.pdf"

    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(1, 2, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    add_boxplot(
        ax1,
        rbm_v,
        rdm_v,
        ylabel="Victims found (per episode)",
        title=f"(A) Coverage effectiveness\n(first {len(rbm_seeds)} seeds RBM, {len(rdm_seeds)} seeds RDM)",
    )
    ax1.text(
        0.02, 0.02,
        f"RBM: n={s_rbm_v['n']}, mean={s_rbm_v['mean']:.2f}, std={s_rbm_v['std']:.2f}, med={s_rbm_v['median']:.2f}\n"
        f"RDM: n={s_rdm_v['n']}, mean={s_rdm_v['mean']:.2f}, std={s_rdm_v['std']:.2f}, med={s_rdm_v['median']:.2f}",
        transform=ax1.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    ax2 = fig.add_subplot(gs[0, 1])
    add_boxplot(
        ax2,
        rbm_t,
        rdm_t,
        ylabel="t_first (steps)",
        title="(B) Responsiveness (time-to-first-detection)",
    )
    ax2.text(
        0.02, 0.02,
        f"RBM: n={s_rbm_t['n']}, mean={s_rbm_t['mean']:.2f}, std={s_rbm_t['std']:.2f}, med={s_rbm_t['median']:.2f}\n"
        f"RDM: n={s_rdm_t['n']}, mean={s_rdm_t['mean']:.2f}, std={s_rdm_t['std']:.2f}, med={s_rdm_t['median']:.2f}",
        transform=ax2.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    fig.suptitle("Coverage heuristic baseline: RBM vs RDM (distribution across episodes pooled over seeds)", y=1.02)
    fig.tight_layout()

    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print("✅ Saved figure:")
    print(f"  - {out_png}")
    print(f"  - {out_pdf}")


if __name__ == "__main__":
    main()