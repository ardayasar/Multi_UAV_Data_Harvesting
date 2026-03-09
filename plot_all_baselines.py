"""
plot_results.py  –  compare MARL variants on RBM / RDM
------------------------------------------------------

Baselines:
• IQL   :  result/iql/<MAP>_finalrun_s*/global_data.npy
• QMIX  :  result/qmix/<MAP>_qm_s*/  global_data.npy
• MOD   :  result/qmix/<MAP>_mod_s*/ global_data.npy
• FED   :  result/qmix/<MAP>_fed_s*/ global_data.npy
• IPPO  :  ippo_eval_<MAP>_s*.npy   (from ippo_custom.py)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

# ───────────────────────── tweak here ──────────────────────────
MAP   = "RDM"           # ← "RBM" or "RDM"
SEEDS = [1, 2, 3]       # seeds you actually ran
WINDOW = 5              # moving-average window
# ───────────────────────────────────────────────────────────────

COLORS = {
    "IQL" : "#FF8200",
    "QMIX": "g",
    "MOD" : "#0082FF",
    "FED" : "r",
    "IPPO": "m",        # magenta for IPPO
}
LABELS = {
    "IQL" : "IQL (model-free)",
    "QMIX": "QMIX (model-free)",
    "MOD" : "Model-aided QMIX",
    "FED" : "Model-aided FedQMIX",
    "IPPO": "IPPO",
}
LN_STYLE = {
    "IQL": ":",
    "QMIX": "-.",
    "MOD" : "--",
    "FED" : "-",
    "IPPO": "-",        # solid line for IPPO
}

BASE_DIRS = {
    "IQL" : f"result/iql/{MAP}_finalrun_s{{}}/global_data.npy",
    "QMIX": f"result/qmix/{MAP}_qm_s{{}}/global_data.npy",
    "MOD" : f"result/qmix/{MAP}_mod_s{{}}/global_data.npy",
    "FED" : f"result/qmix/{MAP}_fed_s{{}}/global_data.npy",
    # IPPO files are in the project root:
    #   ./ippo_eval_RBM_s1.npy, ./ippo_eval_RBM_s2.npy, ...
    "IPPO": f"ippo_eval_{MAP}_s{{}}.npy",
}


def load_variant(variant: str) -> List[np.ndarray]:
    """
    Load all seed runs for one variant (IQL / QMIX / MOD / FED).
    Each global_data.npy is assumed to be a 1D array of evaluation scores.
    """
    runs = []
    for sd in SEEDS:
        f = Path(BASE_DIRS[variant].format(sd))
        if not f.exists():
            print(f"⚠️  missing: {f}")
            continue
        runs.append(np.load(f))
    if not runs:
        raise FileNotFoundError(f"No data found for {variant}")
    return runs


def moving_avg(arr: np.ndarray, k: int = WINDOW) -> np.ndarray:
    if len(arr) < k:
        return arr
    return np.convolve(arr, np.ones(k) / k, mode="valid")


def plot_variant(ax, variant: str):
    """Standard MARL baselines that store global_data.npy."""
    runs = load_variant(variant)  # list of 1D arrays
    # pad to same length...
    max_len = max(len(r) for r in runs)
    padded = [np.pad(r, (0, max_len - len(r)), "edge") for r in runs]
    arr = np.vstack(padded)  # [n_seeds, T]
    y_raw = arr.mean(0)

    mean = moving_avg(y_raw, WINDOW)

    eval_cycle = 10                        # you evaluate every 10 episodes
    x_raw = np.arange(len(y_raw)) * eval_cycle
    x_axis = moving_avg(x_raw, WINDOW)     # same smoothing on x
    if len(x_axis) > 0:
        x_axis[0] = 1.0                    # force starting at 1 for log-scale

    ax.plot(
        x_axis,
        mean,
        linestyle=LN_STYLE[variant],
        label=LABELS[variant],
        lw=3,
        color=COLORS[variant],
    )


def plot_ippo(ax):
    """
    IPPO: ippo_eval_<MAP>_sX.npy has shape [N, 2]:
        column 0: episode number (10, 20, ..., 1000)
        column 1: mean victims (or collected data) at that eval.
    We average across seeds and apply the same moving-average smoothing.
    """
    runs = []
    episodes_ref = None

    for sd in SEEDS:
        f = Path(BASE_DIRS["IPPO"].format(sd))
        if not f.exists():
            print(f"⚠️  missing IPPO file: {f}")
            continue
        arr = np.load(f)      # shape [N, 2]
        eps, vals = arr[:, 0], arr[:, 1]
        if episodes_ref is None:
            episodes_ref = eps
        else:
            if not np.array_equal(episodes_ref, eps):
                print(f"⚠️  IPPO episodes differ for seed {sd}")
        runs.append(vals)

    if not runs:
        print("⚠️  No IPPO runs found, skipping IPPO curve.")
        return

    arr = np.vstack(runs)       # [n_seeds, N]
    y_raw = arr.mean(0)         # mean victims/score
    x_raw = episodes_ref        # episode numbers (10, 20, ..., 1000)

    mean = moving_avg(y_raw, WINDOW)
    x_axis = moving_avg(x_raw, WINDOW)
    if len(x_axis) > 0:
        x_axis[0] = 1.0

    ax.plot(
        x_axis,
        mean,
        linestyle=LN_STYLE["IPPO"],
        label=LABELS["IPPO"],
        lw=3,
        color=COLORS["IPPO"],
    )


def main():
    fig, ax = plt.subplots(figsize=(8, 6))

    # existing four baselines
    for var in ["FED", "MOD", "QMIX", "IQL"]:
        try:
            plot_variant(ax, var)
        except FileNotFoundError as e:
            print(e)

    # IPPO curve
    try:
        plot_ippo(ax)
    except FileNotFoundError as e:
        print(e)

    ax.set_xscale("log")
    ax.set_xlim(1, 1e3)   # 1000 episodes
    ax.set_xlabel("Episode [log scale]", fontsize=14)
    ax.set_ylabel("Total collected data", fontsize=14)  # or "Number of distinct victims localised"
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=12)
    fig.tight_layout()

    out_dir = Path("Figures")
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{MAP}_compare_with_ippo.pdf")
    fig.savefig(out_dir / f"{MAP}_compare_with_ippo.png", dpi=300)
    print(f"✅  Plot saved to {out_dir}/")


if __name__ == "__main__":
    main()