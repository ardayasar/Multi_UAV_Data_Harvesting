# better_bar_rdm_v2.py  ────────────────────────────────────────────
"""
Generates the RDM convergence-time bar chart in the red-bar style.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────── configuration ─────────────── #
MAP           = "RDM"
SEEDS         = [1, 2, 3]
EVAL_CYCLE    = 10          # evaluation every N episodes
THRESHOLD_VAL = 8.0         # performance threshold

METHODS = {
    "IQL": ("result/iql/{map}_finalrun_s{seed}/global_data.npy",
            "IQL (model-free)"),
    "QMIX": ("result/qmix/{map}_qm_s{seed}/global_data.npy",
             "QMIX (model-free)"),
    "MOD": ("result/qmix/{map}_mod_s{seed}/global_data.npy",
            "Model-aided QMIX"),
    "FED": ("result/qmix/{map}_fed_s{seed}/global_data.npy",
            "Model-aided FedQMIX"),
}

HERE = Path(__file__).resolve().parent


# ─────────────── helpers ─────────────── #
def load_np(tpl: str, map_name: str, seed: int) -> np.ndarray:
    """Load <global_data.npy> produced by each training run."""
    path = HERE / tpl.format(map=map_name, seed=seed)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)


def episodes_to_threshold(arr: np.ndarray, thr: float) -> float:
    """First eval-index whose metric ≥ thr → episode count; NaN if never."""
    idx = np.where(arr >= thr)[0]
    return np.nan if idx.size == 0 else (idx[0] + 1) * EVAL_CYCLE


# ───────────── data aggregation ───────────── #
runs = {m: [] for m in METHODS}                       # raw per-seed numbers
for m, (tpl, _) in METHODS.items():
    for sd in SEEDS:
        try:
            g = load_np(tpl, MAP, sd)
            runs[m].append(episodes_to_threshold(g, THRESHOLD_VAL))
        except FileNotFoundError:
            continue

stats = {
    m: {
        "mean":   np.nanmean(vals),
        "std":    np.nanstd(vals),
        "median": np.nanmedian(vals),
        "succ":   np.count_nonzero(~np.isnan(vals)),
    }
    for m, vals in runs.items()
}

# ───────────── plotting ───────────── #
plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(5.2, 3.0))

x      = np.arange(len(METHODS))
width  = 0.22
means  = [stats[m]["mean"] for m in METHODS]
errs   = [max(stats[m]["std"], 1e-3) for m in METHODS]  # avoid zero length

bars = ax.bar(
    x, means, width=width,
    color="tab:red", edgecolor="black",
    yerr=errs, ecolor="black", capsize=4, linewidth=0.8,
)

# annotate: median + success-count
for idx, m in enumerate(METHODS):
    md  = int(stats[m]["median"])
    sc  = stats[m]["succ"]
    y_text = means[idx] + errs[idx] + 2          # 2-episode offset
    ax.text(
        idx, y_text,
        f"{md}\n({sc}/3)",
        ha="center", va="bottom", fontsize=8
    )

# style
ax.set_xticks(x)
ax.set_xticklabels([METHODS[m][1] for m in METHODS],
                   rotation=20, ha="right")
ax.set_ylabel(f"Episodes to reach ≥{THRESHOLD_VAL}")
ax.set_title(f"Convergence time on {MAP}")
ax.set_ylim(0, (max(means) + max(errs)) * 1.25)

ax.yaxis.grid(True, ls="--", color="grey", alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

out_dir = HERE / "Figures"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"{MAP.lower()}_convergence_v2.png"
fig.savefig(out_path, dpi=300)
plt.show()
print("✅ Saved refined RDM plot →", out_path)