# better_bar_rbm.py -----------------------------------------------------------
"""
Re-creates the RBM convergence-time plot using a visual style that
matches the reference ‘Distance-to-Target’ bar chart you provided
– solid red bars, black error bars, dashed grey grid lines, linear y-axis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ────────────────────────── CONFIG ────────────────────────── #
MAP           = "RBM"
SEEDS         = [1, 2, 3]
EVAL_CYCLE    = 10
THRESHOLD_VAL = 8.0

# (path_template, pretty_label)
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


# ───────────────────── utility helpers ───────────────────── #
def load_global(path_tpl, map_name, seed):
    fpath = HERE / path_tpl.format(map=map_name, seed=seed)
    if not fpath.exists():
        raise FileNotFoundError(f"Missing file {fpath}")
    return np.load(fpath)


def episodes_to_thresh(arr, thr=THRESHOLD_VAL):
    hit = np.where(arr >= thr)[0]
    return np.nan if hit.size == 0 else (hit[0] + 1) * EVAL_CYCLE


# ──────────────────── collect statistics ─────────────────── #
data = {m: [] for m in METHODS}

for m, (tpl, _) in METHODS.items():
    for s in SEEDS:
        try:
            g = load_global(tpl, MAP, s)
        except FileNotFoundError:
            continue
        data[m].append(episodes_to_thresh(g))

stats = {
    m: {
        "mean": np.nanmean(v),
        "std": np.nanstd(v),
        "median": np.nanmedian(v),
        "succ": np.count_nonzero(~np.isnan(v)),
    }
    for m, v in data.items()
}

# ───────────────────────── plotting ───────────────────────── #
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
    }
)

fig, ax = plt.subplots(figsize=(5.2, 3.2))

labels = [METHODS[m][1] for m in METHODS]
x = np.arange(len(labels))

bar_w = 0.20                      # ← narrower bar
means = [stats[m]["mean"] for m in METHODS]
errs  = [stats[m]["std"]  for m in METHODS]

ax.bar(
    x,
    means,
    width=bar_w,
    color="tab:red",
    edgecolor="black",
    yerr=errs,
    ecolor="black",
    capsize=5,                    # slightly larger caps
    linewidth=0.8,
)

# annotations
for idx, m in enumerate(METHODS):
    md = int(stats[m]["median"])
    sc = stats[m]["succ"]
    top = means[idx] + errs[idx]
    ax.text(
        idx,
        top + 0.06 * top,         # a bit more offset
        f"{md}\n({sc}/3)",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# styling
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel(f"Episodes to reach ≥{THRESHOLD_VAL}")
ax.set_title(f"Convergence time on {MAP}")

# grid styling: major & minor
ax.yaxis.grid(True, which="major", linestyle="--", color="grey", alpha=0.7)
ax.yaxis.grid(True, which="minor", linestyle=":",  color="lightgrey", alpha=0.5)
ax.set_axisbelow(True)

# remove top/right frame for cleaner look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylim(0, max(means) * 1.25)
ax.margins(x=0.08)                # extra horizontal breathing space

fig.tight_layout()
out_dir = HERE / "Figures"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / f"rbm_convergence_linear_v2.png", dpi=300)
plt.show()

print("✅ Saved refined plot ->", out_dir / "rbm_convergence_linear_v2.png")