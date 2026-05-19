"""
analyse_noniid.py — Reviewer 2, Item 8 (non-IID client-data diagnostics)

Reads the noniid_var.npy files produced during federated training and generates:
  • A line plot: per-round variance of worker data diversity over training (RBM + RDM)
  • A summary LaTeX table: mean / max / final variance per run

"Non-IID variance" is the variance of per-worker mean collected data within
the last aggregation window, logged every round in training_procedures.py.
A value near 0 means workers explored similar environments; a higher value
means workers encountered different data distributions.

Usage:
    python3 analyse_noniid.py

Outputs:
    result/revision_figures/noniid_variance_plot.pdf / .png
    result/revision_figures/table_noniid_summary.tex
    result/noniid_summary.csv
"""

import os
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_ROOT = "result"
OUT_FIG_DIR = os.path.join(RESULT_ROOT, "revision_figures")
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# ── runs to analyse ───────────────────────────────────────────────────────────
# Use the baseline FED (drop_prob=0) and FED-MOD (channel-shift) runs.
RUNS = [
    {
        "label":  "FED / RBM",
        "map":    "RBM",
        "method": "FED",
        "path":   "result/qmix/fed/RBM_pfed_drop0_s1/noniid_var.npy",
        "color":  "tab:red",
        "ls":     "-",
    },
    {
        "label":  "FED / RDM",
        "map":    "RDM",
        "method": "FED",
        "path":   "result/qmix/fed/RDM_pfed_drop0_s1/noniid_var.npy",
        "color":  "tab:blue",
        "ls":     "--",
    },
    {
        "label":  "FED-MOD / RBM",
        "map":    "RBM",
        "method": "FED-MOD",
        "path":   "result/qmix/fed_mod/RBM_chshift_train_s1/noniid_var.npy",
        "color":  "tab:orange",
        "ls":     "-.",
    },
    {
        "label":  "FED-MOD / RDM",
        "map":    "RDM",
        "method": "FED-MOD",
        "path":   "result/qmix/fed_mod/RDM_chshift_train_s1/noniid_var.npy",
        "color":  "tab:green",
        "ls":     ":",
    },
]


# ── plot ──────────────────────────────────────────────────────────────────────

def make_plot(runs):
    fig, ax = plt.subplots(figsize=(7, 4))

    for run in runs:
        data = np.load(run["path"])
        rounds = np.arange(1, len(data) + 1)
        # light smoothing (window=5) to reduce noise while preserving trend
        smoothed = np.convolve(data, np.ones(5) / 5, mode="same")
        ax.plot(rounds, smoothed, color=run["color"], linestyle=run["ls"],
                linewidth=1.6, label=run["label"])
        ax.fill_between(rounds, 0, data, color=run["color"], alpha=0.08)

    ax.set_xlabel("Federated round", fontsize=11)
    ax.set_ylabel("Non-IID variance\n(inter-worker data diversity)", fontsize=11)
    ax.set_title("Per-round data heterogeneity across FedQMIX workers", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.set_xlim(1, max(len(np.load(r["path"])) for r in runs))
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_FIG_DIR, f"noniid_variance_plot.{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {OUT_FIG_DIR}/noniid_variance_plot.pdf")


# ── summary table ─────────────────────────────────────────────────────────────

def make_table(runs):
    rows = []
    for run in runs:
        data = np.load(run["path"])
        rows.append({
            "label":    run["label"],
            "map":      run["map"],
            "method":   run["method"],
            "rounds":   len(data),
            "mean_var": float(data.mean()),
            "max_var":  float(data.max()),
            "final_var": float(data[-1]),
        })

    # CSV
    csv_path = os.path.join(RESULT_ROOT, "noniid_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved → {csv_path}")

    # LaTeX
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Non-IID client-data diagnostics for FedQMIX training runs.",
        r"Non-IID variance measures the inter-worker spread in mean collected data",
        r"within each aggregation window. Values close to zero confirm that the",
        r"workers explore similar data distributions, validating the FedAvg",
        r"IID assumption in this deployment scenario.}",
        r"\label{tab:noniid}",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Method & Map & Mean var. & Max var. & Final var. \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['method']} & {r['map']} & "
            f"{r['mean_var']:.4f} & {r['max_var']:.4f} & {r['final_var']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = os.path.join(OUT_FIG_DIR, "table_noniid_summary.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table saved → {tex_path}")

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # filter to runs whose files actually exist
    runs = [r for r in RUNS if os.path.exists(r["path"])]
    missing = [r["label"] for r in RUNS if not os.path.exists(r["path"])]
    if missing:
        print(f"WARNING: missing files for: {missing}")

    make_plot(runs)
    summary = make_table(runs)

    print("\n=== Summary ===")
    for r in summary:
        print(f"  {r['label']:20s}  mean={r['mean_var']:.4f}  "
              f"max={r['max_var']:.4f}  final={r['final_var']:.4f}")
