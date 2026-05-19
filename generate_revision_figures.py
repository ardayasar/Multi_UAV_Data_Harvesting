"""
generate_revision_figures.py
Produces LaTeX tables and matplotlib figures for the three revision experiments:
  1. Lambda sensitivity  (Reviewer 3, Item 4)
  2. Partial FedAvg      (Reviewer 2, Item 7)
  3. Channel shift       (Reviewer 1, Item 5)

Run from project root:
    python3 generate_revision_figures.py

Outputs land in result/revision_figures/
"""

import os
import csv

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = os.path.join("result", "revision_figures")
os.makedirs(OUT_DIR, exist_ok=True)

MAPS = ["RBM", "RDM"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Lambda sensitivity
# ──────────────────────────────────────────────────────────────────────────────

def make_lambda_tables():
    df = pd.concat([
        pd.read_csv("result/sensitivity_lambda_RBM.csv"),
        pd.read_csv("result/sensitivity_lambda_RDM.csv"),
    ])

    # ── heatmap figure ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, m in zip(axes, MAPS):
        sub = df[df["map"] == m]
        pivot = sub.pivot(index="lambda_thr", columns="lambda_new", values="victims_mean")
        im = ax.imshow(pivot.values, cmap="YlGn", vmin=7, vmax=10,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(r"$\lambda_{\mathrm{new}}$")
        ax.set_ylabel(r"$\lambda_{\mathrm{thr}}$")
        ax.set_title(f"{m} — Victims Localised (out of 10)")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.1f}",
                        ha="center", va="center", fontsize=8,
                        color="black" if pivot.values[i, j] > 8 else "white")
        fig.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lambda_sensitivity_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "lambda_sensitivity_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── LaTeX table ──
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reward-coefficient sensitivity: mean victims localised (out of 10) for",
        r"Model-Aided FedQMIX on RBM (top) and RDM (bottom), seed~1, 3000 episodes.}",
        r"\label{tab:lambda_sensitivity}",
        r"\small",
    ]

    for m in MAPS:
        sub = df[df["map"] == m]
        pivot = sub.pivot(index="lambda_thr", columns="lambda_new",
                          values="victims_mean")
        lnew_vals = list(pivot.columns)
        lthr_vals = list(pivot.index)

        col_str = "c" * len(lnew_vals)
        lines.append(r"\begin{tabular}{l" + col_str + "}")
        lines.append(r"\toprule")
        header = (r"$\lambda_{\mathrm{thr}}$ \textbackslash{} "
                  r"$\lambda_{\mathrm{new}}$")
        lines.append(header + " & " +
                     " & ".join(str(v) for v in lnew_vals) + r" \\")
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(len(lnew_vals) + 1) +
                     r"}{l}{\textit{" + m + r"}} \\")
        for lthr in lthr_vals:
            row_vals = pivot.loc[lthr]
            best = row_vals.max()
            cells = []
            for v in row_vals:
                s = f"{v:.1f}"
                if v == best:
                    s = r"\textbf{" + s + r"}"
                cells.append(s)
            lines.append(f"{lthr} & " + " & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        if m == "RBM":
            lines.append(r"\vspace{4pt}")

    lines += [r"\end{table}"]

    tex_path = os.path.join(OUT_DIR, "table_lambda_sensitivity.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[lambda] heatmap + LaTeX → {OUT_DIR}")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Partial FedAvg
# ──────────────────────────────────────────────────────────────────────────────

def make_partial_fedavg_tables():
    df = pd.concat([
        pd.read_csv("result/sensitivity_partial_fedavg_RBM.csv"),
        pd.read_csv("result/sensitivity_partial_fedavg_RDM.csv"),
    ])

    # ── line figure ──
    fig, ax = plt.subplots(figsize=(6, 3.5))
    markers = {"RBM": "o", "RDM": "s"}
    colors  = {"RBM": "#1f77b4", "RDM": "#ff7f0e"}
    for m in MAPS:
        sub = df[df["map"] == m].sort_values("drop_prob")
        ax.errorbar(sub["drop_prob"] * 100,
                    sub["victims_mean"],
                    yerr=sub["victims_std"],
                    marker=markers[m], color=colors[m],
                    label=m, capsize=4, linewidth=1.5)

    ax.set_xlabel("Worker drop probability (%)")
    ax.set_ylabel("Victims localised (mean ± std, out of 10)")
    ax.set_ylim(0, 11)
    ax.set_xticks([0, 10, 30, 50, 70])
    ax.axhline(10, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend()
    ax.set_title("FedQMIX robustness to worker packet loss")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "partial_fedavg_robustness.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "partial_fedavg_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── LaTeX table ──
    drop_probs = sorted(df["drop_prob"].unique())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{FedQMIX robustness under worker packet loss. Values show mean"
        r" $\pm$ std victims localised out of 10 (seed~1, 3000 episodes).}",
        r"\label{tab:partial_fedavg}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Map & $p_{\mathrm{drop}}=0.0$ & $0.1$ & $0.3$ & $0.5$ & $0.7$ \\",
        r"\midrule",
    ]
    for m in MAPS:
        sub = df[df["map"] == m].sort_values("drop_prob")
        cells = []
        for _, row in sub.iterrows():
            cells.append(f"{row['victims_mean']:.2f}$\\pm${row['victims_std']:.2f}")
        lines.append(m + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = os.path.join(OUT_DIR, "table_partial_fedavg.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[partial-fedavg] line plot + LaTeX → {OUT_DIR}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Channel shift
# ──────────────────────────────────────────────────────────────────────────────

def make_channel_shift_tables():
    df = pd.concat([
        pd.read_csv("result/sensitivity_channel_shift_RBM.csv"),
        pd.read_csv("result/sensitivity_channel_shift_RDM.csv"),
    ])

    # ── line figure ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, m in zip(axes, MAPS):
        sub = df[df["map"] == m].sort_values("snr_offset_db")
        ax.errorbar(sub["snr_offset_db"],
                    sub["victims_mean"],
                    yerr=sub["victims_std"],
                    marker="o", color="#2ca02c",
                    capsize=4, linewidth=1.5)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="Nominal (no shift)")
        ax.set_xlabel("SNR offset at evaluation (dB)")
        ax.set_ylabel("Victims localised (out of 10)")
        ax.set_ylim(-0.5, 11)
        ax.set_xticks(sorted(sub["snr_offset_db"].unique()))
        ax.set_title(f"{m} — Channel-shift robustness")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "channel_shift_robustness.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "channel_shift_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── LaTeX table ──
    offsets = sorted(df["snr_offset_db"].unique())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Digital-twin (SLAL) robustness to train/test channel mismatch.",
        r"Policy trained at nominal SNR; evaluated at SNR offsets up to $-20$\,dB.",
        r"Values: mean $\pm$ std victims localised out of 10 (seed~1).}",
        r"\label{tab:channel_shift}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Map & $-20$\,dB & $-15$\,dB & $-10$\,dB & $-5$\,dB & $0$\,dB (nominal) \\",
        r"\midrule",
    ]
    for m in MAPS:
        sub = df[df["map"] == m].sort_values("snr_offset_db")
        cells = []
        for _, row in sub.iterrows():
            cells.append(f"{row['victims_mean']:.2f}$\\pm${row['victims_std']:.2f}")
        lines.append(m + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = os.path.join(OUT_DIR, "table_channel_shift.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[channel-shift] line plot + LaTeX → {OUT_DIR}")


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_lambda_tables()
    make_partial_fedavg_tables()
    make_channel_shift_tables()
    print("\nDone. All outputs in:", OUT_DIR)
