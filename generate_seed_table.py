#!/usr/bin/env python3
"""
Generate a seed evaluation table matching the style of Table 3 in the paper.

Rows:    Method-s{seed}  (e.g. QMIX-s1, QMIX-s2, ...)
Columns: Eval-5, Eval-4, Eval-3, Eval-2, Eval-1  (last 5 evaluations)
Values:  victims_found_mean (rounded to 1 decimal)
Bold:    highest value per column within each method group

Outputs:
  result/seed_table_{map}.csv
  result/seed_table_{map}.tex
"""

import os
import glob
import numpy as np
import pandas as pd

RESULT_ROOT = "result"
N_LAST = 5          # number of last evaluations to show
MAP_FILTER = None   # set to "RBM" or "RDM" to filter, None = all maps

# Method label mapping: (alg, model, federated) -> display name
METHOD_LABELS = {
    ("iql",  0, 0): "IQL",
    ("qmix", 0, 0): "QMIX",
    ("qmix", 1, 0): "MOD",
    ("qmix", 0, 1): "FED",
    ("qmix", 1, 1): "FED-MOD",
    ("ippo", 0, 0): "IPPO",
}

def method_label(alg, model, federated):
    key = (alg.lower(), int(model), int(federated))
    return METHOD_LABELS.get(key, f"{alg.upper()}_m{model}_f{federated}")


def load_data(result_root=RESULT_ROOT):
    paths = glob.glob(os.path.join(result_root, "*", "*", "eval_metrics.csv"))
    if not paths:
        raise RuntimeError(f"No eval_metrics.csv files found under '{result_root}'")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_table(df, map_name):
    df_map = df[df["map"] == map_name].copy()

    col_names = [f"Eval-{i}" for i in range(N_LAST, 0, -1)]  # Eval-5 ... Eval-1

    rows = []
    # Group by method + seed
    group_keys = ["alg", "model", "federated", "seed"]
    for key, sub in df_map.groupby(group_keys):
        alg, model, federated, seed = key
        label = f"{method_label(alg, model, federated)}-s{seed}"

        # Sort by eval_index, take last N_LAST
        sub_sorted = sub.sort_values("eval_index")
        last_n = sub_sorted["victims_found_mean"].values[-N_LAST:]

        # Pad with NaN if fewer than N_LAST evaluations exist
        if len(last_n) < N_LAST:
            pad = np.full(N_LAST - len(last_n), np.nan)
            last_n = np.concatenate([pad, last_n])

        row = {"Method-Seed": label, "alg": alg, "model": int(model),
               "federated": int(federated), "seed": int(seed)}
        for col, val in zip(col_names, last_n):
            row[col] = round(float(val), 1) if not np.isnan(val) else np.nan
        rows.append(row)

    tbl = pd.DataFrame(rows)
    tbl = tbl.sort_values(["alg", "model", "federated", "seed"]).reset_index(drop=True)
    return tbl, col_names


def to_latex(tbl, col_names, map_name):
    """Render LaTeX table with bold max per column within each method group."""
    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{\\textbf{{Table}}: {map_name} evaluation scores.}}")

    ncols = 1 + len(col_names)
    col_spec = "l" + "r" * len(col_names)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    lines.append("Method-Seed & " + " & ".join(col_names) + r" \\")
    lines.append(r"\midrule")

    # Group by method for bold and separator lines
    method_col = tbl.apply(
        lambda r: method_label(r["alg"], r["model"], r["federated"]), axis=1
    )
    tbl = tbl.copy()
    tbl["_method"] = method_col

    prev_method = None
    for _, row in tbl.iterrows():
        m = row["_method"]
        if prev_method is not None and m != prev_method:
            lines.append(r"\midrule")
        prev_method = m

        # Find max per column within this method group
        group = tbl[tbl["_method"] == m]
        cells = [row["Method-Seed"]]
        for col in col_names:
            val = row[col]
            if np.isnan(val):
                cells.append("--")
            else:
                col_max = group[col].max()
                fmt = f"{val:.1f}"
                if val == col_max:
                    fmt = f"\\textbf{{{fmt}}}"
                cells.append(fmt)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    df = load_data()
    maps = df["map"].unique() if MAP_FILTER is None else [MAP_FILTER]

    os.makedirs(RESULT_ROOT, exist_ok=True)

    for map_name in sorted(maps):
        tbl, col_names = build_table(df, map_name)

        # Save CSV
        csv_cols = ["Method-Seed"] + col_names
        csv_path = os.path.join(RESULT_ROOT, f"seed_table_{map_name}.csv")
        tbl[csv_cols].to_csv(csv_path, index=False)
        print(f"[seed_table] Saved CSV → {csv_path}")

        # Print to console
        print(f"\n{'='*55}")
        print(f"  Table: {map_name} evaluation scores (last {N_LAST} evals)")
        print(f"{'='*55}")
        print(tbl[csv_cols].to_string(index=False))

        # Save LaTeX
        tex = to_latex(tbl, col_names, map_name)
        tex_path = os.path.join(RESULT_ROOT, f"seed_table_{map_name}.tex")
        with open(tex_path, "w") as f:
            f.write(tex)
        print(f"\n[seed_table] Saved LaTeX → {tex_path}")


if __name__ == "__main__":
    main()
