#!/usr/bin/env python3
"""
Summarise every QMIX run that follows the pattern:
   result/qmix/{MAP}_{variant}_s{seed}/global_data.npy

Where
   MAP      ∈ {RBM, RDM, ...}
   variant  ∈ {qm, mod, fed}
   seed     ∈ 1,2,3 …

Prints shape / mean / min / max and first/last 5 points.
With --plot it overlays every curve in one figure.

Author: ChatGPT · 2025-07-15
"""
from pathlib import Path
import re
import argparse
import numpy as np
import textwrap

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None   # plotting is optional

ROOT = Path('result/qmix')
PAT  = re.compile(r'^(?P<map>[A-Za-z]+)_(?P<var>qm|mod|fed)_s(?P<seed>\d+)$')

def short(b: bool) -> str:
    return "T" if b else "F"

def describe(arr: np.ndarray, width=4):
    prefix = " " * width
    print(f"{prefix}shape={arr.shape}, mean={arr.mean():.3f}, "
          f"min={arr.min():.2f}, max={arr.max():.2f}")
    print(textwrap.indent(f"first5: {arr[:5]}", prefix))
    print(textwrap.indent(f" last5: {arr[-5:]}", prefix))

def collect_runs():
    """Yield (map, variant, seed, path) for every matching folder."""
    for folder in ROOT.iterdir():
        if not folder.is_dir():
            continue
        m = PAT.match(folder.name)
        if not m:
            continue
        meta = m.groupdict()
        gpath = folder / 'global_data.npy'
        if gpath.exists():
            yield meta['map'], meta['var'], int(meta['seed']), gpath

def main(plot=False):
    runs = sorted(collect_runs(), key=lambda x: (x[0], x[1], x[2]))  # nice order
    if not runs:
        print("No QMIX result folders found under", ROOT)
        return

    print("\n=== QMIX result summary ===\n")
    current_map = None
    curves = []          # (label, x, y) for plotting

    for mp, var, seed, path in runs:
        if mp != current_map:
            print(f"── {mp} ──")
            current_map = mp

        tag = f"{mp}_{var}_s{seed}"
        try:
            data = np.load(path)
        except Exception as e:
            print(f"❌ {tag:15}  ERROR loading {path}: {e}")
            continue

        # print stats
        print(f"✅ {tag:15}")
        describe(data)

        if plot and plt:
            x = np.arange(1, len(data)+1)
            curves.append( (tag, x, data) )

    # plot if requested
    if plot and plt and curves:
        plt.figure(figsize=(8,6))
        for lbl, x, y in curves:
            plt.plot(x, y, label=lbl, linewidth=1)
        plt.xscale('log')
        plt.xlabel("Episode")
        plt.ylabel("Global data")
        plt.title("QMIX variants – all runs")
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
    elif plot and not plt:
        print("\n[plot] matplotlib not installed – skipping figure.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Summarise QMIX result curves")
    ap.add_argument('-p', '--plot', action='store_true',
                    help="Show a matplotlib overlay of all curves")
    args = ap.parse_args()
    main(plot=args.plot)