"""
run_lambda_sensitivity.py — Reviewer 3, Item 4
Sweep lambda_thr × lambda_new to show how the SAR reward coefficients affect
victim localisation performance (Model-Aided FedQMIX).

Usage:
    python run_lambda_sensitivity.py --map RBM --total_episodes 3000 --seed 1
    python run_lambda_sensitivity.py --map RDM --total_episodes 3000 --seed 1

Results are written to result/sensitivity_lambda_{map}.csv.
"""

import copy
import csv
import os
import random

import numpy as np
import torch

from common.arguments import get_common_args, get_mixer_args
from training_procedures import run_training

# ── grid ──────────────────────────────────────────────────────────────────────
LAMBDA_NEW_GRID = [0.25, 0.5, 1.0, 2.0, 5.0]
LAMBDA_THR_GRID = [0.5, 1.0, 2.0]


def _run_one(base_args, params, lnew, lthr):
    """Run one training configuration and return final eval score."""
    args = copy.deepcopy(base_args)
    args.lambda_new = lnew
    args.lambda_thr = lthr
    args.sar_reward = True
    args.tag = f"_lsens_lnew{lnew}_lthr{lthr}_s{args.seed}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_training(args, params)

    # Read the last eval score from the saved eval_metrics.csv
    csv_path = os.path.join(
        args.result_dir, "qmix", "fed", args.map + args.tag, "eval_metrics.csv"
    )
    if not os.path.exists(csv_path):
        return None, None

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None, None

    last = rows[-1]
    victims_mean = float(last.get("victims_localised_mean", 0.0))
    victims_std  = float(last.get("victims_localised_std",  0.0))
    return victims_mean, victims_std


def main():
    args = get_common_args()
    args = get_mixer_args(args)

    if args.map == "RBM":
        from config.RBM_define import params
    elif args.map == "RDM":
        from config.RDM_define import params
    else:
        raise ValueError(f"Unknown map: {args.map}")

    # Match the paper's FED configuration: FedQMIX without SLAL surrogate
    args.alg = "qmix"
    args.federated = True
    args.model = False
    args.use_federation = True
    args.use_surrogate_channel = False

    out_path = os.path.join(args.result_dir, f"sensitivity_lambda_{args.map}.csv")
    os.makedirs(args.result_dir, exist_ok=True)

    results = []
    total = len(LAMBDA_NEW_GRID) * len(LAMBDA_THR_GRID)
    done = 0
    for lthr in LAMBDA_THR_GRID:
        for lnew in LAMBDA_NEW_GRID:
            done += 1
            print(f"\n[lambda-sensitivity] {done}/{total}  "
                  f"lambda_thr={lthr}  lambda_new={lnew}  map={args.map}")
            vmean, vstd = _run_one(args, params, lnew, lthr)
            results.append({
                "map": args.map,
                "seed": args.seed,
                "lambda_thr": lthr,
                "lambda_new": lnew,
                "victims_mean": vmean if vmean is not None else "",
                "victims_std":  vstd  if vstd  is not None else "",
            })
            # Write incrementally so partial runs are not lost
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)

    print(f"\n[lambda-sensitivity] Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
