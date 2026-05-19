"""
run_partial_fedavg.py — Reviewer 2, Item 7 (packet-loss / partial-client FedAvg)

Trains Model-Aided FedQMIX with different worker-drop probabilities to show how
robustly the federated aggregation performs under packet loss / intermittent
connectivity.

drop_prob=0.0  → all workers always participate (baseline)
drop_prob=0.3  → each worker is dropped with 30% probability per round
drop_prob=0.5  → each worker is dropped with 50% probability per round

Usage:
    python run_partial_fedavg.py --map RBM --total_episodes 3000 --seed 1
    python run_partial_fedavg.py --map RDM --total_episodes 3000 --seed 1

Results are written to result/sensitivity_partial_fedavg_{map}.csv.
"""

import copy
import csv
import os
import random

import numpy as np
import torch

from common.arguments import get_common_args, get_mixer_args
from training_procedures import run_training

# ── drop probabilities to sweep ───────────────────────────────────────────────
DROP_PROBS = [0.0, 0.1, 0.3, 0.5, 0.7]


def _run_one(base_args, params, drop_prob):
    """Train FedQMIX with a given drop probability and return final eval score."""
    args = copy.deepcopy(base_args)
    args.drop_prob = drop_prob
    args.tag = f"_pfed_drop{int(drop_prob*100)}_s{args.seed}"

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

    # Average the last 3 eval checkpoints for stability
    last_rows = rows[-3:]
    vmeans = [float(r.get("victims_localised_mean", 0.0)) for r in last_rows]
    vstds  = [float(r.get("victims_localised_std",  0.0)) for r in last_rows]
    return float(np.mean(vmeans)), float(np.mean(vstds))


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
    args.sar_reward = True

    out_path = os.path.join(args.result_dir, f"sensitivity_partial_fedavg_{args.map}.csv")
    os.makedirs(args.result_dir, exist_ok=True)

    results = []
    for drop_prob in DROP_PROBS:
        print(f"\n[partial-fedavg] drop_prob={drop_prob:.1f}  map={args.map}  seed={args.seed}")
        vmean, vstd = _run_one(args, params, drop_prob)
        results.append({
            "map": args.map,
            "seed": args.seed,
            "drop_prob": drop_prob,
            "effective_clients_expected": f"{args.workers * (1 - drop_prob):.1f}",
            "victims_mean": vmean if vmean is not None else "",
            "victims_std":  vstd  if vstd  is not None else "",
        })
        print(f"  victims: {vmean:.2f} ± {vstd:.2f}" if vmean is not None
              else "  (no result — check paths)")

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"\n[partial-fedavg] Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
