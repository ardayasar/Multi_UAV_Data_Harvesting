"""
run_channel_shift.py — Reviewer 1, Item 5 (digital-twin / channel-shift robustness)

Trains a Model-Aided FedQMIX policy with the nominal channel, then evaluates it
under a range of additive SNR offsets to measure train/test channel mismatch.

Negative eval_snr_offset_db → degraded channel (harder test).
Positive eval_snr_offset_db → improved channel (easier, for sanity check).

Usage:
    python run_channel_shift.py --map RBM --total_episodes 3000 --seed 1
    python run_channel_shift.py --map RDM --total_episodes 3000 --seed 1

Results are written to result/sensitivity_channel_shift_{map}.csv.
"""

import copy
import csv
import os
import random

import numpy as np
import torch

from common.arguments import get_common_args, get_mixer_args
from training_procedures import _train_fedqmix, _fill_env_info_into_args, _set_seeds
from runner import Runner
from Libs.Environments.DataCollection import DataCollection
from Libs.ChannelEstimator import SLAL

# ── SNR offsets to sweep (dB) ─────────────────────────────────────────────────
SNR_OFFSETS_DB = [-20.0, -15.0, -10.0, -5.0, 0.0]
# negative = worse channel at eval time; 0 = nominal (no shift)


def _train_nominal(args, params):
    """Run full FedQMIX training with nominal channel and return save_path."""
    args.alg = "qmix"
    args.federated = True
    args.model = True
    args.use_federation = True
    args.use_surrogate_channel = True
    args.tag = f"_chshift_train_s{args.seed}"

    _set_seeds(args.seed)
    _train_fedqmix(args, params)

    save_path = os.path.join(
        args.result_dir, "qmix", "fed_mod", args.map + args.tag
    )
    return save_path


def _evaluate_with_shift(args, params, snr_offset_db, n_episodes=20):
    """Create a fresh env+runner, apply SNR offset, and run evaluation."""
    eval_args = copy.deepcopy(args)
    eval_args.evaluate = True

    _set_seeds(eval_args.seed)

    # Build env with nominal channel
    env = DataCollection(eval_args, learning_channel_model=None, params=params)
    _fill_env_info_into_args(env, eval_args)

    # Apply channel shift at eval time
    env.snr_offset_db = snr_offset_db

    runner = Runner(env, eval_args)

    victims_list = []
    sr_list = []
    ttf_list = []
    for _ in range(n_episodes):
        _, _, _, total_detected, *_ = runner.rolloutWorker.generate_episode(
            evaluate=True, model=False
        )
        info = runner.rolloutWorker.last_info
        victims_list.append(info["victims_found"])
        sr_list.append(info["success_rate"])
        ttf_list.append(info["time_to_first_detection"])

    return {
        "victims_mean": float(np.mean(victims_list)),
        "victims_std":  float(np.std(victims_list, ddof=1)) if len(victims_list) > 1 else 0.0,
        "success_rate_mean": float(np.mean(sr_list)),
        "ttf_mean":     float(np.mean(ttf_list)),
    }


def main():
    args = get_common_args()
    args = get_mixer_args(args)

    if args.map == "RBM":
        from config.RBM_define import params
    elif args.map == "RDM":
        from config.RDM_define import params
    else:
        raise ValueError(f"Unknown map: {args.map}")

    # Step 1 — train with nominal channel
    print(f"\n[channel-shift] Training nominal FedQMIX on {args.map} (seed={args.seed}) ...")
    _train_nominal(args, params)

    # Step 2 — evaluate under each SNR offset
    out_path = os.path.join(args.result_dir, f"sensitivity_channel_shift_{args.map}.csv")
    os.makedirs(args.result_dir, exist_ok=True)

    results = []
    for offset_db in SNR_OFFSETS_DB:
        print(f"\n[channel-shift] Evaluating  snr_offset_db={offset_db:+.1f} dB ...")
        metrics = _evaluate_with_shift(args, params, snr_offset_db=offset_db)
        row = {
            "map": args.map,
            "seed": args.seed,
            "snr_offset_db": offset_db,
            **metrics,
        }
        results.append(row)
        print(f"  victims: {metrics['victims_mean']:.2f} ± {metrics['victims_std']:.2f}  "
              f"  success_rate: {metrics['success_rate_mean']:.3f}")

        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    print(f"\n[channel-shift] Done. Results saved to {out_path}")


if __name__ == "__main__":
    main()
