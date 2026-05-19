"""
run_device_heterogeneity.py — Reviewer 2, Item 6
BLE / Smartphone hardware heterogeneity test (evaluation-only, no retraining).

For each victim device k a fixed per-episode bias is sampled once:
    ξ_k ~ N(0, σ_device²)
and added to the SNR observed by every UAV before contact detection
and PSO localisation.

Three settings are tested:
    σ_device = 0 dB  — homogeneous (baseline)
    σ_device = 3 dB  — moderate heterogeneity
    σ_device = 6 dB  — strong heterogeneity

The final trained Model-Aided FedQMIX policy is loaded from the
channel-shift training run (fed_mod, model=True, federated=True).

Usage:
    python3 run_device_heterogeneity.py --map RBM --seed 1
    python3 run_device_heterogeneity.py --map RDM --seed 1

Outputs:
    result/device_heterogeneity_evaluation.csv
    result/revision_figures/table_device_heterogeneity.tex
"""

import copy
import csv
import os
import random

import numpy as np
import torch

from common.arguments import get_common_args, get_mixer_args
from training_procedures import _fill_env_info_into_args, _set_seeds
from runner import Runner
from Libs.Environments.DataCollection import DataCollection

SIGMA_DEVICE_LIST = [0.0, 3.0, 6.0]   # dB
N_EVAL_EPISODES   = 20
N_BIAS_DRAWS      = 5   # repeat with different bias realisations → average


# ── model loading ─────────────────────────────────────────────────────────────

def _load_worker0_weights(runner, map_name, tag, model_dir="./model"):
    """Load worker-0 best weights saved by _train_fedqmix."""
    base = os.path.join(model_dir, "qmix", map_name + tag)
    rnn  = os.path.join(base, "0best_rnn_net_params.pkl")
    qmix = os.path.join(base, "0best_qmix_net_params.pkl")
    if os.path.exists(rnn) and os.path.exists(qmix):
        runner.agents.policy.eval_rnn.load_state_dict(
            torch.load(rnn, map_location="cpu"))
        runner.agents.policy.eval_qmix_net.load_state_dict(
            torch.load(qmix, map_location="cpu"))
        print(f"    Loaded weights from {base}")
        return True
    print(f"    WARNING: weights not found at {base} — using uninitialised policy")
    return False


# ── single σ evaluation ───────────────────────────────────────────────────────

def _evaluate_sigma(args, params, model_tag, sigma_device):
    """
    Evaluate the trained policy under per-device bias σ_device.
    Repeats N_BIAS_DRAWS times with fresh bias vectors and averages.
    """
    eval_args = copy.deepcopy(args)
    eval_args.evaluate = True

    all_victims, all_sr, all_energy = [], [], []

    for draw in range(N_BIAS_DRAWS):
        _set_seeds(eval_args.seed + draw * 100)   # different seed per draw

        env = DataCollection(eval_args, learning_channel_model=None, params=params)
        _fill_env_info_into_args(env, eval_args)

        # Sample fixed per-device biases for this draw
        if sigma_device > 0.0:
            env.device_rssi_bias = np.random.normal(
                0.0, sigma_device, size=env.n_devices).astype(np.float32)
        # else: device_rssi_bias stays as np.zeros (set in __init__)

        runner = Runner(env, eval_args)
        _load_worker0_weights(runner, eval_args.map, model_tag)

        for ep in range(N_EVAL_EPISODES):
            _, _, _, total_detected, *_ = runner.rolloutWorker.generate_episode(
                ep, evaluate=True, model=False)
            info = getattr(runner.rolloutWorker, "last_info", {})
            n_devices = env.n_devices or 10
            all_victims.append(float(info.get("victims_found", total_detected)))
            all_sr.append(float(info.get("success_rate", total_detected / n_devices)))
            all_energy.append(float(info.get("energy_used", 0.0)))

    vf = np.array(all_victims)
    sr = np.array(all_sr)
    en = np.array(all_energy)
    ddof = 1 if len(vf) > 1 else 0
    return {
        "victims_mean":      float(vf.mean()),
        "victims_std":       float(vf.std(ddof=ddof)),
        "success_rate_mean": float(sr.mean()),
        "energy_mean":       float(en.mean()),
    }


# ── LaTeX table ───────────────────────────────────────────────────────────────

def _write_latex(results, out_dir):
    maps = sorted(set(r["map"] for r in results))
    sigmas = sorted(set(r["sigma_device_db"] for r in results))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{BLE/smartphone hardware heterogeneity test (evaluation-only,"
        r" single seed). A fixed per-device SNR bias $\xi_k \sim \mathcal{N}(0,"
        r"\,\sigma_{\mathrm{device}}^2)$ is applied before contact detection."
        r" Values: mean\,$\pm$\,std victims localised out of 10"
        r" (averaged over " + str(N_BIAS_DRAWS) + r"\,bias draws"
        r" $\times$\," + str(N_EVAL_EPISODES) + r"\,episodes).}",
        r"\label{tab:device_heterogeneity}",
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Map & $\sigma_{\mathrm{device}}=0$\,dB & $3$\,dB & $6$\,dB \\",
        r"\midrule",
    ]
    for m in maps:
        cells = []
        for s in sigmas:
            row = next(r for r in results if r["map"] == m and r["sigma_device_db"] == s)
            cells.append(f"{row['victims_mean']:.2f}$\\pm${row['victims_std']:.2f}")
        lines.append(m + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    os.makedirs(out_dir, exist_ok=True)
    tex_path = os.path.join(out_dir, "table_device_heterogeneity.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table → {tex_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_common_args()
    args = get_mixer_args(args)

    if args.map == "RBM":
        from config.RBM_define import params
    elif args.map == "RDM":
        from config.RDM_define import params
    else:
        raise ValueError(f"Unknown map: {args.map}")

    args.alg      = "qmix"
    args.federated = True
    args.model     = True
    args.use_federation      = True
    args.use_surrogate_channel = True
    args.tag       = "_chshift_train_s1"   # reuse the FED-MOD model trained for channel-shift

    model_tag = f"_chshift_train_s{args.seed}"

    out_csv = os.path.join(args.result_dir, "device_heterogeneity_evaluation.csv")
    out_fig_dir = os.path.join(args.result_dir, "revision_figures")
    os.makedirs(args.result_dir, exist_ok=True)

    # Load existing CSV rows from other maps so we accumulate across runs
    existing_rows = []
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            for row in csv.DictReader(f):
                # Convert numeric fields back to float
                for k in ["sigma_device_db", "victims_mean", "victims_std",
                           "success_rate_mean", "energy_mean"]:
                    try:
                        row[k] = float(row[k])
                    except (ValueError, KeyError):
                        pass
                existing_rows.append(row)

    new_rows = []
    for sigma in SIGMA_DEVICE_LIST:
        print(f"\n[device-het] σ={sigma:.0f} dB  map={args.map}  seed={args.seed}")
        metrics = _evaluate_sigma(args, params, model_tag=model_tag,
                                  sigma_device=sigma)
        row = {
            "map": args.map,
            "seed": args.seed,
            "sigma_device_db": sigma,
            **metrics,
        }
        new_rows.append(row)
        print(f"  victims: {metrics['victims_mean']:.2f} ± {metrics['victims_std']:.2f}"
              f"  sr: {metrics['success_rate_mean']:.3f}")

    # Merge with existing (other map) rows, dedup by (map, seed, sigma)
    key = lambda r: (r["map"], str(r["seed"]), float(r["sigma_device_db"]))
    seen = {key(r) for r in new_rows}
    merged = [r for r in existing_rows if key(r) not in seen] + new_rows
    merged.sort(key=lambda r: (r["map"], float(r["sigma_device_db"])))

    fieldnames = ["map", "seed", "sigma_device_db", "victims_mean", "victims_std",
                  "success_rate_mean", "energy_mean"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)
    print(f"\nCSV → {out_csv}")

    # Write LaTeX only when both maps are present
    maps_present = {r["map"] for r in merged}
    if {"RBM", "RDM"}.issubset(maps_present):
        _write_latex(merged, out_fig_dir)
    else:
        print("(LaTeX table will be written once both RBM and RDM are done)")


if __name__ == "__main__":
    main()
