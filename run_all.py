#!/usr/bin/env python3
"""
run_all.py — Run all remaining algorithms then generate the seed table.

Already done : QMIX  (seeds 1-5)
This script  : IQL, MOD, FED, IPPO  (seeds 1-5 each)
Then         : generates result/seed_table_RBM.tex + .csv

Usage:
    python3 run_all.py                        # RBM, seeds 1-5
    python3 run_all.py --map RDM
    python3 run_all.py --seeds 1 2 3
    python3 run_all.py --skip_ippo            # if RLLib / IPPO deps missing
"""

import argparse
import subprocess
import sys
import time


def run(cmd: list, label: str):
    print(f"\n{'='*60}")
    print(f"  STARTING: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[DONE] {label} — {elapsed/60:.1f} min\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map",       type=str, default="RBM", choices=["RBM", "RDM"])
    parser.add_argument("--seeds",     type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--episodes",  type=int, default=30000, help="total_episodes for main.py runs")
    parser.add_argument("--ippo_episodes", type=int, default=3000, help="episodes for IPPO")
    parser.add_argument("--skip_ippo",      action="store_true", help="skip IPPO if deps not available")
    parser.add_argument("--device",          type=str, default="cpu")
    parser.add_argument("--evaluate_epoch",  type=int, default=20,
                        help="eval episodes per checkpoint (5=fast, 20=statistically robust)")
    args = parser.parse_args()

    py = sys.executable
    map_name = args.map
    seeds_str = [str(s) for s in args.seeds]
    common = [
        "--map", map_name,
        "--seeds", *seeds_str,
        "--sar_reward", "True",
        "--total_episodes", str(args.episodes),
        "--device", args.device,
        "--evaluate_epoch", str(args.evaluate_epoch),
    ]

    # ── 1. IQL ────────────────────────────────────────────────────
    run(
        [py, "main.py", "--alg", "iql"] + common,
        f"IQL  | map={map_name} | seeds={args.seeds}"
    )

    # ── 2. MOD (Model-aided QMIX) ─────────────────────────────────
    run(
        [py, "main.py", "--alg", "qmix", "--model", "True"] + common,
        f"MOD  | map={map_name} | seeds={args.seeds}"
    )

    # ── 3. FED (Federated QMIX) ───────────────────────────────────
    run(
        [py, "main.py", "--alg", "qmix", "--federated", "True"] + common,
        f"FED  | map={map_name} | seeds={args.seeds}"
    )

    # ── 4. IPPO ───────────────────────────────────────────────────
    if not args.skip_ippo:
        for seed in args.seeds:
            run(
                [py, "ippo_custom.py", "--map", map_name,
                 "--seed", str(seed), "--episodes", str(args.ippo_episodes)],
                f"IPPO | map={map_name} | seed={seed}"
            )
    else:
        print("[SKIP] IPPO skipped (--skip_ippo flag set)\n")

    # ── 5. Generate seed table ────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating seed evaluation table...")
    print(f"{'='*60}\n")
    subprocess.run([py, "generate_seed_table.py"])
    print("\nAll done! Check result/seed_table_RBM.csv and result/seed_table_RBM.tex")


if __name__ == "__main__":
    main()
