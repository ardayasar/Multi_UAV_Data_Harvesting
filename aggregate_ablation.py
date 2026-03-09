import os
import numpy as np
import pandas as pd


# ---- configuration you can tweak if needed ----
MAPS = ["RBM", "RDM"]
SEEDS = [1, 2, 3]          # change to [1,2,3,4,5] later if you run more seeds
FED_FLAGS = [0, 1]         # 0 = nofed, 1 = fed
SUR_FLAGS = [0, 1]         # 0 = nosur, 1 = sur
LAST_K = 5                 # how many eval points to average at the end
RESULT_ROOT = "result/qmix"


def exp_dir(map_name: str, fed: int, sur: int, seed: int) -> str:
    """Reconstruct the folder name used by run_ablation_grid.sh."""
    fed_label = "fed" if fed else "nofed"
    sur_label = "sur" if sur else "nosur"
    tag = f"{map_name}_{fed_label}_{sur_label}_s{seed}"
    return os.path.join(RESULT_ROOT, tag)


def load_final_score(path: str) -> float | None:
    """Load global_data.npy and return mean of last K evals."""
    f = os.path.join(path, "global_data.npy")
    if not os.path.exists(f):
        print(f"[WARN] Missing {f}")
        return None

    arr = np.load(f)
    if arr.size == 0:
        print(f"[WARN] Empty global_data in {path}")
        return None

    k = min(LAST_K, arr.size)
    return float(arr[-k:].mean())


def main():
    rows = []

    for map_name in MAPS:
        for fed in FED_FLAGS:
            for sur in SUR_FLAGS:

                per_seed_vals = []

                for seed in SEEDS:
                    d = exp_dir(map_name, fed, sur, seed)
                    score = load_final_score(d)
                    if score is None:
                        continue

                    rows.append({
                        "map": map_name,
                        "fed": fed,
                        "sur": sur,
                        "seed": seed,
                        "final_lastK": score,
                    })
                    per_seed_vals.append(score)

                if per_seed_vals:
                    mean = float(np.mean(per_seed_vals))
                    std = float(np.std(per_seed_vals, ddof=1)) if len(per_seed_vals) > 1 else 0.0
                    print(
                        f"[SUMMARY] map={map_name}, fed={fed}, sur={sur}  "
                        f"→ mean={mean:.3f}, std={std:.3f}, n={len(per_seed_vals)}"
                    )

    if not rows:
        print("No runs found. Check your paths / finished runs.")
        return

    df = pd.DataFrame(rows)
    os.makedirs("result", exist_ok=True)

    csv_per_seed = os.path.join("result", "ablation_per_seed.csv")
    csv_summary = os.path.join("result", "ablation_summary.csv")

    # Per-seed entries
    df.to_csv(csv_per_seed, index=False)

    # Grouped mean±std for direct use in the paper
    grouped = (
        df.groupby(["map", "fed", "sur"])["final_lastK"]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    grouped.to_csv(csv_summary, index=False)

    print(f"\nSaved per-seed results to:   {csv_per_seed}")
    print(f"Saved grouped summary to:    {csv_summary}")
    print("\nUse the grouped CSV to fill in Table B (Ablations).")


if __name__ == "__main__":
    main()
