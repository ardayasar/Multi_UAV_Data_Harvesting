#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_ablation_grid.sh
# - Supports BOTH flag styles:
#   (A) old: --use_federation / --use_surrogate_channel
#   (B) new: --federated / --model / --total_episodes ...
#
# - Default: LIGHT ablation (6000 episodes, 3 seeds)
#   Change MODE="LIGHT" to MODE="FULL" if needed.
# ============================================================

PROJECT_ROOT="/Users/alparslanguzey/Desktop/Papers/FRL/Multi_UAV_Data_Harvesting"
cd "$PROJECT_ROOT" || exit 1

# (OPTIONAL) Conda
# source "$HOME/miniconda3/etc/profile.d/conda.sh"
# conda activate ippo_env

PYTHON="python3"     # safer than python on macOS
ENTRY="main.py"

MODE="LIGHT"         # LIGHT | FULL
FLAGSTYLE="OLD"      # OLD | NEW
SKIP_FED_NO_SUR=0    # 1 => skip FED=1,SUR=0 (if not implemented yet)

# ----------------------------
# LIGHT settings (ablation)
# ----------------------------
MAPS_LIGHT=("RBM" "RDM")
SEEDS_LIGHT=(1 2 3)
TOTAL_EP_LIGHT=1000
AGG_PERIOD_LIGHT=20
EVAL_CYCLE_LIGHT=20

# ----------------------------
# FULL settings (your original)
# ----------------------------
MAPS_FULL=("RBM" "RDM")
SEEDS_FULL=(1 2 3 4 5)
# (If your code supports total episodes in OLD flags too, you can set it here)
TOTAL_EP_FULL=1000
AGG_PERIOD_FULL=20
EVAL_CYCLE_FULL=20

# Pick config based on MODE
if [[ "$MODE" == "LIGHT" ]]; then
  MAPS=("${MAPS_LIGHT[@]}")
  SEEDS=("${SEEDS_LIGHT[@]}")
  TOTAL_EP="$TOTAL_EP_LIGHT"
  AGG_PERIOD="$AGG_PERIOD_LIGHT"
  EVAL_CYCLE="$EVAL_CYCLE_LIGHT"
else
  MAPS=("${MAPS_FULL[@]}")
  SEEDS=("${SEEDS_FULL[@]}")
  TOTAL_EP="$TOTAL_EP_FULL"
  AGG_PERIOD="$AGG_PERIOD_FULL"
  EVAL_CYCLE="$EVAL_CYCLE_FULL"
fi

echo "=== Running: MODE=${MODE}, FLAGSTYLE=${FLAGSTYLE}, TOTAL_EP=${TOTAL_EP}, SEEDS=(${SEEDS[*]}) ==="
echo "Python: $($PYTHON -c 'import sys; print(sys.executable)')"

for MAP in "${MAPS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for FED in 0 1; do
      for SUR in 0 1; do

        if [[ "$SKIP_FED_NO_SUR" -eq 1 && "$FED" -eq 1 && "$SUR" -eq 0 ]]; then
          echo "⏭ Skipping MAP=${MAP}, SEED=${SEED}, FED=${FED}, SUR=${SUR} (not wired yet)"
          continue
        fi

        # Tags
        if [[ "$FED" -eq 1 ]]; then FEDTAG="fed"; else FEDTAG="nofed"; fi
        if [[ "$SUR" -eq 1 ]]; then SURTAG="sur"; else SURTAG="nosur"; fi
        TAG="_${FEDTAG}_${SURTAG}_s${SEED}"

        echo "▶ MAP=${MAP}, SEED=${SEED} → ${FEDTAG}, ${SURTAG} (TAG=${TAG})"

        if [[ "$FLAGSTYLE" == "NEW" ]]; then
          # New style flags
          $PYTHON "$ENTRY" \
            --map "$MAP" \
            --seed "$SEED" \
            --alg qmix \
            --federated "$FED" \
            --model "$SUR" \
            --total_episodes "$TOTAL_EP" \
            --aggregation_period "$AGG_PERIOD" \
            --evaluate_cycle "$EVAL_CYCLE" \
            --tag "$TAG"
        else
          # Old style flags (your current working interface)
          # If your main.py supports --total_episodes / --aggregation_period / --evaluate_cycle
          # alongside old flags, KEEP these three lines. If it crashes, delete them.
          $PYTHON "$ENTRY" \
            --alg qmix \
            --map "$MAP" \
            --seed "$SEED" \
            --use_federation "$FED" \
            --use_surrogate_channel "$SUR" \
            --total_episodes "$TOTAL_EP" \
            --aggregation_period "$AGG_PERIOD" \
            --evaluate_cycle "$EVAL_CYCLE" \
            --tag "$TAG"
        fi

      done
    done
  done
done