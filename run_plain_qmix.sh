#!/usr/bin/env bash
# Full experiment: plain (model-free) QMIX
set -e

TOTAL_EPS=1000
EVAL_CYCLE=10              # ⇒ 101 points
DEVICE=cpu
N_AGENTS=3
MAPS=(RBM RDM)
SEEDS=(1 2 3)
RESULT_ROOT="result/qmix"

echo "🗑  Removing previous plain-QMIX results …"
for m in "${MAPS[@]}"; do
  rm -rf "${RESULT_ROOT}/${m}_qm_s"* || true
done
echo "✅  Clean slate."

run_one () {
  local MAP=$1  SEED=$2  TAG="_qm_s${SEED}"
  echo -e "\n🏃  plain-QMIX | map=${MAP} | seed=${SEED}\n"

  python main.py \
      --map "$MAP" \
      --alg qmix \
      --n_agents $N_AGENTS \
      --total_episodes $TOTAL_EPS \
      --evaluate_cycle $EVAL_CYCLE \
      --device $DEVICE \
      --seed $SEED \
      --tag "$TAG"
      # (no --model / --federated → both False)
}

for MAP in "${MAPS[@]}"; do
  for SEED in "${SEEDS[@]}"; do run_one "$MAP" "$SEED"; done
done

echo -e "\n🎉  All plain-QMIX jobs finished."