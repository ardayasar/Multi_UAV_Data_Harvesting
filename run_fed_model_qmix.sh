#!/usr/bin/env bash
# Full experiment: model-aided FedQMIX (QMIX + SLAL + FedAvg)
set -e

TOTAL_EPISODES=1000
AGGREGATION_PERIOD=10          # one log per round → 100 + initial = 101
MODEL_LEARNING_PERIOD=1000     # SLAL update once mid-run
WORKERS=3
DEVICE=cpu
N_AGENTS=3
MAPS=(RBM RDM)
SEEDS=(1 2 3)
RESULT_ROOT="result/qmix"

echo "🗑  Removing previous FedQMIX results …"
for m in "${MAPS[@]}"; do
  rm -rf "${RESULT_ROOT}/${m}_fed_s"* || true
done
echo "✅  Clean slate."

run_one () {
  local MAP=$1 SEED=$2 TAG="_fed_s${SEED}"
  echo -e "\n🏃  FedQMIX | map=${MAP} | seed=${SEED}\n"

  python main.py \
      --map "$MAP" \
      --alg qmix \
      --model True \
      --federated True \
      --workers $WORKERS \
      --n_agents $N_AGENTS \
      --total_episodes $TOTAL_EPISODES \
      --aggregation_period $AGGREGATION_PERIOD \
      --model_learning_period $MODEL_LEARNING_PERIOD \
      --device $DEVICE \
      --seed $SEED \
      --tag "$TAG"
}

for MAP in "${MAPS[@]}"; do
  for SEED in "${SEEDS[@]}"; do run_one "$MAP" "$SEED"; done
done

echo -e "\n🎉  FedQMIX jobs finished."