#!/usr/bin/env bash
# Full experiment: model-aided (SLAL) QMIX   –  no federation
set -e

# ── core knobs ──────────────────────────────────────────────────────────────
TOTAL_EPS=1000            # 1 000 episodes  ⇒ 101 eval points with eval_cycle=10
EVAL_CYCLE=10
DEVICE=cpu
N_AGENTS=3
MAPS=(RBM RDM)
SEEDS=(1 2 3)
RESULT_ROOT="result/qmix"

# quick smoke‐test?   ./run_model_qmix_full.sh --quick
if [[ $1 == "--quick" || $1 == "-q" ]]; then
  TOTAL_EPS=30
  EVAL_CYCLE=3
  echo "⚡  QUICK mode → TOTAL_EPISODES=${TOTAL_EPS}"
fi

# ── cleanup ─────────────────────────────────────────────────────────────────
echo "🗑  Removing previous model-QMIX results …"
for m in "${MAPS[@]}"; do
  rm -rf "${RESULT_ROOT}/${m}_mod_s"* || true
done
echo "✅  Clean slate."

# ── helper ──────────────────────────────────────────────────────────────────
run_one () {
  local MAP=$1 SEED=$2 TAG="_mod_s${SEED}"
  echo -e "\n🏃  model-QMIX | map=${MAP} | seed=${SEED}\n"

  python main.py \
      --map "$MAP" \
      --alg qmix \
      --model True --n_agents ${N_AGENTS} \
      --total_episodes $TOTAL_EPS \
      --evaluate_cycle $EVAL_CYCLE \
      --device $DEVICE \
      --seed $SEED \
      --tag "$TAG"
      # (omit --federated ⇒ False)
}

# ── sweep ───────────────────────────────────────────────────────────────────
for MAP in "${MAPS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    run_one "$MAP" "$SEED"
  done
done

echo -e "\n🎉  All model-QMIX jobs finished."