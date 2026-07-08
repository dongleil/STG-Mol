#!/usr/bin/env bash
# ============================================================================
# STG-Mol — Main experiment with 5 random seeds
# ============================================================================
# Reproduces Table 5.1 (main comparison, mean ± std over 5 seeds).
#
# Usage:
#   bash scripts/run_5seeds.sh                 # sequential (5x wall-clock)
#   bash scripts/run_5seeds.sh --parallel      # parallel (needs 5 GPUs)
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_5seeds.sh
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS=(42 123 2024 3407 7)
BASE_CFG="configs/train_main.yaml"
OUT_ROOT="results/main_5seeds"
mkdir -p "$OUT_ROOT"

PARALLEL=0
if [[ "${1:-}" == "--parallel" ]]; then
  PARALLEL=1
fi

echo "==========================================================="
echo "STG-Mol · 5-Seed Main Experiment"
echo "  Seeds:  ${SEEDS[*]}"
echo "  Config: $BASE_CFG"
echo "  Output: $OUT_ROOT"
echo "  Mode:   $([ $PARALLEL -eq 1 ] && echo parallel || echo sequential)"
echo "==========================================================="

run_one_seed() {
  local seed="$1"
  local cfg_tmp="$OUT_ROOT/seed_${seed}.yaml"
  # Rewrite seed and output dir in a temporary config
  python3 - <<PY
import yaml, sys
with open("$BASE_CFG") as f:
    cfg = yaml.safe_load(f)
cfg['experiment']['random_seed'] = ${seed}
cfg['output']['base_dir'] = "$OUT_ROOT/seed_${seed}"
with open("$cfg_tmp", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
PY

  echo ""
  echo "-----------------------------------------------------------"
  echo ">>> Running seed=${seed}"
  echo "-----------------------------------------------------------"
  python src/training/train.py --config "$cfg_tmp"
}

if [[ $PARALLEL -eq 1 ]]; then
  for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    CUDA_VISIBLE_DEVICES="$i" run_one_seed "$seed" &
  done
  wait
else
  for seed in "${SEEDS[@]}"; do
    run_one_seed "$seed"
  done
fi

echo ""
echo "==========================================================="
echo "All 5 seeds completed."
echo "Results are under: $OUT_ROOT"
echo "==========================================================="
echo "To aggregate to a mean ± std table:"
echo "  python src/evaluation/compare_experiments.py $OUT_ROOT/seed_*"
