#!/usr/bin/env bash
# ============================================================================
# run_fusion_ablation.sh — 5-seed sweep for the fusion-component ablation
# ============================================================================
# Trains 5 fusion configurations × 5 seeds = 25 runs on the V3-random split.
# Each run writes per-seed test-set prediction CSVs consumable by
# scripts/recompute_metrics_v42.py, and the summary script below aggregates
# them into a single paper-ready table.
#
# Usage:
#   # From the STG-Mol repo root:
#   bash scripts/run_fusion_ablation.sh
#   # then:
#   python scripts/summarise_fusion_ablation.py \
#       --results_root results/ablation_fusion \
#       --output_md   fusion_ablation_table.md
#
# Estimated runtime on RTX 4090: ~90 min per full-model run × 25 runs ≈ 38 h.
# Set RUN_SEEDS below to reduce for a quick smoke test.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
REPO="$(pwd)"
echo "Repo root: ${REPO}"

CONFIGS=(
  "configs/ablation/fusion_full.yaml"
  "configs/ablation/fusion_no_cross_attn.yaml"
  "configs/ablation/fusion_no_gated.yaml"
  "configs/ablation/fusion_no_bilinear.yaml"
  "configs/ablation/fusion_no_importance_net.yaml"
)

TRAIN_ENTRY="${TRAIN_ENTRY:-src/training/train_v26.py}"
PYTHON="${PYTHON:-python}"

# Each config already carries random_seeds: [42, 123, 2024, 3407, 7]
# and the train script iterates them internally.
for cfg in "${CONFIGS[@]}"; do
  echo ""
  echo "================================================================"
  echo "  ${cfg}"
  echo "================================================================"
  "${PYTHON}" "${TRAIN_ENTRY}" --config "${cfg}" 2>&1 | tee "$(basename "${cfg}" .yaml).log"
done

echo ""
echo "✓ All 5 fusion-ablation runs complete."
echo "  Aggregate with:"
echo "    python scripts/summarise_fusion_ablation.py \\"
echo "        --results_root results/ablation_fusion --output_md fusion_ablation_table.md"
