#!/usr/bin/env bash
# ============================================================================
# STG-Mol — Ablation study suite
# ============================================================================
# Reproduces Tables 5.2 / 5.3 / 5.4 of the paper.
#
# Ablations included (default: all)
#   --modality  Modality-combination (7 configs)
#   --fusion    Fusion-strategy (Concat vs Self-Attention)
#   --reg       Regularisation 2x2 factorial (BalanceLoss × ModalityDropout)
#
# Each ablation is repeated with 3 seeds by default (--seeds).
# Increase to 5 for the final camera-ready.
#
# Usage:
#   bash scripts/run_ablation.sh                     # all ablations, 3 seeds
#   bash scripts/run_ablation.sh --modality          # only modality
#   bash scripts/run_ablation.sh --reg --seeds 5     # regularisation with 5 seeds
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_MODALITY=0
RUN_FUSION=0
RUN_REG=0
SEEDS=(42 123 2024)

# Parse args
if [[ $# -eq 0 ]]; then
  RUN_MODALITY=1; RUN_FUSION=1; RUN_REG=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --modality) RUN_MODALITY=1; shift ;;
    --fusion)   RUN_FUSION=1;   shift ;;
    --reg)      RUN_REG=1;      shift ;;
    --seeds)    IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
    -h|--help)
      grep -E "^#" "$0" | head -30; exit 0 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

OUT_ROOT="results/ablation"
mkdir -p "$OUT_ROOT"

echo "==========================================================="
echo "STG-Mol · Ablation Suite"
echo "  Modality:       $([ $RUN_MODALITY -eq 1 ] && echo YES || echo no)"
echo "  Fusion:         $([ $RUN_FUSION -eq 1 ] && echo YES || echo no)"
echo "  Regularisation: $([ $RUN_REG -eq 1 ] && echo YES || echo no)"
echo "  Seeds:          ${SEEDS[*]}"
echo "==========================================================="

run_config() {
  local cfg_path="$1"
  local ablation_name="$2"
  for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> ablation=${ablation_name}  seed=${seed}"

    local tmp_cfg="$OUT_ROOT/${ablation_name}_seed${seed}.yaml"
    python3 - <<PY
import yaml
with open("$cfg_path") as f:
    cfg = yaml.safe_load(f)
cfg['experiment']['random_seed'] = ${seed}
cfg['output']['base_dir'] = "$OUT_ROOT/${ablation_name}/seed_${seed}"
with open("$tmp_cfg", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
PY
    python src/training/train.py --config "$tmp_cfg"
  done
}

# ------- 1. Modality-combination ablation -------
if [[ $RUN_MODALITY -eq 1 ]]; then
  echo ""
  echo "[Section 5.2.1] Modality-combination ablation"
  for cfg in configs/ablation/modality_1d.yaml \
             configs/ablation/modality_2d.yaml \
             configs/ablation/modality_3d.yaml \
             configs/ablation/modality_1d_2d.yaml \
             configs/ablation/modality_1d_3d.yaml \
             configs/ablation/modality_2d_3d.yaml \
             configs/ablation/modality_1d_2d_3d_concat.yaml; do
    name=$(basename "$cfg" .yaml)
    run_config "$cfg" "$name"
  done
fi

# ------- 2. Fusion strategy ablation -------
# Uses train_main.yaml (self-attention) vs modality_1d_2d_3d_concat.yaml (concat).
# Modality already includes 'modality_1d_2d_3d_concat', so we only add self-attention.
if [[ $RUN_FUSION -eq 1 ]]; then
  echo ""
  echo "[Section 5.2.2] Fusion strategy ablation (Self-Attention arm)"
  run_config "configs/train_main.yaml" "fusion_self_attention"
fi

# ------- 3. Regularisation ablation -------
if [[ $RUN_REG -eq 1 ]]; then
  echo ""
  echo "[Section 5.2.3] Regularisation 2x2 factorial"
  # Row D (full scheme) — uses train_main.yaml; overlaps with fusion_self_attention above.
  run_config "configs/ablation/no_modality_dropout.yaml" "reg_B_balance_only"
  run_config "configs/ablation/no_balance_loss.yaml"    "reg_C_dropout_only"
  run_config "configs/ablation/no_both.yaml"            "reg_A_baseline"
fi

echo ""
echo "==========================================================="
echo "Ablation study finished."
echo "Aggregate results:"
echo "  python src/evaluation/compare_experiments.py $OUT_ROOT/*/seed_*"
echo "==========================================================="
