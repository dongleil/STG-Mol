#!/usr/bin/env bash
# ============================================================================
# STG-Mol — Two-stage cascaded virtual screening on ZINC
# ============================================================================
# Reproduces Section 5.4 of the paper:
#   Stage 0 : Rule-based drug-likeness filter (Lipinski + Veber + PAINS + DILI)
#   Stage 1 : 1D+2D lightweight coarse screening (~5 ms/mol)
#   Stage 2 : Full STG-Mol tri-modal fine screening (~120 ms/mol)
#   Stage 3 : Butina clustering for diversity-preserving redundancy reduction
#
# Prerequisites:
#   1) A trained STG-Mol full model (results/main_5seeds/seed_42/models/...)
#   2) A trained 1D+2D lightweight model (results/ablation/modality_1d_2d/...)
#   3) ZINC library CSV under data/processed/zinc/zinc_druglike.csv
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ------- Configurable paths (override via env vars) -------
: "${ZINC_DB:=data/processed/zinc/zinc_druglike.csv}"
: "${STAGE1_MODEL:=results/ablation/modality_1d_2d/seed_42/models/1D+2D_none.pt}"
: "${STAGE2_MODEL:=results/main_5seeds/seed_42/models/1D+2D+3D_self_attention.pt}"
: "${OUTPUT_DIR:=results/screening/zinc_run_$(date +%Y%m%d_%H%M%S)}"

# ------- Cascade thresholds -------
: "${TOP_K_RATIO:=0.05}"    # Stage 1 pass-through ratio (~5%)
: "${STAGE2_THRESHOLD:=0.5}"# Stage 2 activity probability threshold
: "${DIVERSITY_POOL:=2000}" # Top-N for Butina clustering
: "${FINAL_TOP_N:=142}"     # Final representative count after clustering
: "${TANIMOTO_CUTOFF:=0.80}"

mkdir -p "$OUTPUT_DIR"

echo "==========================================================="
echo "STG-Mol · Cascaded Virtual Screening on ZINC"
echo "  Input db     : $ZINC_DB"
echo "  Stage-1 model: $STAGE1_MODEL"
echo "  Stage-2 model: $STAGE2_MODEL"
echo "  Output dir   : $OUTPUT_DIR"
echo "  Top-K ratio  : $TOP_K_RATIO"
echo "  Final Top-N  : $FINAL_TOP_N"
echo "==========================================================="

# Sanity checks
for f in "$STAGE1_MODEL" "$STAGE2_MODEL" "$ZINC_DB"; do
  if [[ ! -f "$f" ]]; then
    echo "❌ Missing file: $f" >&2
    exit 1
  fi
done

python src/utils/virtual_screening.py \
  --database         "$ZINC_DB" \
  --stage1_model     "$STAGE1_MODEL" \
  --stage2_model     "$STAGE2_MODEL" \
  --output           "$OUTPUT_DIR" \
  --top_k_ratio      "$TOP_K_RATIO" \
  --stage2_threshold "$STAGE2_THRESHOLD" \
  --diversity_pool   "$DIVERSITY_POOL" \
  --final_top_n      "$FINAL_TOP_N" \
  --tanimoto_cutoff  "$TANIMOTO_CUTOFF"

echo ""
echo "==========================================================="
echo "Screening complete. Representative candidates:"
echo "  $OUTPUT_DIR/final_candidates.csv"
echo ""
echo "Next step (multi-level validation, Section 5.5):"
echo "  python src/evaluation/validation_pipeline.py \\"
echo "      --candidates $OUTPUT_DIR/final_candidates.csv"
echo "==========================================================="
