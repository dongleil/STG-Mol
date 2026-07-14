# ============================================================================
# run_fusion_ablation.ps1 — Windows / PowerShell driver for the fusion-component
# ablation (5 configs × 5 seeds).
#
# Usage (from the STG-Mol repo root):
#   .\scripts\run_fusion_ablation.ps1
#
# Optional override:
#   $env:TRAIN_ENTRY = 'src/training/train_v26.py'   # default already
#   $env:PY          = 'python'                       # default already
#
# The bash version (run_fusion_ablation.sh) exists for Linux/macOS; this file
# is the equivalent for the RTX 4090 Windows box.
# ============================================================================
$ErrorActionPreference = 'Stop'

# cd to repo root (this file lives in scripts/)
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
Write-Host "Repo root: $repoRoot"

$configs = @(
    'configs/ablation/fusion_full.yaml',
    'configs/ablation/fusion_no_cross_attn.yaml',
    'configs/ablation/fusion_no_gated.yaml',
    'configs/ablation/fusion_no_bilinear.yaml',
    'configs/ablation/fusion_no_importance_net.yaml'
)

$trainEntry = if ($env:TRAIN_ENTRY) { $env:TRAIN_ENTRY } else { 'src/training/train_v26.py' }
$python     = if ($env:PY)          { $env:PY }          else { 'python' }

foreach ($cfg in $configs) {
    Write-Host ""
    Write-Host "================================================================"
    Write-Host "  $cfg"
    Write-Host "================================================================"
    $logName = [System.IO.Path]::GetFileNameWithoutExtension($cfg) + '.log'
    & $python $trainEntry --config $cfg 2>&1 | Tee-Object -FilePath $logName
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Non-zero exit ($LASTEXITCODE) on $cfg — continuing to next config."
    }
}

Write-Host ""
Write-Host "All 5 fusion-ablation runs complete."
Write-Host "Aggregate with:"
Write-Host "  python scripts/summarise_fusion_ablation.py ``"
Write-Host "      --results_root results/ablation_fusion ``"
Write-Host "      --output_md   fusion_ablation_table.md"
