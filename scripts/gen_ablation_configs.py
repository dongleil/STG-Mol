#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate all ablation study YAML configs from the main config template.
Produces 10 configs under configs/ablation/.
"""
from pathlib import Path
import yaml

MAIN_CFG = Path('/Users/bytedance/code_workspace/nlrp3/STG-Mol/configs/train_main.yaml')
ABL_DIR  = Path('/Users/bytedance/code_workspace/nlrp3/STG-Mol/configs/ablation')
ABL_DIR.mkdir(parents=True, exist_ok=True)

with open(MAIN_CFG, 'r', encoding='utf-8') as f:
    base = yaml.safe_load(f)


def dump(name, cfg, comment):
    """Write with header comment."""
    header = f"# {comment}\n# " + "=" * 74 + "\n\n"
    path = ABL_DIR / f"{name}.yaml"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  ✓ {path.name}")


def clone(base_cfg):
    """Deep copy base config."""
    import copy
    return copy.deepcopy(base_cfg)


# ==========================================================
# 1) Modality-combination ablations (7 configs)
# ==========================================================
print("[1/3] Modality-combination ablations:")

modality_specs = [
    ('modality_1d',    '1D',       'none',           '1D modality only (Mol2Vec)'),
    ('modality_2d',    '2D',       'none',           '2D modality only (D-MPNN)'),
    ('modality_3d',    '3D',       'none',           '3D modality only (SchNet)'),
    ('modality_1d_2d', '1D+2D',    'concat',         '1D + 2D bi-modal (Concat fusion)'),
    ('modality_1d_3d', '1D+3D',    'concat',         '1D + 3D bi-modal (Concat fusion)'),
    ('modality_2d_3d', '2D+3D',    'concat',         '2D + 3D bi-modal (Concat fusion)'),
    ('modality_1d_2d_3d_concat', '1D+2D+3D', 'concat',
        'Tri-modal (1D+2D+3D) with Concat fusion — baseline for fusion ablation'),
]

for name, mode, ftype, desc in modality_specs:
    cfg = clone(base)
    cfg['experiment']['name']       = f"STG-Mol_ablation_{name}"
    cfg['output']['base_dir']       = f"results/ablation/{name}"
    cfg['experiments']              = [{'fusion_mode': mode, 'fusion_type': ftype}]
    dump(name, cfg, f"Ablation: {desc}")


# ==========================================================
# 2) Fusion-strategy ablation (already covered above)
# ==========================================================
# The Concat vs Self-Attention comparison uses:
#   modality_1d_2d_3d_concat.yaml (Concat)
#   train_main.yaml (Self-Attention)
# so no extra file needed here.


# ==========================================================
# 3) Regularisation ablations (3 configs for 2x2 factorial)
# ==========================================================
print("\n[2/3] Regularisation ablations:")

# (B) BalanceLoss ON, ModalityDropout OFF
cfg_b = clone(base)
cfg_b['experiment']['name']            = "STG-Mol_ablation_B_balance_only"
cfg_b['output']['base_dir']            = "results/ablation/reg_B_balance_only"
cfg_b['training']['modality_drop_prob'] = 0.0
dump('no_modality_dropout', cfg_b,
     'Ablation: BalanceLoss ON, ModalityDropout OFF (row B of 2x2 factorial)')

# (C) BalanceLoss OFF, ModalityDropout ON
cfg_c = clone(base)
cfg_c['experiment']['name']            = "STG-Mol_ablation_C_dropout_only"
cfg_c['output']['base_dir']            = "results/ablation/reg_C_dropout_only"
cfg_c['model']['balance_loss_weight']  = 0.0
dump('no_balance_loss', cfg_c,
     'Ablation: BalanceLoss OFF, ModalityDropout ON (row C of 2x2 factorial)')

# (A) Both OFF — Baseline
cfg_a = clone(base)
cfg_a['experiment']['name']            = "STG-Mol_ablation_A_no_regularisation"
cfg_a['output']['base_dir']            = "results/ablation/reg_A_baseline"
cfg_a['model']['balance_loss_weight']  = 0.0
cfg_a['training']['modality_drop_prob'] = 0.0
dump('no_both', cfg_a,
     'Ablation: BalanceLoss OFF, ModalityDropout OFF (row A of 2x2 factorial)')

# Row D (both ON) uses train_main.yaml directly.

print("\n[3/3] All ablation configs generated under configs/ablation/")
print(f"    Total: {len(list(ABL_DIR.glob('*.yaml')))} files")
