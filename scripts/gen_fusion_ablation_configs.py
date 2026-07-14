#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_fusion_ablation_configs.py — generate 4 + 1 configs for the fusion-component
ablation study (Table 5.2.3, paper §5.2.3).

Ablations:
    fusion_no_cross_attn.yaml       -- disable Cross-Modal Attention
    fusion_no_gated.yaml            -- disable Gated Fusion Unit
    fusion_no_bilinear.yaml         -- disable Low-Rank Bilinear branch
    fusion_no_importance_net.yaml   -- disable Importance Net (uniform 1/3 weights)
    fusion_full.yaml                -- full hierarchical fusion (control)

All configs inherit from configs/train_v26_v3_random.yaml (the paper's V3-random
protocol) so that everything else — data, seeds, hyperparams — is identical.

Each config runs 5 seeds and produces a per-seed prediction CSV consumable by
scripts/recompute_metrics_v42.py.

Usage:
    python scripts/gen_fusion_ablation_configs.py
"""
from pathlib import Path
import copy
import yaml

REPO = Path(__file__).resolve().parents[1]
BASE_CFG = REPO / 'configs' / 'train_v26_v3_random.yaml'
OUT_DIR  = REPO / 'configs' / 'ablation'
OUT_DIR.mkdir(parents=True, exist_ok=True)


ABLATIONS = [
    ('fusion_full', {
        'no_cross_attn':     False,
        'no_gated':          False,
        'no_bilinear':       False,
        'no_importance_net': False,
    }, 'Full hierarchical fusion (control) — all four components active.'),
    ('fusion_no_cross_attn', {
        'no_cross_attn':     True,
        'no_gated':          False,
        'no_bilinear':       False,
        'no_importance_net': False,
    }, 'Ablation: disable Cross-Modal Attention (pairwise Q-K-V is bypassed).'),
    ('fusion_no_gated', {
        'no_cross_attn':     False,
        'no_gated':          True,
        'no_bilinear':       False,
        'no_importance_net': False,
    }, 'Ablation: disable Gated Fusion Unit (replaced by pairwise mean).'),
    ('fusion_no_bilinear', {
        'no_cross_attn':     False,
        'no_gated':          False,
        'no_bilinear':       True,
        'no_importance_net': False,
    }, 'Ablation: disable Low-Rank Bilinear branch (removed from final concat).'),
    ('fusion_no_importance_net', {
        'no_cross_attn':     False,
        'no_gated':          False,
        'no_bilinear':       False,
        'no_importance_net': True,
    }, 'Ablation: disable Importance Net (uniform 1/3 modality weighting).'),
]


def main():
    with open(BASE_CFG, 'r', encoding='utf-8') as f:
        base = yaml.safe_load(f)

    for name, ablate, desc in ABLATIONS:
        cfg = copy.deepcopy(base)
        cfg['experiment']['name']    = f'v3_random_{name}'
        cfg['output']['base_dir']    = f'results/ablation_fusion/{name}'
        cfg.setdefault('model', {})['fusion_ablate'] = ablate

        out_path = OUT_DIR / f'{name}.yaml'
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f'# {desc}\n')
            f.write('# ' + '=' * 74 + '\n')
            f.write('# Inherits every hyper-param from train_v26_v3_random.yaml;\n'
                    '# only model.fusion_ablate differs.\n\n')
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False,
                       allow_unicode=True)
        print(f'  ✓ wrote {out_path.relative_to(REPO)}')

    print()
    print('Done. To run the full ablation sweep on the Windows RTX 4090:')
    print('  bash scripts/run_fusion_ablation.sh')


if __name__ == '__main__':
    main()
