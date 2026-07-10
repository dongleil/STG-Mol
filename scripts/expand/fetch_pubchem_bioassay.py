#!/usr/bin/env python3
"""
Fetch NLRP3-related bioassay data from PubChem BioAssay.

Uses PubChem's REST API to pull confirmed actives and inactives from
public assays that measured NLRP3 inflammasome inhibition.

Curated assay IDs (AIDs) targeting NLRP3:
  * 1508591  -- NLRP3 inflammasome inhibition SAR (BROAD)
  * 1443     -- IL-1β release inhibition (proxy for NLRP3)
  * 1319407  -- NLRP3 inflammasome ATP-dependent activation
  * 1259410  -- caspase-1 activation via NLRP3

For each AID we fetch:
  * Confirmed active CIDs → active molecules
  * Confirmed inactive CIDs → real negative molecules (gold standard!)
Then resolve each CID to canonical SMILES.

Usage:
    python scripts/expand/fetch_pubchem_bioassay.py \\
        --output data/raw/pubchem_nlrp3.csv \\
        --max_per_aid 500

If the API is throttled or unavailable, the script writes what it
got and reports partial results.
"""
import argparse
import csv
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Confirmed NLRP3-related PubChem BioAssay IDs
# V5 strict curation: only AID 1508591 (direct NLRP3 inflammasome inhibition).
# AID 1443 (IL-1β release) removed — it's a downstream proxy that introduced
# ~1000 label-noisy actives in V4 and dropped Test AUC by ~0.11.
NLRP3_AIDS = [
    ('1508591', 'NLRP3 inflammasome inhibition SAR (direct)'),
]

BASE = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'


def _get(url, timeout=30):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'STG-Mol/1.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8')
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f'  ⚠ {e}')
        return None


def fetch_active_inactive_cids(aid):
    """
    Fetch CID lists for active and inactive outcomes of an assay.
    Returns (active_cids, inactive_cids) as sets.
    """
    active_url = f'{BASE}/assay/aid/{aid}/cids/JSON?cids_type=active'
    inactive_url = f'{BASE}/assay/aid/{aid}/cids/JSON?cids_type=inactive'

    active, inactive = set(), set()
    for tag, url in [('active', active_url), ('inactive', inactive_url)]:
        data = _get(url)
        if not data:
            continue
        try:
            obj = json.loads(data)
            cids = obj.get('InformationList', {}).get('Information', [{}])[0].get('CID', [])
            (active if tag == 'active' else inactive).update(cids)
            print(f'    {tag}: {len(cids)} CIDs')
        except Exception as e:
            print(f'    parse error {tag}: {e}')
    return active, inactive


def fetch_smiles_batch(cids, batch=100):
    """
    Resolve CID → SMILES in batches.

    PubChem PUG-REST as of 2025 returns the property key `SMILES`
    (previously `CanonicalSMILES`, deprecated). We request `SMILES`
    directly and fall back to any legacy keys defensively.
    Returns dict {cid: smiles}.
    """
    out = {}
    cids = list(cids)
    for i in range(0, len(cids), batch):
        chunk = cids[i:i + batch]
        url = (f'{BASE}/compound/cid/{",".join(map(str, chunk))}'
               f'/property/SMILES/JSON')
        data = _get(url)
        if not data:
            continue
        try:
            obj = json.loads(data)
            props = obj.get('PropertyTable', {}).get('Properties', [])
            for p in props:
                cid = p.get('CID')
                smi = (p.get('SMILES')
                       or p.get('CanonicalSMILES')
                       or p.get('IsomericSMILES'))
                if cid and smi:
                    out[cid] = smi
        except Exception as e:
            print(f'    smiles parse error at batch {i}: {e}')
        time.sleep(0.3)   # be polite to NIH
        if (i + batch) % 500 == 0 and i > 0:
            print(f'    resolved {min(i + batch, len(cids))}/{len(cids)} SMILES '
                  f'(cumulative found: {len(out)})')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--max_per_aid', type=int, default=1000)
    ap.add_argument('--aids', nargs='*', default=None,
                    help='Override AIDs to query (default: curated NLRP3 list).')
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    aids_to_use = args.aids if args.aids else [a for a, _ in NLRP3_AIDS]

    print(f'\nQuerying {len(aids_to_use)} PubChem BioAssays...')
    all_active, all_inactive = set(), set()
    aid_source = {}
    for aid in aids_to_use:
        desc = dict(NLRP3_AIDS).get(aid, 'custom')
        print(f'\n  AID {aid} ({desc}):')
        active, inactive = fetch_active_inactive_cids(aid)
        for cid in active:
            aid_source.setdefault(cid, f'PubChem_AID{aid}_active')
        for cid in inactive:
            aid_source.setdefault(cid, f'PubChem_AID{aid}_inactive')
        all_active.update(list(active)[:args.max_per_aid])
        all_inactive.update(list(inactive)[:args.max_per_aid])

    print(f'\nTotal unique CIDs: active={len(all_active)}, inactive={len(all_inactive)}')

    if not all_active and not all_inactive:
        print('\n⚠ No CIDs retrieved. Writing empty CSV — API may be unreachable.')
        with open(args.output, 'w', newline='') as f:
            csv.writer(f).writerow([
                'cid', 'smiles', 'label', 'source', 'data_source', 'source_target'
            ])
        return

    print(f'\nResolving CID → SMILES (may take a few minutes)...')
    all_cids = all_active | all_inactive
    smi_map = fetch_smiles_batch(all_cids)
    print(f'  Got SMILES for {len(smi_map)}/{len(all_cids)} CIDs.')

    rows = []
    for cid, smi in smi_map.items():
        label = 1 if cid in all_active else 0
        rows.append({
            'cid': cid,
            'smiles': smi,
            'label': label,
            'source': aid_source.get(cid, 'PubChem'),
            'data_source': 'PubChem_BioAssay',
            'source_target': 'NLRP3',
        })

    fieldnames = ['cid', 'smiles', 'label', 'source', 'data_source',
                  'source_target']
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        if rows:
            w.writerows(rows)

    if not rows:
        print('\n⚠ Empty result — check PubChem connectivity or SMILES-key '
              'compatibility (script now expects `SMILES`).')
        return

    n_pos = sum(1 for r in rows if r['label'] == 1)
    print(f'\n✓ Wrote {len(rows)} rows to {args.output}')
    print(f'  Actives:   {n_pos}')
    print(f'  Inactives: {len(rows) - n_pos}  ← confirmed experimental negatives!')


if __name__ == '__main__':
    main()
