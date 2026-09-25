import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--pred-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--benchmarks', nargs='+', default=['mr_refcoco','mr_refcoco+','mr_refcocog','mr_paco'])
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)
    preds = {}
    for path in sorted(pred_dir.glob('rank*.jsonl')):
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get('error'):
                    raise RuntimeError(f'Inference error in {path}: {r["error"]}')
                key = (r['benchmark'], int(r['round']), int(r['row']))
                if key in preds:
                    raise RuntimeError(f'duplicate prediction key: {key}')
                preds[key] = r
    per = defaultdict(lambda: {'count': 0, 'expected': 0, 'inter': 0, 'union': 0, 'iou_sum': 0.0})
    expected_keys = set()
    for bench in args.benchmarks:
        dataset = json.loads((Path(args.data_root) / f'{bench}_val_sampled.json').read_text())
        for rk, rows in dataset.items():
            round_id = int(rk)
            m = per[(bench, round_id)]
            m['expected'] = len(rows)
            for row_id, sample in enumerate(rows):
                key = (bench, round_id, row_id)
                expected_keys.add(key)
                if key not in preds:
                    continue
                pred_rle = preds[key]['prediction']
                gt_rles = sample['target_masks']
                pred = mask_utils.decode(pred_rle).astype(bool)
                if pred.ndim != 2 or list(pred.shape) != gt_rles[0]['size']:
                    raise ValueError(f'Mask shape mismatch: {key}')
                gt = np.zeros_like(pred, dtype=bool)
                for gt_rle in gt_rles:
                    gt |= mask_utils.decode(gt_rle).astype(bool)
                inter = int(np.logical_and(pred, gt).sum())
                union = int(np.logical_or(pred, gt).sum())
                m['count'] += 1
                m['inter'] += inter
                m['union'] += union
                m['iou_sum'] += inter / union if union else 1.0
    unexpected = set(preds) - expected_keys
    if unexpected:
        raise RuntimeError(f'unexpected prediction keys ({len(unexpected)}): {list(unexpected)[:5]}')
    results = {}
    complete = True
    for (bench, round_id), m in sorted(per.items()):
        m['ciou'] = m['inter'] / m['union'] if m['union'] else 0.0
        m['giou'] = m['iou_sum'] / m['count'] if m['count'] else 0.0
        complete &= m['count'] == m['expected']
        results.setdefault(bench, {})[str(round_id)] = m
    summary = {}
    for bench, rounds in results.items():
        count = sum(x['count'] for x in rounds.values())
        expected = sum(x['expected'] for x in rounds.values())
        inter = sum(x['inter'] for x in rounds.values())
        union = sum(x['union'] for x in rounds.values())
        iou_sum = sum(x['iou_sum'] for x in rounds.values())
        summary[bench] = {
            'count': count, 'expected': expected,
            'ciou': inter / union if union else 0.0,
            'giou': iou_sum / count if count else 0.0,
        }
    result = {'complete': bool(complete), 'prediction_count': len(preds), 'expected_count': len(expected_keys), 'summary': summary, 'per_round': results}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'complete': result['complete'], 'prediction_count': len(preds), 'summary': summary}, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
