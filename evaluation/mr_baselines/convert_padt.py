"""Convert completed native PaDT predictions, preserving genuinely empty answers."""
import argparse
import json
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-dir', required=True)
    root = Path(ap.parse_args().pred_dir)
    mapping = json.loads((root / 'sample_map.json').read_text())
    completions, masks = {}, {}
    for path in root.glob('mrbench_*_pred_comp_padtpro.json'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = str(row['image_id'])
            if key in completions:
                raise ValueError(f'Duplicate completion: {key}')
            completions[key] = row['completion']
    if set(completions) != set(mapping):
        raise ValueError('PaDT completion keys do not exactly cover sample_map')
    for path in root.glob('mrbench_*_pred_results_padtpro.json'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            key = str(row['image_id'])
            if key not in mapping:
                raise ValueError(f'Unexpected predicted object: {key}')
            masks.setdefault(key, []).append(row['mask'])
    with (root / 'rank0.jsonl').open('x') as out:
        for key, (bench, rnd, index, targets) in mapping.items():
            pred = np.zeros(targets[0]['size'], dtype=bool)
            for rle in masks.get(key, []):
                decoded = mask_utils.decode(rle).astype(bool)
                if decoded.shape != pred.shape:
                    raise ValueError(f'Wrong native mask shape for {key}')
                pred |= decoded
            rle = mask_utils.encode(np.asfortranarray(pred.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('ascii')
            out.write(json.dumps(dict(benchmark=bench, round=rnd, row=index,
                prediction=rle, response=completions[key], native_mask_count=len(masks.get(key, []))))+'\n')


if __name__ == '__main__':
    main()
