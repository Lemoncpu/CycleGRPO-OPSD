"""Audit canonical MR files, images, and SAMTok history before baseline runs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluation.mask_protocol import parse_mask_groups, complete_mask_group_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--smoke-root', required=True)
    args = ap.parse_args()
    report = {}
    images = set()
    smoke = Path(args.smoke_root)
    smoke.mkdir(parents=True, exist_ok=True)
    for bench in ['mr_refcoco', 'mr_refcoco+', 'mr_refcocog', 'mr_paco']:
        path = Path(args.data_root) / f'{bench}_val_sampled.json'
        raw = path.read_bytes()
        data = json.loads(raw)
        counts, groups, malformed = {}, 0, 0
        for rnd, rows in data.items():
            counts[rnd] = len(rows)
            for row in rows:
                image = Path(row['image'])
                if not image.is_absolute():
                    image = Path(args.images) / str(image).removeprefix('data/coco/train2014/')
                images.add(image)
                assert len(row['target_masks']) == 1, 'Adapter requires one target RLE'
                assert row['history'][-1]['from'] == 'human'
                for turn in row['history']:
                    text = turn['value']
                    legal = parse_mask_groups(text, codebook_size=256, protocol='legacy_union')
                    groups += len(legal)
                    malformed += complete_mask_group_count(text) - len(legal)
        # Exercise every round, keeping histories/targets intact; never report smoke scores.
        (smoke / path.name).write_text(json.dumps({k: v[:1] for k, v in data.items()}))
        report[bench] = dict(samples=sum(counts.values()), per_round=counts,
                             sha256=hashlib.sha256(raw).hexdigest(),
                             legal_history_mask_groups=groups, invalid_complete_groups=malformed)
    missing = [str(p) for p in images if not p.is_file()]
    report['images'] = dict(unique=len(images), missing=missing)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
