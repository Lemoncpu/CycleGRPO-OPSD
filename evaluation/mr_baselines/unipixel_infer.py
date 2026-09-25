import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from pycocotools import mask as mask_utils


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def history_text(history):
    turns = []
    for item in history:
        role = 'User' if item.get('from') == 'human' else 'Assistant'
        turns.append(role + ': ' + item.get('value', '').replace('<image>', '').strip())
    return '\n'.join(turns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--benchmarks', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--start-id', type=int, default=0)
    args = ap.parse_args()

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')

    import unipixel.eval.infer_seg as native_eval
    from unipixel.model.builder import build_model
    from unipixel.utils.transforms import get_sam2_transform
    from torch.utils.data import DataLoader

    model, processor = build_model(args.model, device=f'cuda:{local_rank}', dtype='bfloat16')
    native_eval.processor = processor
    sam2_transform = get_sam2_transform(model.config.sam2_image_size)

    annos = []
    expected = {}
    sample_id = 0
    for bench in args.benchmarks:
        source = json.loads((Path(args.data_root) / f'{bench}_val_sampled.json').read_text())
        for round_key in sorted(source, key=lambda x: int(x)):
            for row_id, row in enumerate(source[round_key]):
                raw_image = row['image']
                image_path = raw_image if os.path.isabs(raw_image) else os.path.join(args.images, raw_image.removeprefix('./data/coco/train2014/'))
                masks = row['target_masks']
                h, w = masks[0]['size']
                anno = {
                    'source': 'mrbench', 'data_type': 'seg_image', 'frames': [image_path],
                    'vid': str(sample_id),
                    'samples': [{
                        'qid': sample_id, 'type': 'sentence', 'query': history_text(row['history']),
                        'mask_type': 'rle', 'masks': [[[m] for m in masks]],
                        'height': h, 'width': w,
                    }],
                }
                if sample_id >= args.start_id:
                    annos.append(anno)
                    expected[sample_id] = (bench, int(round_key), row_id, masks)
                sample_id += 1
                if args.max_samples and sample_id >= args.max_samples:
                    break
            if args.max_samples and sample_id >= args.max_samples:
                break
        if args.max_samples and sample_id >= args.max_samples:
            break

    shard = annos[rank::world]
    dataset = native_eval.EvalDataset(shard, sam2_transform, sample_frames=1)
    loader = DataLoader(dataset, collate_fn=native_eval.collate, num_workers=args.workers)
    os.makedirs(args.output, exist_ok=True)
    output_path = Path(args.output) / f'rank{rank}.jsonl'
    with output_path.open('w') as fout:
        for i, data in enumerate(loader):
            anno, question, frames = data.pop('anno'), data.pop('question'), data.pop('raw_frames')
            data = data.to(next(model.parameters()).device)
            data['frames'] = [data['frames'][0].to(model.sam2.dtype)]
            sample = anno['samples'][0]
            try:
                # Recent Transformers passes a DynamicCache on the first forward,
                # so the native model's empty-cache branch may not initialize seg.
                model.seg = []
                with torch.inference_mode():
                    out_ids = model.generate(
                        **data, do_sample=False, temperature=None, top_k=None,
                        top_p=None, repetition_penalty=None, max_new_tokens=512)
                # A completed generation without a segmentation token is a real
                # empty prediction; runtime exceptions still terminate the run.
                pred = (model.seg[0][0][0].detach().bool().cpu().numpy()
                        if len(model.seg) else np.zeros((sample['height'], sample['width']), dtype=bool))
                raw_id = int(anno['vid'])
                bench, rnd, row_id, targets = expected[raw_id]
                fout.write(json.dumps({
                    'id': raw_id, 'benchmark': bench, 'round': rnd, 'row': row_id,
                    'prediction': encode(pred), 'target': targets,
                    'native_mask_count': len(model.seg),
                    'response': processor.decode(out_ids[0, data.input_ids.size(1):], clean_up_tokenization_spaces=False),
                }) + '\n')
            except Exception:
                fout.flush()
                raise
            if i % 100 == 0:
                fout.flush()
                print(f'rank={rank} progress={i+1}/{len(dataset)}', flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
