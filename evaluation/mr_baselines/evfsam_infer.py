import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def conversation_text(history):
    return '\n'.join(
        ('User: ' if turn.get('from') == 'human' else 'Assistant: ')
        + turn.get('value', '').replace('<image>', '').strip()
        for turn in history
    )


class MRDataset(Dataset):
    def __init__(self, samples, native, transform):
        self.samples = samples
        self.native = native
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        meta = self.samples[i]
        image = cv2.imread(meta['image_path'])
        if image is None:
            raise FileNotFoundError(meta['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.shape[:2]
        image_evf = self.native.image_preprocessor(image)
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.native.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        gt = mask_utils.decode(meta['target']).astype(np.uint8)
        gt = torch.from_numpy(gt[None])
        label = torch.ones(original) * 255
        return (
            meta['image_path'], image_sam, image_evf, gt, label, resize,
            [meta['text']], True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--benchmarks', nargs='+', required=True)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--start-id', type=int, default=0)
    args = ap.parse_args()

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')

    from transformers import AutoTokenizer
    from model.evf_sam import EvfSamModel
    from model.segment_anything.utils.transforms import ResizeLongestSide
    from utils.dataset import ValDataset, collate_fn
    from functools import partial

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='right', use_fast=False)
    model = EvfSamModel.from_pretrained(args.model, low_cpu_mem_usage=True, torch_dtype=torch.float16).cuda(local_rank).eval()

    native = ValDataset.__new__(ValDataset)
    native.model_type = 'ori'
    native.pixel_mean = ValDataset.pixel_mean
    native.pixel_std = ValDataset.pixel_std
    native.img_size = ValDataset.img_size
    native.ignore_label = ValDataset.ignore_label
    native.image_preprocessor = transforms.Compose([
        transforms.ToTensor(), transforms.Resize((224, 224), interpolation=3, antialias=None),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])
    native.preprocess = ValDataset.preprocess.__get__(native, ValDataset)
    transform = ResizeLongestSide(1024)

    samples = []
    sample_id = 0
    for bench in args.benchmarks:
        source = json.loads((Path(args.data_root) / f'{bench}_val_sampled.json').read_text())
        for round_key in sorted(source, key=lambda x: int(x)):
            for row_id, row in enumerate(source[round_key]):
                image_path = row['image'] if os.path.isabs(row['image']) else os.path.join(args.images, row['image'].removeprefix('./data/coco/train2014/'))
                if sample_id >= args.start_id:
                    samples.append({
                    'benchmark': bench, 'round': int(round_key), 'row': row_id,
                    'image_path': image_path, 'text': conversation_text(row['history']),
                    'target': row['target_masks'][0],
                    })
                sample_id += 1
                if args.max_samples and sample_id >= args.max_samples:
                    break
            if args.max_samples and sample_id >= args.max_samples:
                break
        if args.max_samples and sample_id >= args.max_samples:
            break

    shard = samples[rank::world]
    data = MRDataset(shard, native, transform)
    loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4,
                        collate_fn=partial(collate_fn, tokenizer=tokenizer, local_rank=local_rank))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) / f'rank{rank}.jsonl'
    with output_path.open('w') as fout, torch.inference_mode():
        for i, batch in enumerate(loader):
            # collate preserves the single sample's target mask and source text
            batch['images'] = batch['images'].cuda(local_rank).half()
            batch['images_evf'] = batch['images_evf'].cuda(local_rank).half()
            batch['input_ids'] = batch['input_ids'].cuda(local_rank)
            batch['attention_masks'] = batch['attention_masks'].cuda(local_rank)
            batch['offset'] = batch['offset'].cuda(local_rank)
            batch['masks_list'] = [x.cuda(local_rank) for x in batch['masks_list']]
            batch['label_list'] = [x.cuda(local_rank) for x in batch['label_list']]
            output = model(
                images=batch['images'], images_evf=batch['images_evf'], input_ids=batch['input_ids'],
                attention_masks=batch['attention_masks'], offset=batch['offset'],
                masks_list=batch['masks_list'], label_list=batch['label_list'],
                resize_list=batch['resize_list'], inference=True)
            pred = (output['pred_masks'][0][0] > 0).cpu().numpy().astype(np.uint8)
            meta = shard[i]
            fout.write(json.dumps({
                'benchmark': meta['benchmark'], 'round': meta['round'], 'row': meta['row'],
                'prediction': encode(pred), 'target': [meta['target']],
            }) + '\n')
            if i % 100 == 0:
                fout.flush()
                print(f'rank={rank} progress={i+1}/{len(data)}', flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
