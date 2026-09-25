import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, DistributedSampler


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def history_text(history):
    return '\n'.join(
        ('User: ' if x.get('from') == 'human' else 'Assistant: ') + x.get('value', '').replace('<image>', '').strip()
        for x in history
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--vision-tower', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--benchmarks', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--start-id', type=int, default=0)
    args = ap.parse_args()

    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')

    import sys
    sys.path.insert(0, args.repo)
    from transformers import SiglipImageProcessor
    from instructseg.eval.seg import eval_res as native_eval
    from instructseg.utils import conversation as conversation_lib
    from instructseg.utils.builder import load_pretrained_model
    from instructseg.datasets.InstructSegDatasets import DataCollatorForCOCODatasetV2, RefCOCO_dataset

    data_args = native_eval.DataArguments(
        local_rank=rank, model_path=args.model, output_dir=args.output, vision_tower=args.vision_tower,
        json_path='/tmp/unused_mr_instructseg.json', visualize=False,
        eval_batch_size=1, dataloader_num_workers=args.workers,
    )
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model, model_args=data_args, mask_config=os.path.join(
            args.repo, 'instructseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml'),
        device=f'cuda:{local_rank}')
    device = torch.device(f'cuda:{local_rank}')
    model.to(dtype=torch.float32, device=device).eval()
    data_args.image_processor = image_processor
    data_args.is_multimodal = True
    data_args.refcoco_image_folder = '/'
    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)
    collator = DataCollatorForCOCODatasetV2(tokenizer=tokenizer, clip_image_processor=clip_image_processor)

    rows = []
    expected = {}
    sample_id = 0
    for bench in args.benchmarks:
        source = json.loads((Path(args.data_root) / f'{bench}_val_sampled.json').read_text())
        for round_key in sorted(source, key=lambda x: int(x)):
            for row_id, item in enumerate(source[round_key]):
                rle = item['target_masks'][0]
                h, w = rle['size']
                image_path = item['image']
                if not os.path.isabs(image_path):
                    image_path = os.path.join(args.images, image_path.removeprefix('./data/coco/train2014/'))
                if sample_id >= args.start_id:
                    rows.append({
                    'new_img_id': sample_id,
                    'image_info': {'file_name': image_path, 'height': h, 'width': w},
                    'anns': [{
                        'id': sample_id, 'category_id': 1, 'iscrowd': 0,
                        'bbox': [0, 0, w, h], 'segmentation': rle,
                    }],
                    'instruction': [{'sent': history_text(item['history']), 'raw': history_text(item['history'])}],
                    })
                    expected[sample_id] = (bench, int(round_key), row_id, rle)
                sample_id += 1
                if args.max_samples and sample_id >= args.max_samples:
                    break
            if args.max_samples and sample_id >= args.max_samples:
                break
        if args.max_samples and sample_id >= args.max_samples:
            break

    dataset = RefCOCO_dataset.__new__(RefCOCO_dataset)
    dataset.data = rows
    dataset.tokenizer = tokenizer
    dataset.data_args = data_args
    dataset.mask_format = 'bitmask'
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if world > 1 else None
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False, num_workers=args.workers, collate_fn=collator)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    with (Path(args.output) / f'rank{rank}.jsonl').open('w') as fout, torch.inference_mode():
        for i, batch in enumerate(loader):
            image_id = int(batch['seg_info'][0]['image_id'])
            gt_rle = expected[image_id][3]
            gt_mask = torch.from_numpy(mask_utils.decode(gt_rle).astype(np.uint8)).to(device)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch['token_refer_id'] = [x.to(device) for x in batch['token_refer_id']]
            output = model.eval_seg(
                input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                images=batch['images'].float(), images_clip=batch['images_clip'].float(),
                seg_info=batch['seg_info'], token_refer_id=batch['token_refer_id'],
                refer_embedding_indices=batch['refer_embedding_indices'], labels=batch['labels'])
            parsed = native_eval.parse_outputs(output, gt_mask)
            inter = native_eval.AverageMeter('Intersec', ':6.3f', native_eval.Summary.SUM)
            union = native_eval.AverageMeter('Union', ':6.3f', native_eval.Summary.SUM)
            giou = native_eval.AverageMeter('gIoU', ':6.3f', native_eval.Summary.SUM)
            pred, _ = native_eval.compute_metric(inter, union, giou, None, parsed)
            arr = pred[0].detach().cpu().numpy().astype(np.uint8)
            bench, rnd, row, _ = expected[image_id]
            fout.write(json.dumps({
                'benchmark': bench, 'round': rnd, 'row': row,
                'prediction': encode(arr), 'target': [gt_rle],
            }) + '\n')
            if i % 100 == 0:
                fout.flush()
                print(f'rank={rank} progress={i+1}/{len(sampler) if sampler else len(dataset)}', flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
