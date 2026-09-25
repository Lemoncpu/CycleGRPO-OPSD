import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset, Dataset


def conversation_text(history):
    parts = []
    for turn in history:
        role = 'User' if turn.get('from') == 'human' else 'Assistant'
        text = turn.get('value', '').replace('<image>', '').strip()
        parts.append(f'{role}: {text}')
    return '\n'.join(parts) + '\nPlease provide the segmentation mask for the final request.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--benchmarks', nargs='+', required=True)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--start-id', type=int, default=0)
    args = ap.parse_args()

    import torch
    from PaDT import VisonTextProcessingClass
    from utils import load_model, infer_dataset

    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    model, processor, accelerator = load_model(args.model, local_rank)
    if not hasattr(model.model, 'embed_tokens') and hasattr(model.model, 'language_model'):
        model.model.embed_tokens = model.model.language_model.embed_tokens
    processor = VisonTextProcessingClass(processor)
    import deepspeed
    with deepspeed.zero.GatheredParameters([model.model.embed_tokens.weight], enabled=True):
        embed_size = model.model.embed_tokens.weight.shape[0]
    assert embed_size > 0, "Gathered embedding table is empty"
    processor.prepare(embed_size)

    samples = []
    sample_info = {}
    next_id = 0
    for bench in args.benchmarks:
        data = json.loads((Path(args.data_root) / f'{bench}_val_sampled.json').read_text())
        for round_key in sorted(data, key=lambda x: int(x)):
            for row_index, row in enumerate(data[round_key]):
                img = row['image']
                if not os.path.isabs(img):
                    img = os.path.join(args.images, img.removeprefix('./data/coco/train2014/'))
                text = conversation_text(row['history'])
                if next_id >= args.start_id:
                    samples.append({
                    'id': next_id,
                    'image_path': [img],
                    'problem': text,
                    'prompt': [{
                        'role': 'user',
                        'content': [{'type': 'image', 'text': None}, {'type': 'text', 'text': text}],
                    }],
                    })
                    sample_info[next_id] = (bench, int(round_key), row_index, row['target_masks'])
                next_id += 1
                if args.max_samples and next_id >= args.max_samples:
                    break
            if args.max_samples and next_id >= args.max_samples:
                break
        if args.max_samples and next_id >= args.max_samples:
            break
    dataset = Dataset.from_list(samples)
    os.makedirs(args.output, exist_ok=True)
    # Official PaDT inference emits the raw generated labels, boxes and RLE masks.
    infer_dataset(
        model=model, dataset=dataset, processor=processor, accelerator=accelerator,
        output_dir=args.output, batch_size=args.batch_size,
        datasetname='mrbench', suffix='padtpro',
    )
    if accelerator.is_main_process:
        Path(args.output, 'sample_map.json').write_text(json.dumps(sample_info))
    accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
