import argparse
import math
import os
import time
import torch
import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import re
from PIL import Image
import json
import hydra

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


from torchvision.transforms.functional import to_pil_image

from projects.transformers.vq_sam2 import VQ_SAM2, VQ_SAM2Config, SAM2Config

from projects.vlm.tokenmask.evaluation.grefer import G_REFER


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

def _decode_annotation(annotation, height, width):
    segmentation = annotation.get("segmentation", []) if annotation else []
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        decoded = mask_utils.decode(rle)
    else:
        polygons = [p for p in segmentation if len(p) >= 6 and len(p) % 2 == 0]
        if not polygons:
            return np.zeros((height, width), dtype=np.uint8)
        decoded = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polygons, height, width)))
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return (decoded > 0).astype(np.uint8)


def _encode_mask(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def load_dataset(grefs_file, instances_file, image_root, split="val", output_file=None):
    """Convert official gRefCOCO annotations into inference-ready records."""
    refer_api = G_REFER(image_root, grefs_file, instances_file)

    ref_ids_val = refer_api.getRefIds(split=split)
    images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
    refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
    refer_seg_ds = {}
    refer_seg_ds["images"] = []
    loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
    for item in loaded_images:
        item = item.copy()
        item["file_name"] = os.path.join(image_root, item["file_name"])

        refer_seg_ds["images"].append(item)
    refer_seg_ds["annotations"] = refer_api.Anns  # anns_val
    img2refs = {}
    for ref in refs_val:
        image_id = ref["image_id"]
        img2refs[image_id] = img2refs.get(image_id, []) + [ref]
    refer_seg_ds["img2refs"] = img2refs


    all_items = []
    for index in range(len(refer_seg_ds["images"])):
        image_info = refer_seg_ds["images"][index]
        image_path = image_info["file_name"]
        image_id = image_info["id"]
        image_size = image_info["width"], image_info["height"]

        refs = img2refs[image_id]
        if len(refs) == 0:
            continue

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                sents.append(sent["sent"].strip().lower())
                ann_ids.append(ref["ann_id"])
        sampled_sents = sents
        sampled_ann_ids = ann_ids

        anno_masks = []
        for i, ann_id in enumerate(sampled_ann_ids):
            ann_ids = ann_id if isinstance(ann_id, list) else [ann_id]
            mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
            for sub_ann_id in ann_ids:
                if sub_ann_id == -1:
                    continue
                mask |= _decode_annotation(
                    refer_seg_ds["annotations"].get(sub_ann_id),
                    image_info["height"], image_info["width"],
                )
            anno_masks.append(mask)

        for sent, binary_mask in zip(sents, anno_masks):
            assert len(binary_mask.shape) == 2
            binary_mask = (binary_mask > 0).astype(np.uint8)
            rle = _encode_mask(binary_mask)

            all_items.append({
                "image": image_path,
                "phrase": sent,
                "segmentation": rle,
            })
    
    if output_file is None:
        raise ValueError("output_file is required")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_items, f)
    print(f"Saved {len(all_items)} GRES samples at {output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description='GRES')
    parser.add_argument(
        '--model_path',
        default="zhouyik/Qwen3-VL-4B-SAMTok-co",
        help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="Qwen/Qwen3-VL-4B-SAMTok/mask_tokenizer_256x2.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--sam2_path',
        default="Qwen/sam2.1_hiera_large.pt",
        help='sam2 model path.')
    parser.add_argument(
        '--save_dir',
        default='./results/grefcoco/',
        help='save path')
    parser.add_argument(
        '--dataset',
        default=None,
        help='Prepared JSON dataset; created with --prepare-dataset when omitted.')
    parser.add_argument('--grefs-file', default=None,
                        help='Official gRefCOCO refs JSON, e.g. grefs(unc).json.')
    parser.add_argument('--instances-file', default=None,
                        help='COCO instances JSON used by gRefCOCO.')
    parser.add_argument('--image-root', default=None,
                        help='Directory containing COCO train2014 images.')
    parser.add_argument('--split', default='val', choices=('train', 'val', 'testA', 'testB'))
    parser.add_argument('--prepare-dataset', action='store_true',
                        help='Build the inference JSON from official gRefCOCO annotations and exit.')
    parser.add_argument('--task_id', '--task-id', type=int, default=0,
                        help='Shard index for this process (0 .. num_tasks-1).')
    parser.add_argument('--num_tasks', '--num-tasks', type=int, default=1,
                        help='Total number of shards / data-parallel processes (one per GPU).')
    parser.add_argument('--gpu_id', '--gpu-id', type=int, default=-1,
                        help='CUDA device to bind this process to. Default -1 => use task_id.')
    parser.add_argument('--metric_only', '--metric-only', action='store_true',
                        help='Skip inference; just compute the metric over existing save_dir.')
    parser.add_argument('--metrics-file', default=None,
                        help='Optional JSON path for the aggregate GRES metrics.')
    args = parser.parse_args()
    return args


def load_image_with_retry(path, retries=6, base_delay=0.5):
    """Open an image, retrying transient NAS I/O failures with backoff.

    The coco images live on a slow NAS; under N concurrent shards Image.open can
    raise transient I/O errors. Retrying (instead of silently skipping) avoids
    dropping thousands of valid samples. Raises the last error if all retries fail.
    """
    last = None
    for i in range(retries):
        try:
            return Image.open(path).convert('RGB')
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(base_delay * (2 ** i))  # 0.5, 1, 2, 4, 8, 16s
    raise last


def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(mask_utils.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = mask_utils.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask


def extract_mt_token_ids_v1(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]

def extract_mt_token_ids_v2(text):
    pattern = re.compile(r'<\|mt_start\|><\|mt_(\d{4})\|><\|mt_(\d{4})\|><\|mt_end\|>')
    matches = pattern.findall(text)
    ret_list = []
    for num1, num2 in matches:
        ret_list.append(int(num1))
        ret_list.append(int(num2))
    return ret_list

def find_first_index(arr, value):
    """
    在NumPy数组中找到第一个指定值的第一个出现的索引
    
    参数:
        arr: NumPy数组
        value: 要查找的值
        
    返回:
        第一个匹配值的索引，如果没有找到则返回-1
    """
    # 使用where找到所有匹配值的索引
    indices = np.where(arr == value)[0]
    
    # 返回第一个索引，如果没有找到则返回-1
    return indices[0] if len(indices) > 0 else -1

def fix_mt_format_comprehensive(text):
    """
    全面修正 <|mt_...> 格式的函数。
    它会处理以下几种情况：
    1. 标记太少 (1个): <|mt_start|><|mt_0198|><|mt_end|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    2. 标记太少 (1个, 无end): <|mt_start|><|mt_0198|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    3. 标记太多 (3个或以上): <|mt_start|><|mt_0186|><|mt_0410|><|mt_0186|><|mt_end|> -> <|mt_start|><|mt_0186|><|mt_0410|><|mt_end|>
    4. 正确格式: <|mt_start|><|mt_0044|><|mt_0442|><|mt_end|> -> 不变
    """
    # 规则 1: 处理标记太多的情况 (3个或以上)
    # 捕获前两个，匹配掉多余的，然后用前两个重构
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)
    # 规则 2: 处理标记太少的情况 (只有1个，且有<|mt_end|>)
    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)
    # 规则 3: 处理标记太少的情况 (只有1个，且没有<|mt_end|>)
    # 使用负向前瞻确保后面不是另一个mt_token
    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text


def _iter_json_records(json_file_path):
    """Yield dict records from JSON/JSONL/concatenated JSON content."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    def _collect(obj, out):
        if isinstance(obj, dict):
            out.append(obj)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    out.append(item)

    records = []
    try:
        parsed = json.loads(content)
        _collect(parsed, records)
        return records
    except json.JSONDecodeError:
        pass

    # Fallback 1: JSONL
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
            _collect(parsed, records)
        except json.JSONDecodeError:
            continue
    if records:
        return records

    # Fallback 2: concatenated JSON objects without separators
    decoder = json.JSONDecoder()
    idx = 0
    n = len(content)
    while idx < n:
        while idx < n and content[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            parsed, end = decoder.raw_decode(content, idx)
            _collect(parsed, records)
            idx = end
        except json.JSONDecodeError:
            break
    return records


def metric(args):
    intersection = 0
    union = 0
    target_ious = []
    all_ious = []
    no_target_correct = 0
    no_target_total = 0
    target_correct = 0
    target_total = 0

    for json_file in os.listdir(args.save_dir):
        if not json_file.startswith("case_") or not json_file.endswith(".json"):
            continue
        json_file_path = os.path.join(args.save_dir, json_file)
        records = _iter_json_records(json_file_path)
        if not records:
            print(f"[warn] skip unreadable/empty file: {json_file_path}")
            continue

        for json_data in records:
            if "pred_masks" not in json_data or "gt_masks" not in json_data:
                continue

            pred_mask = rle_to_mask([json_data["pred_masks"]])[0].astype(bool)
            gt_mask = rle_to_mask([json_data["gt_masks"]])[0].astype(bool)
            inter = int(np.logical_and(pred_mask, gt_mask).sum())
            current_union = int(np.logical_or(pred_mask, gt_mask).sum())
            is_target = bool(gt_mask.any())
            is_predicted = bool(pred_mask.any())
            if is_target:
                target_total += 1
                target_correct += int(is_predicted)
                intersection += inter
                union += current_union
                iou = inter / current_union if current_union else 0.0
                target_ious.append(iou)
            else:
                no_target_total += 1
                no_target_correct += int(not is_predicted)
                iou = 1.0 if not is_predicted else 0.0
                # Match the original GRES protocol: an empty-target false
                # positive increases cIoU's union, while a correct rejection
                # has no foreground pixels to contribute.
                if is_predicted:
                    union += current_union
            all_ious.append(iou)

    metrics = {
        "samples": len(all_ious),
        "no_target_samples": no_target_total,
        "target_samples": target_total,
        "N_acc": 100.0 * no_target_correct / no_target_total if no_target_total else 0.0,
        "T_acc": 100.0 * target_correct / target_total if target_total else 0.0,
        "gIoU": 100.0 * sum(all_ious) / len(all_ious) if all_ious else 0.0,
        "cIoU": 100.0 * intersection / union if union else 0.0,
        "mIoU_target": 100.0 * sum(target_ious) / len(target_ious) if target_ious else 0.0,
    }
    if args.metrics_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_file)), exist_ok=True)
        with open(args.metrics_file, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics
        
def main():
    args = parse_args()

    if args.prepare_dataset:
        required = ("dataset", "grefs_file", "instances_file", "image_root")
        missing = [name for name in required if not getattr(args, name)]
        if missing:
            raise ValueError(f"Missing required preparation arguments: {', '.join(missing)}.")
        for name in ("grefs_file", "instances_file", "image_root"):
            path = getattr(args, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"GRES {name.replace('_', ' ')} not found: {path}")
        load_dataset(args.grefs_file, args.instances_file, args.image_root, args.split, args.dataset)
        return args

    # metric-only: skip all model loading, just score existing predictions.
    if args.metric_only:
        metric(args)
        return args

    if not args.dataset:
        raise ValueError("--dataset is required for GRES inference; run with --prepare-dataset first.")
    if not os.path.isfile(args.dataset):
        raise FileNotFoundError(f"GRES dataset not found: {args.dataset}")

    # ---- multi-GPU data parallelism: bind this process to one GPU ----
    gpu_id = args.task_id if args.gpu_id < 0 else args.gpu_id
    torch.cuda.set_device(gpu_id)            # makes every subsequent .cuda() target this GPU
    device = torch.device(f"cuda:{gpu_id}")
    print(f"[task {args.task_id}/{args.num_tasks}] using GPU {gpu_id}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto"
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    # build vq-sam2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../projects/transformers/vq_sam2/sam2/sam2_configs"))
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        sam2_config = SAM2Config(
            cfg_path="sam2.1_hiera_l.yaml",
            ckpt_path=args.sam2_path,
        )
        vq_sam2_config = VQ_SAM2Config(
            sam2_config=sam2_config,
            codebook_size=CODEBOOK_SIZE,
            codebook_depth=CODEBOOK_DEPTH,
            shared_codebook=False,
            latent_dim=256,
        )

    vq_sam2 = VQ_SAM2(vq_sam2_config).to(device).eval()

    state = torch.load(args.vq_sam2_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    vq_sam2.load_state_dict({key.removeprefix("hf_model."): value for key, value in state.items()})

    sam2_image_processor = DirectResize(1024)

    os.makedirs(args.save_dir, exist_ok=True)  # exist_ok: shards may race to create it

    all_data_dict = []
    case_id = 0
    with open(args.dataset, 'r') as f:
        json_data = json.load(f)
        for item in json_data:
            item.update({'case_id': case_id})
            all_data_dict.append(item)
            case_id += 1
    
    rows = len(all_data_dict)
    # Data-parallel sharding across `num_tasks` processes (one per GPU).
    chunk_size = math.ceil(rows / args.num_tasks)
    _start_ = args.task_id * chunk_size
    _end_ = min(_start_ + chunk_size, rows)
    print(f"[task {args.task_id}/{args.num_tasks}] rows {_start_}:{_end_} of {rows}")

    count = 0
    for data_dict in tqdm.tqdm(all_data_dict[_start_:_end_]):
        image_path = data_dict['image']
        phrase = data_dict['phrase']
        rle = data_dict['segmentation']
        case_id = data_dict['case_id']

        try:
            image = load_image_with_retry(image_path)
        except Exception as e:
            raise RuntimeError(f"Could not read GRES image {image_path}: {e}") from e

        ori_width, ori_height = image.size

        if rle['size'][0] != ori_height or rle['size'][1] != ori_width:
            raise ValueError(
                f"GRES annotation/image size mismatch for {image_path}: "
                f"RLE={rle['size']}, image={[ori_height, ori_width]}"
            )

        # gt_masks = rle_to_mask([rle])
        # output_image = visualize(image, gt_masks, ["gt"])
        # output_image.save('grefcoco_gt.jpg')
        # print("GT RLE: ", rle)

        question = f"Please segment {phrase} in this image."
        
        output_path = os.path.join(args.save_dir, f"case_{case_id}.json")
        if os.path.exists(output_path):
            print("file exists.............")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=128,
            do_sample=False,  # 关闭采样，使用贪婪解码
            top_p=1.0,  # 配合do_sample=False使用
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        # print("User: ", phrase)
        print("Assistant: ", output_text)

        quant_ids = extract_mt_token_ids_v1(output_text[0])
        if len(quant_ids) == 0:
            zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            prediction = {'gt_masks': rle, 'pred_masks': zero_mask[0]}

            # exit(0)

            with open(output_path, 'w') as f:
                json.dump(prediction, f)
            continue

        if len(quant_ids) % CODEBOOK_DEPTH != 0:
            print("FORMAT ERROR: ", output_text)
            output_text = [fix_mt_format_comprehensive(output_text[0])]
            print("FIXED OUTPUT TEXT: ", output_text)
            quant_ids = extract_mt_token_ids_v2(output_text[0])
        # assert len(quant_ids) % CODEBOOK_DEPTH == 0
        if len(quant_ids) % CODEBOOK_DEPTH != 0:
            zero_mask = np.zeros((1, ori_height, ori_width)).astype(np.uint8)
            zero_mask = mask_to_rle(zero_mask)
            prediction = {'gt_masks': rle, 'pred_masks': zero_mask[0]}

            with open(output_path, 'w') as f:
                json.dump(prediction, f)
            continue

        batch_size = len(quant_ids) // CODEBOOK_DEPTH
        remap_quant_ids = []
        for bs_id in range(batch_size):
            chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
            remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
            code1 = remap_chunk_quant_ids[0]
            code2 = remap_chunk_quant_ids[1]
            if not (code1 >= 0 and code1 < CODEBOOK_SIZE):
                continue
            if not (code2 >= 0 and code2 < CODEBOOK_SIZE):
                code2 = -1
            remap_chunk_quant_ids_error_handle = [code1, code2]
            remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

        batch_size = len(remap_quant_ids)
        if batch_size == 0:
            zero_mask = mask_to_rle(np.zeros((1, ori_height, ori_width), dtype=np.uint8))
            with open(output_path, 'w') as f:
                json.dump({'gt_masks': rle, 'pred_masks': zero_mask[0]}, f)
            continue
        sam2_image = np.array(image)
        sam2_image = sam2_image_processor.apply_image(sam2_image)
        sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
        sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
        sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

        quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

        with torch.no_grad():
            _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
        _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
        _pred_masks = _pred_masks > 0.5
        _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
        _pred_masks = np.sum(_pred_masks, axis=0).astype(np.uint8)[np.newaxis, :, :]
        _pred_masks = (_pred_masks > 0).astype(np.uint8)

        # output_image = visualize(image, _pred_masks, tags=['pred'])
        # output_image.save("grefcoco_pred.jpg")
        # exit(0)

        _pred_masks = mask_to_rle(_pred_masks)
        prediction = {'gt_masks': rle, 'pred_masks': _pred_masks[0]}
        with open(output_path, 'w') as f:
            json.dump(prediction, f)

    return args


if __name__ == "__main__":
    # load_dataset('testB')
    main()
