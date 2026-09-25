"""Finish interrupted MR baseline runs after their independent suffix jobs exit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from finalize import BENCHMARKS, MODELS, REPO, live_inference, publish, verify_metrics


SEGMENTS = {
    'padt': [('padt_full', 2000), ('padt_full_resume1', 24360)],
    'unipixel': [('unipixel_full', 4101), ('unipixel_full_resume2', 22259)],
    'instructseg': [('instructseg_full', 3901), ('instructseg_full_resume2', 22459)],
    'evfsam': [('evfsam_full', 6901), ('evfsam_full_resume1', 19459)],
}


def canonical_keys(data_root):
    keys = []
    for benchmark in BENCHMARKS:
        source = json.loads((data_root / f'{benchmark}_val_sampled.json').read_text())
        for round_key in sorted(source, key=int):
            keys.extend((benchmark, int(round_key), row) for row in range(len(source[round_key])))
    if len(keys) != 26360:
        raise ValueError(f'Canonical data contains {len(keys)} instead of 26360 samples')
    return keys


def finished_job(path):
    if live_inference(path.name.split('_full')[0], path):
        return False
    status = path / 'exit_status'
    log = path / 'inference.log'
    if not status.exists() or not log.exists() or status.stat().st_mtime < log.stat().st_mtime:
        if log.exists() and time.time() - log.stat().st_mtime > 120:
            raise RuntimeError(f'{path}: no live inference and no current exit status')
        return False
    code = status.read_text().strip()
    if code != '0':
        raise RuntimeError(f'{path}: inference/conversion exited {code}; inspect inference.log')
    return True


def merge_and_score(model, root, data_root, keys):
    merged = root / f'{model}_full_merged'
    merged.mkdir(exist_ok=True)
    output = merged / 'rank0.jsonl'
    temporary = merged / 'rank0.jsonl.tmp'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    offset = 0
    with temporary.open('w') as dst:
        for directory, expected_count in SEGMENTS[model]:
            path = root / directory / 'rank0.jsonl'
            count = 0
            with path.open() as src:
                for line in src:
                    record = json.loads(line)
                    actual = (record.get('benchmark'), record.get('round'), record.get('row'))
                    if actual != keys[offset + count] or record.get('error'):
                        raise ValueError(f'{model}: wrong or failed record at global index {offset + count}: {actual}')
                    dst.write(line)
                    count += 1
            if count != expected_count:
                raise ValueError(f'{model}: {directory} has {count} records, expected {expected_count}')
            offset += count
    if offset != len(keys):
        raise ValueError(f'{model}: merged {offset} records, expected {len(keys)}')
    temporary.replace(output)
    subprocess.run([sys.executable, str(REPO / 'evaluation/mr_baselines/score.py'),
                    '--data-root', str(data_root), '--pred-dir', str(merged),
                    '--output', str(merged / 'metrics.json')], cwd=REPO, check=True)
    metrics = verify_metrics(merged / 'metrics.json')
    publish(REPO / 'evaluation/mr_benchmarks_comparison_20260924.md', MODELS[model], metrics)
    return {b: [round(100 * metrics['summary'][b]['ciou'], 2),
                round(100 * metrics['summary'][b]['giou'], 2)] for b in BENCHMARKS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', default=str(REPO / 'logs/mr_baselines_20260924'))
    ap.add_argument('--data-root', default='/volume/ybo/xyc/CycleGRPO/data/segllm_data/conversation_folder/all_data_mix_val')
    ap.add_argument('--interval', type=int, default=30)
    args = ap.parse_args()
    root, data_root = Path(args.run_root), Path(args.data_root)
    keys = canonical_keys(data_root)
    pending = set(SEGMENTS)
    state = {}
    while pending:
        for model in list(pending):
            suffix = root / SEGMENTS[model][-1][0]
            try:
                if not finished_job(suffix):
                    continue
                state[model] = {'complete': merge_and_score(model, root, data_root, keys)}
            except Exception as exc:
                state[model] = {'error': f'{type(exc).__name__}: {exc}'}
            pending.remove(model)
            (root / 'resume_finalization_status.json').write_text(json.dumps(state, indent=2) + '\n')
            print(model, state[model], flush=True)
        if pending:
            time.sleep(args.interval)
    subprocess.run(['bash', str(REPO / 'tools/gpu_power_hold.sh'), 'start'], cwd=REPO, check=True)
    if any('error' in item for item in state.values()):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
