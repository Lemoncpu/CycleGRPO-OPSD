"""Validate completed full baseline runs and publish only complete metric rows."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
MODELS = {'padt': 'PaDT Pro', 'unipixel': 'UniPixel', 'instructseg': 'InstructSeg', 'evfsam': 'EVF-SAM'}
BENCHMARKS = ['mr_refcoco', 'mr_refcoco+', 'mr_refcocog', 'mr_paco']
COUNTS = [6678, 6655, 3746, 9281]


def live_inference(model, output):
    matches = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = (proc / 'cmdline').read_bytes().decode().split('\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if not any(arg.endswith(f'/{model}_infer.py') for arg in args):
            continue
        if '--output' in args and Path(args[args.index('--output')+1]).resolve() == output.resolve():
            matches.append(int(proc.name))
    return matches


def checked(command):
    subprocess.run([sys.executable, *map(str, command)], cwd=REPO, check=True)


def verify_metrics(path):
    result = json.loads(path.read_text())
    assert result['complete'] and result['prediction_count'] == result['expected_count'] == 26360
    assert set(result['summary']) == set(BENCHMARKS)
    for bench, count in zip(BENCHMARKS, COUNTS):
        row = result['summary'][bench]
        assert row['count'] == row['expected'] == count
        assert 0 <= row['ciou'] <= 1 and 0 <= row['giou'] <= 1
    return result


def publish(table, label, metrics):
    text = table.read_text()
    rows = text.splitlines()
    matches = [i for i, row in enumerate(rows) if row.startswith(f'| {label} |')]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one table row for {label}')
    cells = ['%.2f / %.2f' % (100*metrics['summary'][b]['ciou'], 100*metrics['summary'][b]['giou']) for b in BENCHMARKS]
    rows[matches[0]] = '| ' + ' | '.join([label, *cells, 'Complete (serialized-history adapter)']) + ' |'
    # Same filesystem atomic replacement, preserving all other current rows.
    temporary = table.with_suffix('.md.tmp')
    temporary.write_text('\n'.join(rows)+'\n')
    temporary.replace(table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-root', default=str(REPO/'logs/mr_baselines_20260924'))
    ap.add_argument('--data-root', default='/volume/ybo/xyc/CycleGRPO/data/segllm_data/conversation_folder/all_data_mix_val')
    ap.add_argument('--watch', action='store_true')
    args = ap.parse_args()
    root = Path(args.run_root)
    table = REPO/'evaluation/mr_benchmarks_comparison_20260924.md'
    done, failed = set(), {}
    previous = None
    while True:
        status = {}
        for model, label in MODELS.items():
            output = root/(model+'_full')
            if model in done:
                status[model] = 'verified_complete'
                continue
            if model in failed:
                status[model] = failed[model]
                continue
            pids = live_inference(model, output)
            if pids:
                status[model] = {'running_pids': pids}
                continue
            try:
                # A ended launcher does not prove successful inference. Convert
                # only a full PaDT completion map, then re-score canonical GT.
                if model == 'padt' and not (output/'rank0.jsonl').exists():
                    checked([REPO/'evaluation/mr_baselines/convert_padt.py', '--pred-dir', output])
                checked([REPO/'evaluation/mr_baselines/score.py', '--data-root', args.data_root,
                         '--pred-dir', output, '--output', output/'metrics.json'])
                metrics = verify_metrics(output/'metrics.json')
                publish(table, label, metrics)
                done.add(model)
                status[model] = 'verified_complete'
            except Exception as exc:
                failed[model] = f'{type(exc).__name__}: {exc}'
                status[model] = failed[model]
        if status != previous:
            print(json.dumps(status, ensure_ascii=False), flush=True)
            (root/'finalization_status.json').write_text(json.dumps(status, indent=2)+'\n')
            previous = status
        if not args.watch or len(done)+len(failed) == len(MODELS):
            break
        time.sleep(15)
    if failed:
        raise SystemExit(2)
    if len(done) == len(MODELS):
        verify_metrics(root/'sa2va/metrics.json')
        print('All five models have complete full-split metrics; final GPU hold audit remains.', flush=True)


if __name__ == '__main__':
    main()
