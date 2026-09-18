"""Record a point-in-time health check; never restart or change training."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import os
import subprocess

ROOT = Path('/NHNHOME/data/sukim/adcl')
REPORT = ROOT / 'reports/md_a2_scene_extensions_20260918'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    receipt = json.loads((REPORT / 'launch_receipt.json').read_text())
    base = ROOT / 'work_dirs/md_a2_nominal_mh4_20260918/A2-BASE-NOM-s1'
    control = {x['step']: x for x in rows(base / 'metrics.jsonl') if x['kind'] == 'train'}
    now = datetime.datetime.now(datetime.timezone.utc)
    korea = datetime.timezone(datetime.timedelta(hours=9))
    result = dict(captured_at_utc=now.isoformat(), captured_at_kst=now.astimezone(korea).isoformat(),
        implementation_commit=receipt['implementation_commit'], status='healthy', runs=[],
        eta_warning='Measured startup throughput extrapolation, not a completion guarantee')
    common_orders = []
    for run in receipt['runs']:
        path = Path(run['run_dir'])
        manifest = json.loads((path / 'manifest.json').read_text())
        protocol = json.loads((REPORT / f"protocol_{run['arm']}_s1.json").read_text())
        train = [m for m in rows(path / 'metrics.jsonl') if m['kind'] == 'train']
        assert train[-1]['step'] >= 100
        assert manifest['git_sha'] == receipt['implementation_commit']
        if 'nonfinite_count' in manifest:
            assert manifest['nonfinite_count'] == 0
        assert manifest['status'] in ('running', 'completed')
        log = Path(run['log']).read_text()
        assert not any(marker in log for marker in ('Traceback (', 'FloatingPointError:', 'CUDA out of memory'))
        assert all(math.isfinite(value) for row in train for value in row.values()
                   if isinstance(value, float))
        os.kill(run['pid'], 0)
        env = Path(f"/proc/{run['pid']}/environ").read_bytes().split(b'\0')
        assert f"CUDA_VISIBLE_DEVICES={run['gpu']}".encode() in env
        for source, expected in protocol['source'].items():
            assert sha(ROOT / source) == expected, source
        assert protocol['initial_load']['shared_base_state_sha256'] == protocol['expected_shared_base_initial_sha256']
        assert protocol['initial_load']['all_parameters_trainable'] is True
        orders = {t['step']: t['sample_order_sha256'] for t in train}
        assert all(control[step]['sample_order_sha256'] == value for step, value in orders.items())
        common_orders.append(orders)
        last, previous = train[-1], train[-2]
        speed = (last['elapsed_seconds'] - previous['elapsed_seconds']) / (last['step'] - previous['step'])
        schedule = [i for i in range(3426, 20554, 3426)] + [20554]
        future = [step for step in schedule if step > last['step']]
        # A smoke V0 evaluation took about 30s; use 35s per evaluation/IO.
        first = now + datetime.timedelta(seconds=(future[0] - last['step']) * speed + 35)
        terminal = now + datetime.timedelta(seconds=(20554 - last['step']) * speed + 35 * len(future))
        result['runs'].append(dict(arm=run['arm'], gpu=run['gpu'], pid=run['pid'],
            step=last['step'], target_updates=20554,
            completed_manifest_nonfinite_count=manifest.get('nonfinite_count'),
            observed_log_errors=0, logged_loss_and_gradient_norms_finite=True,
            seconds_per_update=speed, expected_next_eval_kst=first.astimezone(korea).isoformat(),
            expected_terminal_kst=terminal.astimezone(korea).isoformat(),
            compared_BASE_sample_order_logs=len(orders), source_hashes_match=True,
            shared_initial_tensors_match=True, latest_train=last))
        out = REPORT / 'launch_records' / run['arm']
        out.mkdir(parents=True, exist_ok=True)
        (out / 'manifest_at_capture.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (out / 'metrics_at_capture.jsonl').write_text('\n'.join(json.dumps(t) for t in train) + '\n')
    shared_steps = set(common_orders[0]) & set(common_orders[1])
    assert all(common_orders[0][s] == common_orders[1][s] for s in shared_steps)
    result['between_arm_matching_logged_steps'] = sorted(shared_steps)
    result['gpu_state'] = subprocess.check_output(['nvidia-smi', '-i', '0,1,2,3',
        '--query-gpu=index,name,utilization.gpu,memory.used', '--format=csv,noheader'], text=True).splitlines()
    (REPORT / 'launch_health_snapshot.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
