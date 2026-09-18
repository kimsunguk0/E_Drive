"""Start the three approved runs once, with persistent logs and PID receipts."""
from pathlib import Path
import datetime
import json
import os
import subprocess

ROOT = Path('/NHNHOME/data/sukim/adcl')
REPORT = ROOT / 'reports/md_a2_nominal_mh4_20260918'
PYTHON = '/home/<B200-USER>/cv2env/bin/python'
ARMS = [(0, 'A2-FULL-NOM', 24931), (1, 'A2-BASE-NOM', 20554),
        (2, 'A2-MH4-NOM', 20554)]


def main():
    assert json.loads((REPORT / 'preflight.json').read_text())['status'] == 'passed'
    assert json.loads((REPORT / 'throughput_benchmark.json').read_text())['selected_microbatch'] == 8
    receipt = REPORT / 'launch_receipt.json'
    if receipt.exists():
        raise RuntimeError('Launch receipt already exists; inspect existing processes first')
    runtime = REPORT / 'runtime'
    runtime.mkdir(exist_ok=True)
    runs = []
    for gpu, arm, updates in ARMS:
        run = ROOT / 'work_dirs/md_a2_nominal_mh4_20260918' / f'{arm}-s1'
        log = runtime / f'{arm}-s1.log'
        if run.exists() or log.exists():
            raise RuntimeError(f'Existing run/log: {arm}')
        memory = int(subprocess.check_output([
            'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).strip())
        if memory != 0:
            raise RuntimeError(f'GPU {gpu} has {memory} MiB in use; inspect before launching')
        command = [PYTHON, '-u', str(Path(__file__).with_name('train_nominal.py')),
                   '--arm', arm, '--gpu', str(gpu), '--seed', '1',
                   '--microbatch', '8', '--run-dir', str(run)]
        runs.append(dict(arm=arm, gpu=gpu, updates=updates, run_dir=str(run),
                         log=str(log), command=command))
    state = dict(implementation_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        launched_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='launching', gpu3='reserved for validation and SIDE-SCENE follow-up', runs=[])
    # Exclusive creation prevents duplicate launches by a second invocation.
    with receipt.open('x') as stream:
        json.dump(state, stream, indent=2)
        stream.write('\n')
    for run in runs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(run['gpu']),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                   PYTHONUNBUFFERED='1')
        with open(run['log'], 'xb') as output:
            child = subprocess.Popen(run['command'], cwd=ROOT, env=env,
                                     stdin=subprocess.DEVNULL, stdout=output,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     close_fds=True)
        run['pid'] = child.pid
        state['runs'].append(run)
        tmp = receipt.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(receipt)
    state['status'] = 'spawned; verify manifests and metrics before reporting healthy'
    tmp = receipt.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(receipt)
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
