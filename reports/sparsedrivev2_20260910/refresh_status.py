"""Read durable receipts and atomically refresh this task's status report."""
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910')
REPORT = ROOT / 'reports/sparsedrivev2_20260910'
RUNS = ROOT / 'work_dirs/sparsedrivev2_20260910'

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

status = dict(captured_unix=time.time(), scope='B200 GPUs 0,1,4 only', runs={})
for path in sorted(RUNS.glob('*/manifest.json')):
    name = path.parent.name
    launch = read(REPORT / 'launches' / (name + '.json'))
    result = read(path.parent / 'result.json')
    progress = read(path.parent / 'progress.json')
    lines = (path.parent / 'train.jsonl')
    record = dict(gpu=launch.get('gpu') if launch else None,
        status=launch.get('status') if launch else (result or {}).get('status', 'unknown'),
        child_pid=(launch or {}).get('child_pid'), returncode=(launch or {}).get('returncode'),
        result=result, progress=progress)
    if record['child_pid']:
        cmdline = Path('/proc') / str(record['child_pid']) / 'cmdline'
        record['child_process_exists'] = cmdline.exists()
    if lines.exists():
        record['last_train'] = json.loads(lines.read_text().splitlines()[-1])
    status['runs'][name] = record
status['confirmation_pipeline'] = read(REPORT / 'confirmation_pipeline_v1/status.json')
status['primary_live_b8'] = read(REPORT / 'public_init/relative_soft_ce_live_tune1998/result.json')
status['primary_live_b1'] = read(REPORT / 'public_init/relative_soft_ce_live_tune1998_b1/result.json')
status['confirmation_candidate'] = read(REPORT / 'confirmation_candidate_ce_terminal_v1.json')
status['confirmation_result'] = read(REPORT / 'confirmation_relative_ce_b1_v1/result.json')
status['gpu_snapshot'] = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
    '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.splitlines()
temporary = REPORT / 'RUN_STATUS.json.tmp'
temporary.write_text(json.dumps(status, indent=2, allow_nan=False) + '\n')
temporary.replace(REPORT / 'RUN_STATUS.json')
print(json.dumps(dict(captured_unix=status['captured_unix'],
    runs={k: {'status':v['status'], 'returncode':v['returncode'],
        'step':(v.get('last_train') or {}).get('step'),
        'terminal_d3':((v.get('result') or {}).get('terminal') or {}).get('official_d3')}
        for k,v in status['runs'].items()},
    pipeline=(status['confirmation_pipeline'] or {}).get('phase')), indent=2))
