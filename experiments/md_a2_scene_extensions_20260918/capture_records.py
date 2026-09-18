"""Archive smoke contracts and the completed FULL candidate without weights."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil

ROOT = Path('/NHNHOME/data/sukim/adcl')
REPORT = ROOT / 'reports/md_a2_scene_extensions_20260918'
PARENT = ROOT / 'work_dirs/md_a2_nominal_mh4_20260918'
INITIAL = '116f67a476b5ab519ec4384d6e15d5f5e36e83d1c89812f733b056ba5874593f'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def archive(run, folder, step):
    folder.mkdir(parents=True, exist_ok=True)
    manifest, experiment, evaluation = [read(run / f) for f in
        ('manifest.json', 'experiment.json', 'final_eval.json')]
    assert manifest['status'] == 'completed' and manifest['step'] == step
    assert manifest['nonfinite_count'] == 0 and len(evaluation['records']) == 1998
    assert experiment['initial_load']['shared_base_state_sha256'] == INITIAL
    assert evaluation['report']['step'] == step
    for name in ('manifest.json', 'experiment.json', 'metrics.jsonl', f'diagnostics_step{step}.json'):
        shutil.copy2(run / name, folder / name)
    write(folder / 'evaluation_report.json', evaluation['report'])
    checkpoint = run / f'ckpt_step{step}.pth'
    hashes = {f: {'path': str(run / f), 'sha256': digest(run / f)} for f in
        ('manifest.json', 'experiment.json', 'final_eval.json', 'metrics.jsonl', checkpoint.name)}
    write(folder / 'artifacts.json', hashes)
    return manifest, experiment, evaluation, hashes


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    base_metrics = [json.loads(s) for s in (PARENT / 'A2-BASE-NOM-s1/metrics.jsonl').read_text().splitlines()]
    base_first = next(m for m in base_metrics if m['kind'] == 'train')
    smoke = {'status': 'passed', 'captured_at_utc': stamp, 'runs': {},
             'warning': 'Two updates check execution, not convergence or comparative performance'}
    first_orders = []
    for arm in ('A2-SIDE-SCENE-NOM', 'A2-QREFINE-NOM'):
        run = ROOT / 'work_dirs/md_a2_scene_extensions_20260918' / f'{arm}-s1_smoke'
        m, e, v, hashes = archive(run, REPORT / 'smoke_records' / arm, 2)
        metrics = [json.loads(s) for s in (run / 'metrics.jsonl').read_text().splitlines()]
        trains = [t for t in metrics if t['kind'] == 'train']
        assert len(trains) == 2 and trains[0]['sample_order_sha256'] == base_first['sample_order_sha256']
        first_orders.append([t['sample_order_sha256'] for t in trains])
        if arm == 'A2-QREFINE-NOM':
            for key in ('total', 'plan_d3', 'occ_bce', 'lane_bce', 'history', 'state', 'motion', 'plan_interval_length'):
                assert trains[0][key] == base_first[key], key
        smoke['runs'][arm] = dict(status=m['status'], step=2, nonfinite_count=0,
            V0_rows=1998, smoke_PREFIX=v['report']['official_d3'],
            shared_initial_sha256=e['initial_load']['shared_base_state_sha256'],
            step1=trains[0], step2=trains[1], checkpoint=hashes['ckpt_step2.pth'])
    assert first_orders[0] == first_orders[1]
    smoke['sample_order_matches_between_arms_and_BASE_step1'] = True
    smoke['QREFINE_step1_all_loss_terms_equal_BASE'] = True
    write(REPORT / 'smoke_summary.json', smoke)
    full_run = PARENT / 'A2-FULL-NOM-s1'
    full_report = ROOT / 'reports/md_a2_nominal_mh4_20260918'
    m, e, v, hashes = archive(full_run, full_report / 'full_terminal_records', 24931)
    assert e['full_fit']['evaluation_is_in_fit'] is True
    write(full_report / 'full_terminal_summary.json', dict(status='completed',
        captured_at_utc=stamp, arm='A2-FULL-NOM-s1', step=24931,
        train_rows=e['train_data']['rows'], train_scenes=e['train_data']['scenes'],
        elapsed_seconds=m['elapsed_seconds'], nonfinite_count=m['nonfinite_count'],
        terminal_checkpoint=hashes['ckpt_step24931.pth'],
        diagnostic_PREFIX=v['report']['official_d3'], evaluation_is_in_fit=True,
        official_server_score=None, deployment_adapter_status='A2 raw-input adapter not yet certified',
        warning='Do not compare the FULL in-fit diagnostic with held-out DEV or official submission scores'))
    print(json.dumps({'smokes': smoke['status'], 'FULL': 'completed, in-fit evaluation only'}))


if __name__ == '__main__':
    main()
