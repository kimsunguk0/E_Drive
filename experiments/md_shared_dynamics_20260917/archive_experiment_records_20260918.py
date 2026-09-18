"""Archive this session's protocols, curves, evaluations and artifact identities."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT = Path('/NHNHOME/data/sukim/adcl')
DEST = ROOT / 'reports/md_shared_dynamics_20260917/records_20260918'
W = np.asarray([11, 11, 5, 5, 2, 2], np.float64) / 36


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def summarize_predictions(source):
    data = json.loads(source.read_text())
    records = data['records']
    p = np.asarray([r['pred_abs_xy'] for r in records], np.float64)
    gt = np.asarray([r['gt_abs_xy'] for r in records], np.float64)
    err = np.linalg.norm(p - gt, axis=-1)
    d3 = err @ W
    stored = np.asarray([r['d3'] for r in records])
    result = {
        'report': data['report'],
        'recomputed': {
            'n': len(records), 'L2_1s': float(err[:, :2].mean()),
            'L2_2s': float(err[:, :4].mean()), 'L2_3s': float(err.mean()),
            'PREFIX': float(d3.mean()), 'point_l2': err.mean(0).tolist(),
            'max_abs_row_difference_from_stored': float(np.abs(d3 - stored).max()),
        },
        'buckets': {}, 'source': str(source.relative_to(ROOT)),
        'source_sha256': digest(source),
    }
    if result['recomputed']['max_abs_row_difference_from_stored'] > 1e-5:
        raise ValueError(f'PREFIX replay mismatch: {source}')
    for name in sorted(set(r.get('bucket', 'missing') for r in records)):
        mask = np.asarray([r.get('bucket', 'missing') == name for r in records])
        result['buckets'][name] = {'n': int(mask.sum()), 'PREFIX': float(d3[mask].mean())}
    if all('base_abs_xy' in r for r in records):
        base = np.asarray([r['base_abs_xy'] for r in records], np.float64)
        base_d3 = np.linalg.norm(base - gt, axis=-1) @ W
        result['recomputed']['base_PREFIX'] = float(base_d3.mean())
        result['recomputed']['final_minus_base'] = float((d3 - base_d3).mean())
    return result


def paired_comparison(base_path, candidate_path):
    baseline = json.loads(base_path.read_text())['records']
    candidate = json.loads(candidate_path.read_text())['records']
    assert [(r['row'], r['scenario'], r['frame']) for r in baseline] == [
        (r['row'], r['scenario'], r['frame']) for r in candidate]
    def arrays(records):
        p = np.asarray([r['pred_abs_xy'] for r in records], np.float64)
        gt = np.asarray([r['gt_abs_xy'] for r in records], np.float64)
        return gt, np.linalg.norm(p - gt, axis=-1) @ W
    gt0, b = arrays(baseline)
    gt1, c = arrays(candidate)
    np.testing.assert_array_equal(gt0, gt1)
    sessions = np.asarray([r['session'] for r in baseline])
    names = np.unique(sessions)
    delta = c - b
    sums = np.asarray([delta[sessions == s].sum() for s in names])
    counts = np.asarray([(sessions == s).sum() for s in names])
    draw = np.random.default_rng(20260918).integers(0, len(names), (20000, len(names)))
    boot = sums[draw].sum(1) / counts[draw].sum(1)
    return {'baseline': str(base_path.relative_to(ROOT)),
            'candidate': str(candidate_path.relative_to(ROOT)),
            'n': len(b), 'sessions': len(names),
            'delta_PREFIX': float(delta.mean()),
            'relative_reduction': float(-delta.mean() / b.mean()),
            'improved_sessions': int((sums < 0).sum()),
            'session_cluster_bootstrap_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'bootstrap_repetitions': 20000, 'bootstrap_seed': 20260918,
            'limitation': 'One training seed; session resampling does not establish server or multi-seed performance.'}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    output = {'schema_version': 1, 'captured_at_utc': now,
              'source_git': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'runs': [], 'artifact_inventory': [], 'runtime_logs': [],
              'storage_policy': 'Text protocols, metrics and summaries are committed; original prediction arrays and checkpoints stay on the training server and are referenced by SHA256.'}
    roots = [ROOT / 'work_dirs/md_progress_residual_20260917',
             ROOT / 'work_dirs/md_shared_dynamics_20260917']
    for root in roots:
        for receipt in sorted(root.rglob('completed.json')):
            target = DEST / 'runs' / receipt.relative_to(ROOT / 'work_dirs')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(receipt, target)
        for metrics in sorted(root.rglob('metrics.jsonl')):
            run = metrics.parent
            relative = run.relative_to(ROOT / 'work_dirs')
            dest = DEST / 'runs' / relative
            dest.mkdir(parents=True, exist_ok=True)
            events = [json.loads(s) for s in metrics.read_text().splitlines() if s.startswith('{')]
            train = [e for e in events if e.get('kind') == 'train']
            evals = [e for e in events if e.get('kind') == 'eval']
            item = {'run': str(run.relative_to(ROOT)), 'archive': str(dest.relative_to(ROOT)),
                    'last_logged_train_step': train[-1].get('step') if train else None,
                    'saved_evaluations': [{k: e[k] for k in ('step', 'official_d3', 'base_official_d3', 'n') if k in e} for e in evals],
                    'has_final_eval': (run / 'final_eval.json').exists(),
                    'evaluation_role': 'overlapping in-fit diagnostic' if 'FULL' in run.name else
                                       'held-out V0 diagnostic; producer new107 inference remains pending' if 'OOF' in run.name else 'held-out V0'}
            for p in sorted(run.iterdir()):
                if not p.is_file():
                    continue
                if p.name == 'final_eval.json':
                    summary = summarize_predictions(p)
                    write_json(dest / 'final_eval_summary.json', summary)
                    item['final_eval_summary'] = summary
                elif p.suffix in ('.json', '.jsonl'):
                    shutil.copy2(p, dest / p.name)
                if p.suffix in ('.pth', '.npz') or p.name == 'final_eval.json':
                    spec = {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size}
                    if p.name in ('last.pth', 'final_eval.json', 'ckpt_step20554.pth', 'ckpt_step24931.pth'):
                        spec['sha256'] = digest(p)
                    output['artifact_inventory'].append(spec)
            output['runs'].append(item)
    baseline = ROOT / 'work_dirs/md_r0_reset_20260914/MR-NATIVE-s1/final_eval.json'
    write_json(DEST / 'registered_mr_final_eval_summary.json', summarize_predictions(baseline))
    shared = ROOT / 'work_dirs/md_shared_dynamics_20260917'
    output['paired_comparisons'] = {
        arm: paired_comparison(baseline, shared / (arm + '-s1') / 'final_eval.json')
        for arm in ('A2-DIRECT', 'A3-DIRECT')}
    for folder in ('reports/md_progress_residual_20260917', 'reports/md_shared_dynamics_20260917'):
        for p in sorted((ROOT / folder).glob('*.npz')):
            output['artifact_inventory'].append({'path': str(p.relative_to(ROOT)),
                                                 'bytes': p.stat().st_size, 'sha256': digest(p)})
        for p in sorted((ROOT / folder / 'runtime').glob('*.log')):
            lines = p.read_text(errors='replace').splitlines()
            parsed = []
            diagnostics = []
            for line in lines:
                try:
                    v = json.loads(line.removeprefix('PLAN '))
                    if isinstance(v, dict): parsed.append(v)
                except ValueError:
                    if any(token in line for token in ('Traceback', 'Error', 'failed', 'signal')):
                        diagnostics.append(line)
            output['runtime_logs'].append({'path': str(p.relative_to(ROOT)),
                                            'bytes': p.stat().st_size, 'sha256': digest(p),
                                            'last_structured_event': parsed[-1] if parsed else None,
                                            'diagnostic_lines': diagnostics})
    write_json(DEST / 'experiment_index.json', output)
    rows = ['# 2026-09-17/18 실행 기록 색인', '',
            '실험 설정·학습 곡선·저장 평가 요약은 각 archive 디렉터리에 있다.',
            '원본 prediction과 checkpoint는 서버에 보존하며 experiment_index.json에 경로·크기 및 주요 파일의 SHA256을 기록했다.',
            'final_eval 파일이 있다는 것만으로 원래 계획한 학습 예산을 모두 마쳤다는 뜻은 아니다. 평가 step과 마지막 학습 로그를 따로 기록한다.', '',
            '| 실행 단계 | 마지막 학습 로그 | 저장된 평가 step | PREFIX |',
            '|---|---:|---:|---:|']
    for item in output['runs']:
        report = item.get('final_eval_summary', {}).get('report', {})
        score = report.get('official_d3')
        rows.append('| ' + item['run'].removeprefix('work_dirs/') + ' | ' +
                    str(item['last_logged_train_step']) + ' | ' + str(report.get('step', '없음')) + ' | ' +
                    (f'{score:.6f}' if score is not None else '미평가') + ' |')
    rows.extend(['', 'A3-FP-VA의 마지막 학습 로그는 step 600이다. 이전 문서의 350은 중간 관측 시점이었다.',
                 'OOF-MR-T203-s1은 20,554 update와 V0 평가를 완료했다. V0 0.226444는 unseen new107의 결과가 아니다.',
                 'MR-NATIVE-FULL의 0.097245는 학습에 포함된 V0 행의 진단 수치다.', ''])
    (DEST / 'README.md').write_text('\n'.join(rows))
    print(json.dumps({'runs_archived': len(output['runs']),
                      'artifact_count': len(output['artifact_inventory']),
                      'runtime_logs_indexed': len(output['runtime_logs']),
                      'index': str((DEST / 'experiment_index.json').relative_to(ROOT))}))


if __name__ == '__main__':
    main()
