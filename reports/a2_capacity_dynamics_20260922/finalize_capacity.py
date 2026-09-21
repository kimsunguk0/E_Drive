"""Summarize completed, fixed-budget runs without new inference or training."""
from pathlib import Path
import datetime, hashlib, json

ROOT = Path('/NHNHOME/data/sukim/adcl')
REPORT = ROOT / 'reports/a2_capacity_dynamics_20260922'
RUNS = ROOT / 'work_dirs/a2_capacity_dynamics_20260922'
ARMS = ('C-CTRL', 'C-R101', 'C-DECSPLIT', 'C-AGENT')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, value):
    (REPORT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    summaries = {s: json.loads((REPORT / f'result_step{s}.json').read_text())
                 for s in (0, 3426, 6852, 10277)}
    result = json.loads((REPORT / 'results.json').read_text())
    receipt = json.loads((REPORT / 'runtime/main_orchestrator.json').read_text())
    assert result['status'] == 'completed' and receipt['status'] == 'DEV_completed'
    assert result['sample_stream']['steps_compared'] == 10277
    assert result['terminal_baseline_and_sample_stream_equal']
    terminal = summaries[10277]
    parent, control = terminal['parent'], terminal['arms']['C-CTRL']
    evidence, candidates = {}, []
    for arm in ARMS:
        run = RUNS / f'{arm}-s1'
        manifest = json.loads((run / 'manifest.json').read_text())
        assert manifest['status'] == 'completed' and manifest['step'] == 10277
        assert manifest['nonfinite_count'] == 0
        protocol = json.loads((run / 'experiment.json').read_text())
        assert not protocol['full_fit']['enabled']
        assert all(sha(ROOT / p) == h for p, h in protocol['source'].items())
        row = terminal['arms'][arm]
        probes = {}
        for step in (3426, 6852, 10277):
            p = run / f'train_probe_step{step}.json'
            probe = json.loads(p.read_text())
            probes[step] = {'PREFIX': probe['report']['official_d3'],
                            'n': probe['report']['n'], 'file_sha256': sha(p)}
        evidence[arm] = {
            'terminal': row, 'train_probe': probes,
            'interval_length_MAE_mean': sum(row['interval_length_MAE']) / 6,
            'interval_vector_MAE_mean': sum(row['interval_vector_MAE']) / 6,
            'nonstop_error_share': row['groups']['nonstop']['contribution'] / row['PREFIX'],
            'relative_improvement_vs_parent_percent': 100 * (parent['PREFIX'] - row['PREFIX']) / parent['PREFIX'],
            'relative_improvement_vs_control_percent': 100 * (control['PREFIX'] - row['PREFIX']) / control['PREFIX'],
            'nonfinite_count': 0, 'manifest_sha256': sha(run / 'manifest.json')}
        for step in ((10277, 6852) if arm == 'C-AGENT' else (10277,)):
            values = summaries[step]['arms'][arm]
            checkpoint = Path(values['checkpoint'])
            assert checkpoint.exists() and sha(checkpoint) == values['checkpoint_sha256']
            pred = Path(values['predictions'])
            assert sha(pred) == values['predictions_sha256']
            candidates.append({'arm': arm, 'step': step, 'scope': 'DEV',
                'selection': 'scheduled terminal' if step == 10277 else 'best of scheduled points on reused V0',
                'PREFIX': values['PREFIX'], 'checkpoint': str(checkpoint),
                'checkpoint_bytes': checkpoint.stat().st_size,
                'checkpoint_sha256': values['checkpoint_sha256'],
                'predictions': str(pred), 'predictions_sha256': values['predictions_sha256'],
                'deploy_validation': 'Architecture and five-update exports checked; terminal raw adapter/package not built.'})
    summary = {'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Existing saved predictions, manifests and checkpoints only; no new training or GPU evaluation.',
        'parent': parent, 'arms': evidence,
        'primary_lowest': {'arm': 'C-DECSPLIT', 'step': 10277, 'PREFIX': terminal['arms']['C-DECSPLIT']['PREFIX']},
        'selected_intermediate': {'arm': 'C-AGENT', 'step': 6852, 'PREFIX': summaries[6852]['arms']['C-AGENT']['PREFIX']},
        'all_treatment_vs_control_terminal_CI_include_zero': all(
            terminal['arms'][a]['delta_control']['session_CI95'][0] <= 0 <= terminal['arms'][a]['delta_control']['session_CI95'][1]
            for a in ARMS[1:]),
        'decision': 'Preserve DECSPLIT terminal, CTRL, and AGENT selected middle point. R101 has negligible incremental gain at higher cost. No demonstrated large breakthrough; no automatic extension/FULL/upload.',
        'limitation': 'Single-seed matched continuations on repeatedly used V0, 11-session conditional bootstrap. Not a test score or a universal capacity/pretraining bound.',
        'FULL_started': False, 'uploaded': False}
    write('terminal_review.json', summary)
    write('candidate_registry.json', {'artifacts': candidates, 'official_server_result_unchanged': 0.1336848279459137})
    write('completion_receipt.json', receipt)
    lines = ['# 세 축 + 공통 대조군 terminal 검토', '',
        '2026-09-22 05:45 KST까지 네 arm 모두10,277update 완료. NaN/Inf0, 전 update sample 순서 및 terminal RGB 증강/native 입력 hash 일치.',
        '기존 DEV 부모0.144597662. 이번 최저 terminal0.142952519이며 신규 구조의 대조군 대비 이득은 작다. 신규 서버 점수가 아니다.', '',
        '|Arm|Terminal PREFIX|Δ vs CTRL|Δ vs parent|고정 train256 PREFIX|',
        '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        r = evidence[arm]; t = r['terminal']
        lines.append(f"|{arm}|{t['PREFIX']:.9f}|{t['delta_control']['mean_delta']:+.9f}|{t['delta_parent']['mean_delta']:+.9f}|{r['train_probe'][10277]['PREFIX']:.9f}|")
    lines += ['', '## 해석', '',
        '- **R101**: CTRL 대비−0.000040545(0.0283%). 이번 identity-growth continuation에서 깊이 확장의 추가 이득은 거의 없다. 4090 forward49.4→64.6ms, FLOPs1444.9→2607.7G여서 현재 주력 후보로 승격하지 않는다. 다른 사전학습이나 R101 전체의 한계라는 뜻은 아니다.',
        '- **DECSPLIT**: terminal 최저, 부모 대비1.14% 개선. CTRL 대비−0.000269615(0.1883%), 11-session paired95%CI[−0.001327318,+0.000247009]로 추가 이득의 불확실성이 남는다. 최대 기여 session092를 제외하면 차이는−0.000045498이다. 작은 저비용 후보로 보존한다.',
        '- **AGENT**: terminal0.143833604는 CTRL보다+0.000611470 나쁘다. 예정 중간6,852update의0.142939500은 별도 선택 후보로 보존한다. 같은 시점 CTRL0.143643305보다 낮지만, 재사용 V0에서 고른 중간점이므로 안정된 terminal 승리나 독립 검증 성과가 아니다.',
        '- 세 treatment 모두 terminal의 CTRL 대비 session CI는0을 포함한다. 효과가 정확히0이라는 증명은 아니지만, 새 축에서 큰 개선을 확인했다는 근거도 아니다.', '',
        '## 길이·방향과 남은 오차', '',
        '|지표|부모|CTRL|DECSPLIT|', '|---|---:|---:|---:|']
    dec = terminal['arms']['C-DECSPLIT']
    for name, key in (('GT tangent 기준 종방향 절대오차','weighted_longitudinal_abs'),
                      ('GT tangent 기준 횡방향 절대오차','weighted_lateral_abs'),
                      ('곡선 집단 PREFIX,50행','curve_PREFIX')):
        lines.append(f'|{name}|{parent[key]:.9f}|{control[key]:.9f}|{dec[key]:.9f}|')
    lines += ['',
        '같은 GT interval length>0.05m mask와 공식 시점 가중치로 계산했다. 두 투영 오차의 합은 공식 PREFIX가 아니다.',
        'DECSPLIT은 횡방향/구간 방향을 개선했으나 종방향은CTRL보다 약0.19% 증가했다. 평균 구간 길이 MAE도CTRL0.069000779→DECSPLIT0.069163908로 약간 증가했다. 길이와 방향이 함께 좋아졌다고 해석하지 않는다.',
        '일반주행 평균은 부모0.147621728→CTRL0.146294830→DECSPLIT0.145972669. DECSPLIT 잔여 전체 오차의95.83%가 일반주행1875/1998행에서 나온다. 이 기여율에는 집단 빈도 효과가 포함된다.',
        '곡선 집단은모든GT구간길이>0.05m·3초총진행량≥3m·최대절대구간heading≥15도라는기존정의다. 공식좌/우회전/U-turn command 분류가 아니다.',
        'DECSPLIT의 출발0.375450139는 부모0.373569749보다 소폭 악화했고, 정지유지0.029389645는 부모0.031815293보다 개선했다. Occupancy/lane 지표는 terminal CTRL과 거의 같다.', '',
        '## 후반 곡선과 다음 판단', '',
        'Train probe는 네 arm 모두6,852→10,277에서 감소했다. DEV는CTRL/R101/DECSPLIT이 낮아졌고AGENT는0.142939500→0.143833604로 되돌아갔다. 낮아지는 마지막 구간만으로 추가 연장의 이득을 예측하지 않는다.',
        '현재 자료로 크게 남아 있는 문제는 일반주행의 진행량 정밀도다. 이번 결과는 가감속 타이밍, 데이터 오차, 특정 영상 표현 중 어느 하나를 단일 원인으로 확정하지 않는다.',
        'R101 추가 확대나agent loss계수 스윕, FULL, 공식업로드를 자동 실행하지 않는다. DECSPLIT·CTRL·AGENT중간 후보와 모든 기존 가중치를 보존한다. 0.12 서버 점수를 환산하거나 보장하지 않는다.', '',
        '원자료: results.csv, result_step3426/6852/10277.json. 가중치 위치·SHA: candidate_registry.json. 세부 시점/인지/train probe: terminal_review.json.']
    (REPORT / 'TERMINAL_REVIEW_KO.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'lowest_terminal': summary['primary_lowest'],
                      'selected_midpoint': summary['selected_intermediate'],
                      'preserved_checkpoints': len(candidates)}, indent=2))


if __name__ == '__main__':
    main()
