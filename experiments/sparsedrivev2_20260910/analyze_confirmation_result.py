"""Analyze one completed, frozen confirmation B1 evaluation; CPU arrays only.

Do not run until the root authorizes analysis of the completed held evaluation.
This is group-excluded retraining confirmation, not a historically unseen set.
No MotionDrive comparison, checkpoint search, model forward or label-cache load.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

WEIGHTS = np.array([11, 11, 5, 5, 2, 2], np.float64) / 36.
HELD_SHA = '2809febcd692040870821e2575062722226b4f58152e53b7eed27d9cc5aaa3b7'
TRAIN_SHA = '701e7ea7b76acd6b0d99845a0f400c9b65a5c470131d2d3b6324192e09c28a35'
SPLIT_SHA = '11707e7d67cdd7ca69a991abdaba336f2550e480a139e0c0fd5627cb9005d21e'
EGO_SHA = 'd35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd'
BANK_SHA = '73c29e27f321a99e6654ebdeb0877fd260c2b22983fdffce58358335bdaa207f'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def validate_receipt(receipt, expected_head_sha):
    require(receipt.get('status') == 'completed' and receipt.get('audit_only') is False,
            'A completed actual evaluation receipt is required')
    require(receipt.get('population') == 'confirmation12' and receipt.get('n') == 1728
            and receipt.get('held_evaluation_enabled') is True, 'Wrong evaluation population')
    require(receipt.get('same_forward_base_comparison') is True and receipt.get('same_candidate_set') is True,
            'The base comparison must use the same forward and candidate set')
    require(receipt.get('evaluation_batch_size') == 1
            and receipt.get('evaluation_precision') == 'bf16_base_fp32_head',
            'Expected the fixed B1 BF16-base/FP32-head evaluation')
    require(receipt.get('confirmation_binding') is not None, 'Missing frozen confirmation decision binding')
    base, relative = receipt['base'], receipt['relative']
    require(base['rows_sha256'] == HELD_SHA and base['fit_rows_sha256'] == TRAIN_SHA,
            'Confirmation row ancestry changed')
    require(base['fit_rows'] == 46170 and base['fit_scenes'] == 171
            and base['evaluation_scenes'] == 32 and base['evaluation_sessions'] == 12,
            'Confirmation group counts changed')
    require(base['bank_sha256'] == BANK_SHA == relative['bank_sha256'], 'Expected the frozen fresh train171 bank')
    require(relative['base_checkpoint_sha256'] == base['checkpoint_sha256'], 'Head/base checkpoint mismatch')
    require(relative['relative_checkpoint_sha256'] == expected_head_sha
            and relative['relative_step'] == 2000 and base['step'] == 2000,
            'Expected the preselected fixed 2000-step candidate')
    require(relative['relative_objective'] == 'soft_ce' and base['goal_mode'] == 'none'
            and relative['base_goal_mode'] == 'none' and relative['feature_goal_mode'] == 'selection',
            'Preselected method changed')


def validate_arrays(arrays):
    fields = {'rows', 'pred', 'candidate_id', 'shortlist_oracle', 'point_l2', 'error_xy', 'd3',
              'base_pred', 'base_candidate_id', 'base_point_l2', 'base_error_xy', 'base_d3'}
    require(fields <= set(arrays), 'Required head/base arrays are absent')
    rows = arrays['rows']
    require(rows.ndim == 1 and rows.dtype.kind in 'iu' and len(rows) > 0
            and (np.diff(rows) > 0).all(), 'Rows must be sorted unique integers')
    n = len(rows)
    result, validation = {}, {}
    for label, prefix in [('head', ''), ('base', 'base_')]:
        pred = np.asarray(arrays[prefix+'pred'], np.float64)
        error = np.asarray(arrays[prefix+'error_xy'], np.float64)
        recorded_point = np.asarray(arrays[prefix+'point_l2'], np.float64)
        recorded_d3 = np.asarray(arrays[prefix+'d3'], np.float64)
        ids = arrays[prefix+'candidate_id']
        require(pred.shape == error.shape == (n, 6, 2) and recorded_point.shape == (n, 6)
                and recorded_d3.shape == (n,) and ids.shape == (n,) and ids.dtype.kind in 'iu',
                f'Invalid {label} array dimensions/types')
        require(all(np.isfinite(a).all() for a in [pred, error, recorded_point, recorded_d3])
                and (ids >= 0).all(), f'Invalid {label} numeric values')
        point = np.linalg.norm(error, axis=-1)
        d3 = point @ WEIGHTS
        point_difference = float(np.abs(point-recorded_point).max())
        d3_difference = float(np.abs(d3-recorded_d3).max())
        require(max(point_difference, d3_difference) < 1e-5, f'{label} saved metric does not recompute')
        result[label] = {'pred': pred, 'error': error, 'point': point, 'd3': d3, 'ids': ids}
        validation[label] = {'point_max_abs_recompute_error': point_difference,
                             'd3_max_abs_recompute_error': d3_difference}
    implied_gt_delta = ((result['head']['pred'] - result['head']['error'])
                        - (result['base']['pred'] - result['base']['error']))
    require(np.abs(implied_gt_delta).max() < 1e-5, 'Head/base were compared against different targets')
    oracle = np.asarray(arrays['shortlist_oracle'], np.float64)
    require(oracle.shape == (n,) and np.isfinite(oracle).all() and (oracle >= 0).all(), 'Invalid shared oracle')
    require(all((oracle <= r['d3'] + 1e-5).all() for r in result.values()), 'Oracle exceeds selected cost')
    validation['same_implied_target_max_abs_error'] = float(np.abs(implied_gt_delta).max())
    return result, oracle, validation


def tails(value):
    value = np.asarray(value, np.float64)
    p95, p99 = np.quantile(value, [.95, .99])
    return {'p95': float(p95), 'p99': float(p99), 'max': float(value.max()),
            'worst5pct_mean': float(value[value >= p95].mean()),
            'worst1pct_mean': float(value[value >= p99].mean())}


def summarize(arrays, sessions, repetitions=20000, seed=0):
    values, oracle, validation = validate_arrays(arrays)
    sessions = np.asarray(sessions)
    require(sessions.shape == arrays['rows'].shape, 'Session labels do not match rows')
    unique = sorted(set(sessions.tolist()))
    counts = np.array([(sessions == s).sum() for s in unique])
    draws = np.random.default_rng(seed).integers(len(unique), size=(repetitions, len(unique)))

    def ci(value):
        totals = np.array([value[sessions == s].sum(0) for s in unique])
        numerator = totals[draws].sum(1)
        denominator = counts[draws].sum(1)
        if value.ndim > 1:
            denominator = denominator.reshape((-1,) + (1,) * (value.ndim - 1))
        return np.quantile(numerator / denominator, [.025, .975], axis=0).tolist()

    out = {'rows': len(sessions), 'session_count': len(unique), 'validation': validation,
           'shortlist_oracle_d3': float(oracle.mean()), 'methods': {}, 'sessions': {},
           'bootstrap': {'repetitions': repetitions, 'seed': seed, 'unit': 'raw session',
                         'estimator': 'equal session resampling, frame-weighted mean within each draw'}}
    for name, value in values.items():
        out['methods'][name] = {'d3': float(value['d3'].mean()), 'd3_95ci': ci(value['d3']),
            'endpoint3s_l2': float(value['point'][:, -1].mean()), 'endpoint3s_l2_95ci': ci(value['point'][:, -1]),
            'point_l2': value['point'].mean(0).tolist(), 'point_l2_95ci': ci(value['point']),
            'weighted_abs_xy': (np.abs(value['error']) * WEIGHTS[None, :, None]).sum(1).mean(0).tolist(),
            'fine_regret': float((value['d3'] - oracle).mean()),
            'tails': {'d3': tails(value['d3']), 'endpoint3s_l2': tails(value['point'][:, -1])}}
    delta = values['head']['d3'] - values['base']['d3']
    point_delta = values['head']['point'] - values['base']['point']
    out['head_minus_base'] = {'d3': float(delta.mean()), 'd3_95ci': ci(delta),
        'point_l2': point_delta.mean(0).tolist(), 'point_l2_95ci': ci(point_delta),
        'endpoint3s_l2': float(point_delta[:, -1].mean()), 'endpoint3s_l2_95ci': ci(point_delta[:, -1]),
        'fine_regret_difference': float(delta.mean()), 'improved_rows': int((delta < 0).sum()),
        'equal_d3_rows': int((delta == 0).sum()), 'changed_candidate_ids': int((values['head']['ids'] != values['base']['ids']).sum()),
        'changed_prediction_rows': int(np.any(values['head']['pred'] != values['base']['pred'], axis=(1, 2)).sum())}
    for session, count in zip(unique, counts):
        mask = sessions == session
        out['sessions'][session] = {'rows': int(count),
            'head_d3': float(values['head']['d3'][mask].mean()), 'base_d3': float(values['base']['d3'][mask].mean()),
            'head_minus_base_d3': float(delta[mask].mean()),
            'head_endpoint3s_l2': float(values['head']['point'][mask, -1].mean()),
            'base_endpoint3s_l2': float(values['base']['point'][mask, -1].mean())}
    out['head_minus_base']['improved_sessions'] = sum(r['head_minus_base_d3'] < 0 for r in out['sessions'].values())
    return out


def markdown(result):
    lines = ['# 그룹 제외 재학습: 12-session 단일 확인 결과', '',
        '고정된 배치1 평가 1회에서 동일 forward·동일 200후보의 base와 상대 head를 비교했다. '
        '12개 세션은 새 train171 모델·은행·head 적합에서 제외됐지만 과거 primary train203에 포함됐으므로 역사적으로 미노출인 독립 검증 집합은 아니다.', '',
        '| 방법 | D3 | 세션 bootstrap 95% CI | Fine regret | 3초 L2 |', '|---|---:|---|---:|---:|']
    for name in ['base', 'head']:
        m = result['methods'][name]
        lines.append(f"| {name} | {m['d3']:.6f} | [{m['d3_95ci'][0]:.6f}, {m['d3_95ci'][1]:.6f}] | {m['fine_regret']:.6f} | {m['endpoint3s_l2']:.6f} |")
    d = result['head_minus_base']
    lines += ['', f"Head−base D3 차이는 {d['d3']:+.6f}, paired 95% CI [{d['d3_95ci'][0]:+.6f}, {d['d3_95ci'][1]:+.6f}]다. "
        f"12개 중 {d['improved_sessions']}개 세션과 1,728개 중 {d['improved_rows']}개 행에서 D3가 감소했다. "
        f"공통 shortlist oracle은 {result['shortlist_oracle_d3']:.6f}이며 D3 차이는 fine regret 차이와 같다.", '',
        '세션 12개를 20,000회 복원추출(seed 0)하고 각 추출에서 행 수로 가중한 평균을 계산했다. '
        'CI는 이 그룹 확인의 변동성 지표이며 과거 방법 개발의 적응적 선택이나 미지의 도로/센서 분포를 보정하지 않는다.', '',
        '| 방법 | D3 p95 / p99 | 3초 L2 p95 / p99 |', '|---|---|---|']
    for name in ['base', 'head']:
        t = result['methods'][name]['tails']
        lines.append(f"| {name} | {t['d3']['p95']:.6f} / {t['d3']['p99']:.6f} | {t['endpoint3s_l2']['p95']:.6f} / {t['endpoint3s_l2']['p99']:.6f} |")
    lines += ['', '| Session | 행 | Base D3 | Head D3 | Head−base |', '|---|---:|---:|---:|---:|']
    for sid, m in result['sessions'].items():
        lines.append(f"| {sid} | {m['rows']} | {m['base_d3']:.6f} | {m['head_d3']:.6f} | {m['head_minus_base_d3']:+.6f} |")
    lines += ['', '평가 후 checkpoint/방법을 다시 고르지 않았다. MotionDrive의 held 결과와 비교하지 않았다. '
        '원 데이터의 미래 정답 배열은 분석기가 열지 않았으며, 평가가 저장한 오차 배열을 재계산하고 두 예측의 정답 일관성을 검사했다.', '',
        '행/후보/소스 및 평가 receipt SHA, 시간별 L2와 paired CI는 동명 JSON에 보존한다.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--evaluation-receipt', type=Path, required=True)
    parser.add_argument('--expected-head-sha256', required=True)
    parser.add_argument('--split-manifest', type=Path, required=True)
    parser.add_argument('--ego-identities', type=Path, default=Path('/tmp/pm97/data/etri/ego_cache.npz'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.with_suffix('.md').exists(), 'Output already exists')
    receipt_bytes = args.evaluation_receipt.read_bytes()
    receipt = json.loads(receipt_bytes)
    validate_receipt(receipt, args.expected_head_sha256)  # Before opening predictions.
    require(sha(args.split_manifest) == SPLIT_SHA, 'Frozen confirmation split changed')
    split = json.loads(args.split_manifest.read_text())
    require(sha(args.ego_identities) == EGO_SHA, 'Identity source changed')
    with np.load(args.ego_identities, allow_pickle=False) as data:
        # Only identifiers. Do not load fut, goal, status or any label arrays.
        scene_names = data['scenarios'].astype(str)[data['scen_idx']]
        frames = data['frame']
    expected_rows = np.flatnonzero(np.isin(scene_names, split['splits']['confirmation12'])
                                  & (frames >= 30) & (frames % 5 == 0))
    require(hashlib.sha256(expected_rows.astype('<i8').tobytes()).hexdigest() == HELD_SHA,
            'Confirmation identity rows changed')
    prediction_bytes = args.predictions.read_bytes()
    with np.load(io.BytesIO(prediction_bytes), allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    require(np.array_equal(arrays['rows'], expected_rows), 'Predictions are not the exact frozen confirmation row order')
    sessions = np.array([split['scene_to_session'][s] for s in scene_names[expected_rows]])
    require(len(expected_rows) == 1728 and len(set(sessions)) == 12, 'Wrong confirmation population')
    result = summarize(arrays, sessions)
    require(abs(result['methods']['head']['d3'] - receipt['official_d3']) < 1e-5, 'Receipt/array score mismatch')
    result['provenance'] = {'evaluation_receipt_sha256': hashlib.sha256(receipt_bytes).hexdigest(),
        'prediction_sha256': hashlib.sha256(prediction_bytes).hexdigest(), 'head_sha256': args.expected_head_sha256,
        'base_checkpoint_sha256': receipt['base']['checkpoint_sha256'], 'bank_sha256': BANK_SHA,
        'train_rows_sha256': TRAIN_SHA, 'evaluation_rows_sha256': HELD_SHA, 'split_sha256': SPLIT_SHA,
        'analysis_source_sha256': sha(__file__), 'same_forward_base_comparison': True,
        'same_candidate_set': True, 'evaluation_batch_size': 1,
        'evaluation_precision': 'bf16_base_fp32_head', 'gpu_used': False, 'raw_future_label_arrays_opened': False,
        'checkpoint_selection': 'one fixed 2000-step candidate, no post-evaluation selection',
        'scope': 'group-excluded retraining confirmation; historically used primary training groups'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    with args.output.with_suffix('.md').open('x') as stream:
        stream.write(markdown(result))
    print(json.dumps({'output': str(args.output), 'methods': result['methods'], 'head_minus_base': result['head_minus_base']}, indent=2))


if __name__ == '__main__':
    main()
