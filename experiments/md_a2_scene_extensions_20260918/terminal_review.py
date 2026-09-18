"""Compare frozen terminal predictions; no fitting, inference or model selection."""
from pathlib import Path
import datetime
import json
import numpy as np
from capture_records import archive, digest, ROOT, REPORT

W = np.asarray([11, 11, 5, 5, 2, 2], dtype=np.float64) / 36
CURRENT = ('A2-BASE-NOM', 'A2-MH4-NOM', 'A2-SIDE-SCENE-NOM', 'A2-QREFINE-NOM')
PRIOR = 'A2-PRIOR-NOMINAL-EVAL'


def load(path):
    return json.loads(path.read_text())


def metrics(pred, gt, buckets, gt_valid):
    error = np.linalg.norm(pred - gt, axis=-1)
    score = error @ W
    dg = np.diff(np.concatenate([np.zeros((len(gt), 1, 2)), gt], axis=1), axis=1)
    tangent = dg / np.maximum(np.linalg.norm(dg, axis=-1)[..., None], 1e-8)
    delta = pred - gt
    vw = gt_valid * W[None]
    long = np.abs((delta * tangent).sum(-1))
    lat = np.abs(delta[..., 0] * tangent[..., 1] - delta[..., 1] * tangent[..., 0])
    groups = {b: {'n': int((buckets == b).sum()), 'PREFIX': float(score[buckets == b].mean()),
                 'overall_contribution': float(score[buckets == b].sum() / len(score))}
              for b in sorted(set(buckets))}
    # Geometric bins, not semantic command classes. Includes bends and lane changes.
    turn_group = np.where(gt[:, -1, 1] >= 2, 'positive_y_ge2m',
                         np.where(gt[:, -1, 1] <= -2, 'negative_y_le_minus2m', 'other'))
    turn = {b: {'n': int((turn_group == b).sum()), 'PREFIX': float(score[turn_group == b].mean())}
            for b in sorted(set(turn_group))}
    return dict(PREFIX=float(score.mean()), L2_1s=float(error[:, :2].mean()),
        L2_2s=float(error[:, :4].mean()), L2_3s=float(error.mean()),
        per_waypoint_L2=error.mean(0).tolist(),
        first2s_PREFIX_contribution=float((error[:, :4] * W[None, :4]).sum(1).mean()),
        weighted_longitudinal_abs_m=float((long * vw).sum() / vw.sum()),
        weighted_lateral_abs_m=float((lat * vw).sum() / vw.sum()),
        groups=groups, geometric_turn_bins=turn), score


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    ref_keys = ref_gt = ref_buckets = sessions = None
    models, predictions, scores, orders = {}, {}, {}, {}
    for arm in CURRENT:
        folder = ('md_a2_nominal_mh4_20260918' if arm in CURRENT[:2]
                  else 'md_a2_scene_extensions_20260918')
        run = ROOT / 'work_dirs' / folder / f'{arm}-s1'
        manifest, experiment, evaluation = [load(run / name) for name in
            ('manifest.json', 'experiment.json', 'final_eval.json')]
        assert manifest['status'] == 'completed' and manifest['step'] == 20554
        assert manifest['nonfinite_count'] == 0 and len(evaluation['records']) == 1998
        assert experiment['initial_load']['shared_base_state_sha256'] == experiment['expected_shared_base_initial_sha256']
        assert experiment['recipe']['updates'] == 20554 and experiment['recipe']['batch'] == 16
        assert not experiment['full_fit']['enabled']
        records = evaluation['records']
        keys = [(r['row'], r['session'], r['scenario'], r['frame']) for r in records]
        gt = np.asarray([r['gt_abs_xy'] for r in records], dtype=np.float64)
        buckets = np.asarray([r['bucket'] for r in records])
        pred = np.asarray([r['pred_abs_xy'] for r in records], dtype=np.float64)
        assert np.isfinite(pred).all()
        if ref_keys is None:
            ref_keys, ref_gt, ref_buckets = keys, gt, buckets
            sessions = np.asarray([r['session'] for r in records])
            dg = np.diff(np.concatenate([np.zeros((len(gt), 1, 2)), gt], axis=1), axis=1)
            gt_valid = np.linalg.norm(dg, axis=-1) > .05
        else:
            assert keys == ref_keys and np.array_equal(gt, ref_gt) and np.array_equal(buckets, ref_buckets)
        rows = [json.loads(s) for s in (run / 'metrics.jsonl').read_text().splitlines()]
        orders[arm] = {r['step']: r['sample_order_sha256'] for r in rows if r['kind'] == 'train'}
        result, scores[arm] = metrics(pred, gt, buckets, gt_valid)
        assert abs(result['PREFIX'] - evaluation['report']['official_d3']) < 1e-7
        predictions[arm] = pred
        models[arm] = dict(status=manifest['status'], step=20554, nonfinite_count=0,
            run_dir=str(run), evaluation_role='held-out DEV V0',
            completed_manifest_mtime_kst=datetime.datetime.fromtimestamp(
                (run / 'manifest.json').stat().st_mtime, datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
            elapsed_seconds=manifest['elapsed_seconds'], initial_parameter_count=manifest['initial_parameter_count'],
            manifest_reported_PREFIX=evaluation['report']['official_d3'], **result,
            state_vx_MAE=evaluation['report']['state_mae_vx_vy_ax_ay_yawrate'][0],
            occupancy_IoU=evaluation['report']['occ_iou'], lane_IoU=evaluation['report']['lane_iou'],
            learning_curve=[{'step': r['step'], 'PREFIX': r['official_d3']}
                            for r in rows if r['kind'] == 'eval'],
            prediction_file_sha256=digest(run / 'final_eval.json'))
        if arm in CURRENT[2:]:
            archive(run, REPORT / 'terminal_records' / arm, 20554)
    assert all(orders[a] == orders[CURRENT[0]] for a in CURRENT)
    old_path = ROOT / 'reports/md_a2_deploy_status_20260918/paired_status_eval.npz'
    old = np.load(old_path, allow_pickle=False)
    lookup = {int(row): i for i, row in enumerate(old['row'])}
    assert len(lookup) == 1998
    index = [lookup[k[0]] for k in ref_keys]
    assert np.array_equal(old['gt'][index], ref_gt)
    pred = old['plan_nominal'][index]
    result, scores[PRIOR] = metrics(pred, ref_gt, ref_buckets, gt_valid)
    assert abs(result['PREFIX'] - .1644553140831734) < 1e-12
    models[PRIOR] = dict(evaluation_role='prior A2 trained with real-timestamp status, evaluated with deployment nominal status',
        pure_architecture_control=False, source=str(old_path), source_sha256=digest(old_path), **result)
    unique = sorted(set(sessions.tolist()))
    session_indices = [np.flatnonzero(sessions == session) for session in unique]
    counts = np.asarray([len(i) for i in session_indices])
    draws = np.random.default_rng(0).integers(0, len(unique), size=(20000, len(unique)))
    def compare(base, arm):
        delta = scores[arm] - scores[base]
        sums = np.asarray([delta[i].sum() for i in session_indices])
        boot = sums[draws].sum(1) / counts[draws].sum(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return dict(delta=float(delta.mean()), relative_percent=float(100 * delta.mean() / scores[base].mean()),
            session_cluster_bootstrap_ci95=[float(lo), float(hi)], ci_includes_zero=bool(lo <= 0 <= hi),
            sessions_improved=int((sums < 0).sum()), sessions=len(unique),
            per_session_delta={s: float(delta[i].mean()) for s, i in zip(unique, session_indices)},
            group_mean_deltas={b: float(delta[ref_buckets == b].mean()) for b in sorted(set(ref_buckets))},
            group_overall_contribution_deltas={b: float(delta[ref_buckets == b].sum() / len(delta))
                                             for b in sorted(set(ref_buckets))})
    pairs = [('A2-BASE-NOM', 'A2-SIDE-SCENE-NOM'), ('A2-BASE-NOM', 'A2-QREFINE-NOM'),
             ('A2-MH4-NOM', 'A2-QREFINE-NOM'), (PRIOR, 'A2-QREFINE-NOM')]
    comparisons = {f'{arm} minus {base}': compare(base, arm) for base, arm in pairs}
    payload = dict(captured_at_utc=now.isoformat(), frozen_terminal_step=20554,
        row_count=1998, session_count=len(unique), models=models, comparisons=comparisons,
        contracts=dict(same_rows_gt_buckets=True, all_logged_sample_orders_equal=True,
            matching_logged_steps=len(orders[CURRENT[0]]), same_initial_shared_tensors=True,
            same_nominal_provided_status_policy=True, FULL_excluded_from_DEV_ranking=True),
        bootstrap=dict(unit='session', resamples=20000, seed=0,
            statistic='row-weighted paired mean after sampling 11 whole sessions with replacement',
            limitation='11 observed sessions only; does not estimate training-seed variance or hidden-test performance'),
        limitations=['Prior A2 is a candidate benchmark, not a pure architecture control: its training status producer differs.',
            'Longitudinal and lateral projections are not additive components of PREFIX.',
            'The late learning curve uses a cosine LR decaying to zero, so flatness alone does not prove a model capacity ceiling.',
            'These are local held-out scores, not official server submissions.'],
        decision=dict(SIDE='Do not promote this recipe to FULL or extend it automatically; preserve its artifacts.',
            QREFINE='Preserve as the numerical best of these DEV candidates; practical improvement over prior A2 is tiny.',
            next_priority='Preserve the completed single-head A2 FULL and establish raw-input deployment parity before submission.',
            new_training_started=False))
    (REPORT / 'terminal_summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'models': {a: {'PREFIX': m['PREFIX'], 'first2s': m['first2s_PREFIX_contribution'],
                                  'nonstop': m['groups']['nonstop']['PREFIX']} for a, m in models.items()},
                      'comparisons': comparisons, 'contracts': payload['contracts']}, indent=2))


if __name__ == '__main__':
    main()
