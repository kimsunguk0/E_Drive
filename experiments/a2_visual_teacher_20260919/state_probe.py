"""Frozen QREFINE state/history-information intervention; no optimizer updates."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0, str(ROOT/'experiments/a2_next_20260919'))
from common import (PARENT, load_parent, nominal, mr, NominalStatusDataset,
                    inputs, to_device, tensor_state_sha256, trainer, torch, sha)
from torch.utils.data import DataLoader
from terminal_review import metrics
import numpy as np

REPORT = ROOT/'reports/a2_visual_teacher_20260919'
W = np.array([11, 11, 5, 5, 2, 2], dtype=np.float64)/36


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    REPORT.mkdir(parents=True, exist_ok=True)
    assert not (REPORT/'state_on_off.json').exists()
    torch.set_num_threads(4)
    trainer.seed_all(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model, checkpoint = load_parent()
    before = tensor_state_sha256(model.state_dict())
    manifest = checkpoint['manifest']
    del checkpoint
    assert model.planner.config.state_on is True
    model.cuda().eval().requires_grad_(False)
    _, tune = nominal.raw_datasets(False, 1)
    data = NominalStatusDataset(mr.MotionCanvasDataset(tune, 'native'))
    loader = DataLoader(data, batch_size=8, num_workers=8, pin_memory=True, shuffle=False)
    stored = json.loads((PARENT.parent/'final_eval.json').read_text())
    records = stored['records']
    predictions = {True: [], False: []}
    rows, gt_list = [], []
    part_diff = {k: 0.0 for k in ('scene_features', 'motion_features', 'state_hat', 'history_hat')}
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for index, raw in enumerate(loader):
                x = inputs(to_device(raw, torch.device('cuda:0')))
                outputs = {}
                for enabled in (True, False):
                    model.planner.config.state_on = enabled
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        outputs[enabled] = model(**x)
                    p = outputs[enabled]['plan_abs'].float().cpu().numpy()
                    assert np.isfinite(p).all()
                    predictions[enabled].append(p)
                for key in part_diff:
                    a, b = outputs[True][key], outputs[False][key]
                    value = float((a.float()-b.float()).abs().max())
                    part_diff[key] = max(part_diff[key], value)
                    assert torch.equal(a, b), key
                rows.extend(int(v) for v in raw['row'])
                gt_list.append(raw['gt_plan'].numpy())
                del outputs
                if (index+1) % 50 == 0:
                    print(json.dumps({'rows':len(rows), 'elapsed_s':time.monotonic()-start}), flush=True)
    finally:
        model.planner.config.state_on = True
    after = tensor_state_sha256(model.state_dict())
    assert before == after
    assert rows == [r['row'] for r in records]
    gt = np.concatenate(gt_list).astype(np.float64)
    assert np.array_equal(gt, np.array([r['gt_abs_xy'] for r in records]))
    pred = {k: np.concatenate(v).astype(np.float64) for k,v in predictions.items()}
    original = np.array([r['pred_abs_xy'] for r in records], dtype=np.float64)
    replay = float(np.abs(pred[True]-original).max())
    buckets = np.array([r['bucket'] for r in records])
    sessions = np.array([r['session'] for r in records])
    dg = np.diff(np.concatenate([np.zeros((len(gt),1,2)), gt],axis=1),axis=1)
    mask = np.linalg.norm(dg,axis=-1) > .05
    stats, scores = {}, {}
    for flag in (True, False):
        name = 'ON' if flag else 'OFF'
        stats[name], scores[name] = metrics(pred[flag],gt,buckets,mask)
    assert replay <= 5e-4 and abs(stats['ON']['PREFIX']-stored['report']['official_d3']) < 1e-6
    unique = sorted(set(sessions))
    indices = [np.flatnonzero(sessions==s) for s in unique]
    delta = scores['OFF']-scores['ON']
    sums = np.array([delta[i].sum() for i in indices])
    counts = np.array([len(i) for i in indices])
    draws = np.random.default_rng(0).integers(0,len(unique),(20000,len(unique)))
    boot = sums[draws].sum(1)/counts[draws].sum(1)
    distance = float((np.linalg.norm(pred[False]-pred[True],axis=-1)@W).mean())
    np.savez_compressed(REPORT/'state_on_off_predictions.npz', row=np.array(rows),
        gt=gt, ON=pred[True], OFF=pred[False])
    result = dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint=str(PARENT), checkpoint_sha256=sha(PARENT), model_state_sha256=before,
        source_sha256=sha(Path(__file__)), config=manifest['model_config'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        row_sha256=manifest.get('eval_rows_sha256'), n=len(rows), sessions=len(unique),
        physical_gpu=args.gpu, optimizer_updates=0, parameter_buffer_hash_unchanged=before==after,
        original_flag_restored=True, replay_max_abs_m=replay, internal_max_difference=part_diff,
        numeric_information_removed='normalized predicted state/history before biased MLP; constant token remains',
        provided_status5_unchanged=True, full_lineage_used=False, models=stats,
        OFF_minus_ON=dict(PREFIX=float(delta.mean()), session_CI95=np.quantile(boot,[.025,.975]).tolist(),
            sessions_improved=int((sums<0).sum()),
            group_contribution_delta={g:float(delta[buckets==g].sum()/len(delta)) for g in sorted(set(buckets))}),
        plan_PREFIX_distance=distance, elapsed_seconds=time.monotonic()-start,
        limitation='Frozen-weight out-of-training-distribution intervention; reused V0, conditional bootstrap over 11 observed sessions. Not OFF training or organizer approval.')
    (REPORT/'state_on_off.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# QREFINE 예측 state/history 정보 ON/OFF','',
        '동일 고정 DEV 가중치의 추론 개입이다. 제공 status와 continuous motion 입력은 유지했다.','',
        '|조건|PREFIX|L2_1s|L2_2s|L2_3s|일반 주행|','|---|---:|---:|---:|---:|---:|']
    for name,m in stats.items():
        lines.append(f"|{name}|{m['PREFIX']:.9f}|{m['L2_1s']:.9f}|{m['L2_2s']:.9f}|{m['L2_3s']:.9f}|{m['groups']['nonstop']['PREFIX']:.9f}|")
    lines += ['',f"OFF−ON: {delta.mean():+.9f}. Session95%CI: {result['OFF_minus_ON']['session_CI95']}.",
        f"두 계획의 PREFIX 거리: {distance:.9f}.",
        'Scene/motion/state/history 출력은 ON/OFF 간 동일했고 parameter/buffer hash와 원래 flag를 확인했다.',
        'OFF는 MLP 입력 정보를 0으로 만든다. Bias로 생기는 constant token은 남는다.',
        '이 결과는 학습부터 OFF인 모델의 성능을 의미하지 않는다. 자동 OFF 학습이나 제거 조합 스윕은 실행하지 않는다.','']
    (REPORT/'STATE_RESULT_KO.md').write_text('\n'.join(lines))
    print('RESULT '+json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
