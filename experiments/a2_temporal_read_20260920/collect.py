"""CPU-only collection of the six planned evaluations; no training launcher."""
from pathlib import Path
import argparse
import csv
import datetime
import json
import os
import subprocess
import time
import traceback

import numpy as np
from train_temporal import ROOT, REPORT, RUNS, CONTROL, ARM, UPDATES, EVERY, nominal, sha

RUN=RUNS/f'{ARM}-s1'
STEPS=[*range(EVERY,UPDATES,EVERY),UPDATES]
W=np.array([11,11,5,5,2,2],np.float64)/36


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def compare(step):
    source=RUN/f'predictions_step{step}.json'
    reference_path=CONTROL/f'predictions_step{step}.json'
    assert reference_path.exists(),str(reference_path)
    reference=json.loads(reference_path.read_text())
    current=json.loads(source.read_text())
    a,b=reference['records'],current['records']
    key=lambda rr:[(r['row'],r['session'],r['scenario'],r['frame']) for r in rr]
    assert key(a)==key(b) and len(a)==1998
    gt=np.array([r['gt_abs_xy'] for r in a],np.float64)
    assert np.array_equal(gt,np.array([r['gt_abs_xy'] for r in b]))
    control=np.array([r['pred_abs_xy'] for r in a],np.float64)
    pred=np.array([r['pred_abs_xy'] for r in b],np.float64)
    assert np.isfinite(pred).all()
    error=np.linalg.norm(pred-gt,axis=-1)
    score=error@W;control_score=np.linalg.norm(control-gt,axis=-1)@W
    delta=score-control_score
    session=np.array([r['session'] for r in a]);unique=sorted(set(session))
    indices=[np.flatnonzero(session==s) for s in unique]
    sums=np.array([delta[i].sum() for i in indices]);counts=np.array([len(i) for i in indices])
    draw=np.random.default_rng(0).integers(0,len(indices),(20000,len(indices)))
    bootstrap=sums[draw].sum(1)/counts[draw].sum(1)
    assert abs(float(score.mean())-current['report']['official_d3'])<1e-6
    result=dict(step=step,PREFIX=float(score.mean()),control_same_step_PREFIX=float(control_score.mean()),
        delta_vs_matched_control=float(delta.mean()),
        L2_1s=float(error[:,:2].mean()),L2_2s=float(error[:,:4].mean()),L2_3s=float(error.mean()),
        per_waypoint_L2=error.mean(0).tolist(),
        diagnostics=nominal.diagnostics(b),control_diagnostics=nominal.diagnostics(a),
        session_CI95=np.quantile(bootstrap,[.025,.975]).tolist(),
        sessions_improved=int((sums<0).sum()),session_count=len(unique),
        checkpoint=str(RUN/f'ckpt_step{step}.pth'),checkpoint_sha256=sha(RUN/f'ckpt_step{step}.pth'),
        predictions=str(source),predictions_sha256=sha(source),
        control_predictions=str(reference_path),control_predictions_sha256=sha(reference_path),
        auxiliary={k:current['report'][k] for k in ('occ_iou','lane_iou',
            'state_mae_vx_vy_ax_ay_yawrate','history_position_mae_by_offset')},
        interpretation='primary fixed terminal' if step==UPDATES else 'planned intermediate, reused DEV',
        server_prediction=False)
    atomic(REPORT/f'result_step{step}.json',result)
    return result


def row_hashes():
    def logs(path):
        result={}
        for line in path.read_text().splitlines():
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if 'sample_order_sha256' in r:result[r['step']]=r['sample_order_sha256']
        return result
    a=logs(CONTROL/'metrics.jsonl');b=logs(RUN/'metrics.jsonl')
    shared=sorted(set(a)&set(b))
    mismatch=[s for s in shared if a[s]!=b[s]]
    return dict(logged_steps_compared=len(shared),last_compared_step=shared[-1] if shared else None,
        mismatch_steps=mismatch,all_compared_equal=not mismatch,
        terminal_unlogged_tail_note='Last row digest is at a logging step; deterministic sampler policy covers the remaining updates.')


def summarize(completed):
    rows=sorted([json.loads(p.read_text()) for p in REPORT.glob('result_step*.json')],key=lambda r:r['step'])
    selected=min(rows,key=lambda r:r['PREFIX']) if rows else None
    terminal=next((r for r in rows if r['step']==UPDATES),None)
    sample=row_hashes()
    value=dict(status='completed' if completed else 'running',arm=ARM,
        primary_step=UPDATES,matched_control='A2-FRESH-NUIM-s1 same step',
        control_terminal_PREFIX=0.1587072591966789,terminal=terminal,
        selected_best_on_reused_V0=None if selected is None else
            dict(step=selected['step'],PREFIX=selected['PREFIX'],independent_validation=False),
        planned_steps=STEPS,available_steps=[r['step'] for r in rows],sample_stream=sample,
        limits='Same 11 DEV sessions reused. Additional attention parameters are part of the tested package. No hidden-test conversion.',
        automatic_new_training=False,automatic_FULL_or_submission=False)
    atomic(REPORT/'results.json',value)
    fields=['step','PREFIX','control_same_step_PREFIX','delta_vs_matched_control',
        'nonstop','depart','steady','first2s_contribution','long_abs','lat_abs']
    with (REPORT/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in rows:
            d=r['diagnostics'];g=d['groups']
            writer.writerow({**{k:r[k] for k in fields[:4]},
                **{name:g[name]['PREFIX'] for name in ('nonstop','depart','steady')},
                'first2s_contribution':d['first2s_PREFIX_contribution'],
                'long_abs':d['weighted_longitudinal_abs_m'],'lat_abs':d['weighted_lateral_abs_m']})
    lines=['# A2 시점별 motion read 결과','',
        '20,554 update 완료.' if completed else '실행 중. 아래 예정 중간점은 최종 결과가 아니다.',
        '동일 공개 FRESH 초기값·학습 조건에서 planner의 unpooled visual motion read만 추가했다.',
        '주 비교는 20,554-update terminal끼리다. 더 오래 학습한 FRESH-CONT와 같은 예산이라고 설명하지 않는다.',
        '', '|Step|Temporal read PREFIX|동일 step FRESH|차이|일반 주행|출발|정지 유지|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        g=r['diagnostics']['groups']
        lines.append(f"|{r['step']}|{r['PREFIX']:.9f}|{r['control_same_step_PREFIX']:.9f}|{r['delta_vs_matched_control']:+.9f}|{g['nonstop']['PREFIX']:.9f}|{g['depart']['PREFIX']:.9f}|{g['steady']['PREFIX']:.9f}|")
    if selected:
        lines+=['',f"예정 평가 중 최저: step{selected['step']}, {selected['PREFIX']:.9f}. 같은 V0에서 선택한 값이며 독립 검증이 아니다."]
    if terminal:
        lines+=['',f"고정 terminal − matched control: {terminal['delta_vs_matched_control']:+.9f}, session paired95%CI {terminal['session_CI95']}."]
    lines+=['',f"Sample stream 확인: {sample['logged_steps_compared']}개 로그, 마지막 step {sample['last_compared_step']}, 불일치 {sample['mismatch_steps']}.",
        '시점별 L2·첫2초·종/횡·인지/state/history·checkpoint SHA는 result_step*.json에 있다.',
        'Attention이나 state MAE를 성능으로 대체하지 않는다. 새 학습/FULL/제출/가중치 평균은 자동 실행하지 않는다.']
    (REPORT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    return value


def command(argv,cwd=ROOT):
    return subprocess.check_output(argv,cwd=cwd,text=True,stderr=subprocess.STDOUT)


def publish():
    receipt={}
    try:
        if command(['git','diff','--cached','--name-only']).strip():
            raise RuntimeError('Concurrent staging; result publication deferred')
        files=[str(p.relative_to(ROOT)) for p in sorted(REPORT.glob('result_step*.json'))]
        files += [str((REPORT/name).relative_to(ROOT)) for name in ('results.csv','results.json','RESULTS_KO.md')]
        command(['git','add','--',*files]);command(['git','diff','--cached','--check'])
        command(['git','commit','-m','Record matched FRESH temporal-memory read results'])
        work=command(['git','rev-parse','HEAD']).strip();receipt['work_commit']=work
        mirror=Path('/home/<B200-USER>/edrive_mirror')
        if command(['git','status','--porcelain'],mirror).strip():
            raise RuntimeError('Concurrent mirror changes; result publication deferred')
        assert command(['git','branch','--show-current'],mirror).strip()=='motiondrive-v2-20260910'
        command(['git','fetch','github','motiondrive-v2-20260910'],mirror)
        command(['git','merge','--ff-only','FETCH_HEAD'],mirror)
        command(['git','fetch',str(ROOT),work],mirror);command(['git','cherry-pick',work],mirror)
        receipt['mirror_commit']=command(['git','rev-parse','HEAD'],mirror).strip()
        command(['git','push','github','HEAD:motiondrive-v2-20260910'],mirror)
        receipt['push_succeeded']=True
        # Remote verification failure is recorded separately from training.
        try:
            receipt['remote_head']=command(['git','ls-remote','github',
                'refs/heads/motiondrive-v2-20260910'],mirror).split()[0]
            receipt['remote_verified']=receipt['remote_head']==receipt['mirror_commit']
        except subprocess.CalledProcessError as exc:
            receipt['remote_verification_error']=exc.output
        receipt['status']='pushed'
    except BaseException:
        receipt.update(status='publication_incomplete',error=traceback.format_exc())
    atomic(REPORT/'completion_receipt.json',receipt)
    return receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--publish',action='store_true');args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    status=REPORT/'runtime/watcher_status.json'
    state=dict(status='watching',pid=os.getpid(),training_started_by_collector=False,
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic(status,state)
    deadline=time.monotonic()+8*3600
    collected=set()
    try:
        while True:
            path=RUN/'manifest.json'
            if not path.exists():
                if time.monotonic()>deadline:raise TimeoutError('No training manifest')
                time.sleep(30);continue
            manifest=json.loads(path.read_text())
            for step in STEPS:
                if step not in collected and (RUN/f'predictions_step{step}.json').exists() and (RUN/f'ckpt_step{step}.pth').exists():
                    compare(step);collected.add(step)
            finished=manifest['status']=='completed' and (RUN/'experiment.json').exists()
            if finished:
                assert manifest['step']==UPDATES and manifest['nonfinite_count']==0 and len(collected)==len(STEPS)
            summary=summarize(finished)
            state.update(available_steps=sorted(collected),training_status=manifest['status'],
                sample_stream=summary['sample_stream'],last_checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            atomic(status,state)
            if finished:
                state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                if args.publish:state['publication']=publish()
                atomic(status,state);return
            if manifest['status'] in ('failed','stopped'):
                raise RuntimeError('Training '+manifest['status'])
            proc=Path(f"/proc/{manifest['pid']}/stat")
            if not proc.exists() or proc.read_text().split()[2]=='Z':
                raise RuntimeError('Training process ended without completion')
            if time.monotonic()>deadline:
                raise TimeoutError('8h collection deadline; training untouched')
            time.sleep(30)
    except BaseException:
        state.update(status='collection_failed',error=traceback.format_exc())
        atomic(status,state);raise


if __name__=='__main__':main()
