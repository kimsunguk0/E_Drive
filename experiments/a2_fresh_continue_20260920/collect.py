"""Bounded CPU-only progress/terminal collector; never starts a training job."""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import os
import subprocess
import time
import traceback
import numpy as np
from train_continue import ROOT,REPORT,RUNS,PARENT_RUN,PARENT_SHA,UPDATES,EVERY,nominal,sha

RUN=RUNS/'A2-FRESH-CONT-s1'
W=np.array([11,11,5,5,2,2],np.float64)/36


def atomic(path,v):
    p=path.with_suffix(path.suffix+'.tmp');p.write_text(json.dumps(v,indent=2)+'\n');p.replace(path)


def compare(step):
    source=RUN/f'predictions_step{step}.json'
    reference=json.loads((PARENT_RUN/'final_eval.json').read_text())
    current=json.loads(source.read_text())
    a,b=reference['records'],current['records']
    key=lambda rr:[(r['row'],r['session'],r['scenario'],r['frame']) for r in rr]
    assert key(a)==key(b) and len(a)==1998
    gt=np.array([r['gt_abs_xy'] for r in a],np.float64)
    assert np.array_equal(gt,np.array([r['gt_abs_xy'] for r in b]))
    base=np.array([r['pred_abs_xy'] for r in a],np.float64)
    pred=np.array([r['pred_abs_xy'] for r in b],np.float64)
    assert np.isfinite(pred).all()
    error=np.linalg.norm(pred-gt,axis=-1)
    score=error@W;delta=score-np.linalg.norm(base-gt,axis=-1)@W
    sessions=np.array([r['session'] for r in a]);unique=sorted(set(sessions))
    ix=[np.flatnonzero(sessions==s) for s in unique]
    sums=np.array([delta[i].sum() for i in ix]);counts=np.array([len(i) for i in ix])
    draw=np.random.default_rng(0).integers(0,len(ix),(20000,len(ix)))
    bootstrap=sums[draw].sum(1)/counts[draw].sum(1)
    d=nominal.diagnostics(b)
    assert abs(float(score.mean())-current['report']['official_d3'])<1e-6
    checkpoint=RUN/f'ckpt_step{step}.pth'
    result=dict(step=step,total_updates=20554+step,PREFIX=float(score.mean()),
        L2_1s=float(error[:,:2].mean()),L2_2s=float(error[:,:4].mean()),L2_3s=float(error.mean()),
        per_waypoint_L2=error.mean(0).tolist(),delta_vs_parent=float(delta.mean()),
        session_CI95=np.quantile(bootstrap,[.025,.975]).tolist(),sessions_improved=int((sums<0).sum()),
        diagnostics=d,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        predictions=str(source),predictions_sha256=sha(source),
        auxiliary={k:current['report'][k] for k in ('occ_iou','lane_iou','state_mae_vx_vy_ax_ay_yawrate','history_position_mae_by_offset')},
        interpretation='planned intermediate on reused DEV' if step!=UPDATES else 'primary fixed terminal on reused DEV',
        server_prediction=False)
    atomic(REPORT/f'result_step{step}.json',result)
    return result


def summarize(completed):
    reference=json.loads((PARENT_RUN/'final_eval.json').read_text())
    d0=nominal.diagnostics(reference['records'])
    r0=dict(step=0,PREFIX=reference['report']['official_d3'],delta_vs_parent=0.,diagnostics=d0,
            interpretation='existing frozen parent; no new step0 forward')
    results=[json.loads(p.read_text()) for p in sorted(REPORT.glob('result_step*.json'))]
    results.sort(key=lambda r:r['step'])
    selected=min([r0,*results],key=lambda r:r['PREFIX'])
    fields=['step','PREFIX','delta_vs_parent','nonstop','depart','steady','first2s_contribution','long_abs','lat_abs']
    with (REPORT/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in [r0,*results]:
            d=r['diagnostics']
            writer.writerow(dict(step=r['step'],PREFIX=r['PREFIX'],delta_vs_parent=r['delta_vs_parent'],
                **{g:d['groups'][g]['PREFIX'] for g in ['nonstop','depart','steady']},
                first2s_contribution=d['first2s_PREFIX_contribution'],long_abs=d['weighted_longitudinal_abs_m'],lat_abs=d['weighted_lateral_abs_m']))
    value=dict(status='completed' if completed else 'running',parent_checkpoint_sha256=PARENT_SHA,
        parent_PREFIX=r0['PREFIX'],primary_step=UPDATES,
        terminal=next((r for r in results if r['step']==UPDATES),None),
        selected_best_on_reused_V0=dict(step=selected['step'],PREFIX=selected['PREFIX'],independent_validation=False),
        planned_steps=[0,*range(EVERY,UPDATES+1,EVERY)],available_steps=[r['step'] for r in results],
        limits='same 11 DEV sessions reused; no hidden-test conversion; no automatic FULL or submission')
    atomic(REPORT/'results.json',value)
    lines=['# FRESH 낮은 LR continuation','',
        ('6,852 update 완료.' if completed else '학습 진행 중. 아래 중간점은 최종 결과가 아니다.'),
        '구조·입력·loss를 유지한 한 번의 추가 학습이다. Primary는 고정 terminal−parent다.',
        '', '|추가 step|PREFIX|부모 대비|일반 주행|출발|정지 유지|횡방향 절대오차|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in [r0,*results]:
        d=r['diagnostics'];g=d['groups']
        lines.append(f"|{r['step']}|{r['PREFIX']:.9f}|{r['delta_vs_parent']:+.9f}|{g['nonstop']['PREFIX']:.9f}|{g['depart']['PREFIX']:.9f}|{g['steady']['PREFIX']:.9f}|{d['weighted_lateral_abs_m']:.9f}|")
    lines += ['',f"예정 평가 중 최저(부모 포함): step {selected['step']}, {selected['PREFIX']:.9f}. 같은 V0에서의 선택이며 독립 검증이 아니다.",
        '첫2초·시점별 L2·종방향·인지·state/history·session CI·checkpoint SHA는 result_step*.json에 있다.',
        '현재 stop과 미래 출발은 다르다. 그룹별 손익을 감추지 않는다. DEV를 서버 점수로 환산하지 않는다.',
        '새 FULL·제출·다음 학습은 자동 시작하지 않는다.']
    (REPORT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    return value


def command(argv,cwd=ROOT):
    return subprocess.check_output(argv,cwd=cwd,text=True,stderr=subprocess.STDOUT)


def publish():
    if command(['git','diff','--cached','--name-only']).strip():
        raise RuntimeError('Concurrent index; results saved, publication paused')
    files=[str(p.relative_to(ROOT)) for p in sorted(REPORT.glob('result_step*.json'))]
    files += [str((REPORT/f).relative_to(ROOT)) for f in ['results.csv','results.json','RESULTS_KO.md']]
    command(['git','add','--',*files]);command(['git','diff','--cached','--check'])
    command(['git','commit','-m','Record fixed-budget FRESH continuation results and group tradeoffs'])
    work=command(['git','rev-parse','HEAD']).strip()
    mirror=Path('/home/<B200-USER>/edrive_mirror')
    if command(['git','status','--porcelain'],mirror).strip():raise RuntimeError('Concurrent mirror changes')
    assert command(['git','branch','--show-current'],mirror).strip()=='motiondrive-v2-20260910'
    command(['git','fetch','github','motiondrive-v2-20260910'],mirror)
    command(['git','merge','--ff-only','FETCH_HEAD'],mirror)
    command(['git','fetch',str(ROOT),work],mirror);command(['git','cherry-pick',work],mirror)
    command(['git','push','github','HEAD:motiondrive-v2-20260910'],mirror)
    remote=command(['git','rev-parse','HEAD'],mirror).strip()
    assert command(['git','ls-remote','github','refs/heads/motiondrive-v2-20260910'],mirror).split()[0]==remote
    return dict(work_commit=work,mirror_commit=remote)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--publish',action='store_true');args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    status=REPORT/'runtime/watcher_status.json'
    state=dict(status='watching',pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_started_by_collector=False)
    atomic(status,state)
    deadline=time.monotonic()+8*3600
    collected=set()
    try:
        while True:
            manifest=json.loads((RUN/'manifest.json').read_text())
            for step in range(EVERY,UPDATES+1,EVERY):
                if step not in collected and (RUN/f'predictions_step{step}.json').exists() and (RUN/f'ckpt_step{step}.pth').exists():
                    compare(step);collected.add(step)
            finished=manifest['status']=='completed' and (RUN/'experiment.json').exists()
            if finished:
                assert manifest['step']==UPDATES and manifest['nonfinite_count']==0 and len(collected)==6
            summary=summarize(finished)
            state.update(available_steps=sorted(collected),training_status=manifest['status'],
                         last_checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            atomic(status,state)
            if finished:
                if args.publish:state['publication']=publish()
                state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                atomic(status,state);return
            if manifest['status'] in ('failed','stopped'):raise RuntimeError('Training '+manifest['status'])
            proc=Path(f"/proc/{manifest['pid']}/stat")
            if not proc.exists() or proc.read_text().split()[2]=='Z':raise RuntimeError('Training process ended without completion')
            if time.monotonic()>deadline:raise TimeoutError('8h collector deadline; training untouched')
            time.sleep(30)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(status,state);raise


if __name__=='__main__':main()
