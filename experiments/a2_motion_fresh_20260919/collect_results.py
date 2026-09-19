"""Bounded CPU-only terminal collector. Never trains, promotes FULL or submits."""
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

from train_experiment import ROOT, REPORT, RUNS, CONTROL, ARMS, nominal, sha

WEIGHTS=np.array([11,11,5,5,2,2],dtype=np.float64)/36


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def train_logs(run):
    return {r['step']:r for r in map(json.loads,(run/'metrics.jsonl').read_text().splitlines())
            if r.get('kind')=='train'}


def collect(arm):
    run=RUNS/f'{arm}-s1'
    manifest=json.loads((run/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['step']==20554
    assert manifest['nonfinite_count']==0
    reference=json.loads((CONTROL/'final_eval.json').read_text())
    evaluation=json.loads((run/'final_eval.json').read_text())
    a,b=reference['records'],evaluation['records']
    keys=lambda rr:[(r['session'],r['scenario'],int(r['frame'])) for r in rr]
    assert keys(a)==keys(b)
    gt=np.asarray([r['gt_abs_xy'] for r in a],dtype=np.float64)
    assert np.array_equal(gt,np.asarray([r['gt_abs_xy'] for r in b],dtype=np.float64))
    base=np.asarray([r['pred_abs_xy'] for r in a],dtype=np.float64)
    pred=np.asarray([r['pred_abs_xy'] for r in b],dtype=np.float64)
    error=np.linalg.norm(pred-gt,axis=-1);scores=error@WEIGHTS
    delta=scores-np.linalg.norm(base-gt,axis=-1)@WEIGHTS
    sessions=np.asarray([r['session'] for r in a]);unique=sorted(set(sessions))
    ix=[np.flatnonzero(sessions==s) for s in unique]
    sums=np.asarray([delta[x].sum() for x in ix]);counts=np.asarray([len(x) for x in ix])
    draws=np.random.default_rng(0).integers(0,len(ix),(20000,len(ix)))
    boots=sums[draws].sum(1)/counts[draws].sum(1)
    logs,control_logs=train_logs(run),train_logs(CONTROL)
    assert set(logs)==set(control_logs)
    matched=all(logs[s]['sample_order_sha256']==control_logs[s]['sample_order_sha256']
                and logs[s]['lr']==control_logs[s]['lr'] for s in logs)
    assert matched,'Matched control row/LR stream mismatch'
    checkpoint=run/'ckpt_step20554.pth'
    protocol=json.loads((run/'experiment.json').read_text())
    result={'arm':arm,'status':'completed','terminal_updates':20554,
        'PREFIX':float(scores.mean()),'L2_1s':float(error[:,:2].mean()),
        'L2_2s':float(error[:,:4].mean()),'L2_3s':float(error.mean()),
        'per_waypoint_L2':error.mean(0).tolist(),
        'control_PREFIX':float((np.linalg.norm(base-gt,axis=-1)@WEIGHTS).mean()),
        'delta_vs_control':float(delta.mean()),
        'session_CI95':np.quantile(boots,[.025,.975]).tolist(),
        'sessions_improved':int((sums<0).sum()),'session_count':len(ix),
        'diagnostics':nominal.diagnostics(b),
        'control_diagnostics':nominal.diagnostics(a),
        'auxiliary':{k:evaluation['report'][k] for k in
            ('occ_iou','lane_iou','state_mae_vx_vy_ax_ay_yawrate','history_position_mae_by_offset')},
        'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'predictions':str(run/'final_eval.json'),'predictions_sha256':sha(run/'final_eval.json'),
        'source_commit':manifest['git_sha'],'initial_state_sha256':manifest['initial_model_state_sha256'],
        'protocol_source':protocol['source'],'elapsed_seconds':manifest['elapsed_seconds'],
        'control_stream_check':{'all_logged_hashes_and_LRs_equal':matched,'last_logged_step':max(logs),
            'scope':'All logged rows through20550; last4 updates follow same deterministic sampler without a separate terminal rolling hash'},
        'validation_limit':'Repeated DEV use, paired bootstrap conditional on same11 sessions; no server score inference',
        'initialization_limit':protocol.get('fresh_training_limit'),
        'checkpoint_selection':'preregistered terminal; intermediates retained separately',
        'automatic_FULL_or_submission':False}
    atomic(REPORT/f'result_{arm}.json',result)
    return result


def write_summary():
    results=[]
    for arm in ARMS:
        path=REPORT/f'result_{arm}.json'
        if path.exists():results.append(json.loads(path.read_text()))
    fields=['arm','status','PREFIX','L2_1s','L2_2s','L2_3s','delta_vs_control','sessions_improved']
    with (REPORT/'results.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in results:writer.writerow({k:r[k] for k in fields})
    lines=['# A2 motion / fresh initializer 결과','',
        '주 비교는 완료된 QREFINE과 같은 20,554-update terminal이다. 공식 서버 점수가 아니다.',
        '', '|Run|PREFIX|QREFINE 대비|Session paired 95% CI|', '|---|---:|---:|---|']
    for r in results:
        lo,hi=r['session_CI95']
        lines.append(f"|{r['arm']}|{r['PREFIX']:.9f}|{r['delta_vs_control']:+.9f}|[{lo:+.9f}, {hi:+.9f}]|")
    for arm in ARMS:
        if not any(r['arm']==arm for r in results):lines.append(f'|{arm}|미완료|—|—|')
    lines += ['', 'Fresh 초기값은 과거 ETRI 학습 노출을 제거했다. 동일 새 update 예산의 결과가',
        'scratch 수렴이나 구조 성능의 상한을 입증하지 않는다. 반복 사용한 V0 11개 session의',
        '조건부 bootstrap이며, 중간점 선택과 독립 검증을 혼동하지 않는다.',
        '', '세부 그룹·첫 2초·종/횡 오차·인지 지표·checkpoint SHA는 각 result JSON에 있다.',
        '새 FULL이나 공식 제출은 자동으로 시작하지 않았다.']
    (REPORT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')


def command(args,cwd=ROOT):
    return subprocess.check_output(args,cwd=cwd,text=True,stderr=subprocess.STDOUT)


def publish_results():
    # Existing user authorization includes recording experiments in this mirror.
    # Only this collector's artifacts are staged; concurrent work is preserved.
    if command(['git','diff','--cached','--name-only']).strip():
        raise RuntimeError('Index contains concurrent work; results saved, publishing stopped')
    files=[str((REPORT/f'result_{arm}.json').relative_to(ROOT)) for arm in ARMS]
    files += [str((REPORT/f).relative_to(ROOT)) for f in ('results.csv','RESULTS_KO.md')]
    command(['git','add','--',*files]);command(['git','diff','--cached','--check'])
    command(['git','commit','-m','Record A2 fine-motion and public-initialization terminal comparisons'])
    work=command(['git','rev-parse','HEAD']).strip()
    mirror=Path('/home/korea_sdv01/edrive_mirror')
    if command(['git','status','--porcelain'],mirror).strip():
        raise RuntimeError('Mirror contains concurrent work; publishing stopped')
    assert command(['git','branch','--show-current'],mirror).strip()=='motiondrive-v2-20260910'
    command(['git','fetch','github','motiondrive-v2-20260910'],mirror)
    command(['git','merge','--ff-only','FETCH_HEAD'],mirror)
    command(['git','fetch',str(ROOT),work],mirror)
    command(['git','cherry-pick',work],mirror)
    command(['git','push','github','HEAD:refs/heads/motiondrive-v2-20260910'],mirror)
    remote=command(['git','rev-parse','HEAD'],mirror).strip()
    assert command(['git','ls-remote','github','refs/heads/motiondrive-v2-20260910'],mirror).split()[0]==remote
    return {'work_commit':work,'mirror_commit':remote}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');ap.add_argument('--publish',action='store_true')
    args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','Collector must use CPU only'
    path=REPORT/'watcher_status.json'
    state={'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'training_or_FULL_or_submission_started':False,'arms':{}}
    atomic(path,state)
    deadline=time.monotonic()+24*3600
    try:
        while len(state['arms'])<len(ARMS):
            for arm in ARMS:
                if arm in state['arms']:continue
                run=RUNS/f'{arm}-s1';manifest=json.loads((run/'manifest.json').read_text())
                if manifest['status']=='completed' and (run/'experiment.json').exists():
                    r=collect(arm);state['arms'][arm]={'status':'completed','PREFIX':r['PREFIX']};write_summary()
                    atomic(path,state)
                elif manifest['status']=='failed':
                    state['arms'][arm]={'status':'failed','error':manifest.get('error')};atomic(path,state)
                else:
                    proc=Path(f"/proc/{manifest['pid']}/stat")
                    if not proc.exists() or proc.read_text().split()[2]=='Z':
                        raise RuntimeError(f'{arm} ended without a completed manifest')
            if len(state['arms'])==len(ARMS):break
            if not args.watch:return
            if time.monotonic()>deadline:raise TimeoutError('Collector24h deadline; training left untouched')
            time.sleep(30)
        if all(v['status']=='completed' for v in state['arms'].values()) and args.publish:
            state['publication']=publish_results()
        state['status']='completed' if all(v['status']=='completed' for v in state['arms'].values()) else 'failed_arm'
        state['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(path,state)
    except BaseException:
        state.update(status='failed',error=traceback.format_exc());atomic(path,state);raise


if __name__=='__main__':main()
