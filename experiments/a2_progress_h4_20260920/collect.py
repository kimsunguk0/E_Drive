"""Collect matched scheduled evaluations on CPU; publish results, never train."""
from pathlib import Path
import argparse,csv,datetime,json,os,subprocess,time,traceback
import numpy as np
from train_progress import ROOT,REPORT,RUNS,ARMS,DIRECT,PROGRESS,UPDATES,EVERY,nominal,sha

STEPS=[*range(EVERY,UPDATES,EVERY),UPDATES]
W=np.array([11,11,5,5,2,2],np.float64)/36
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def command(argv,cwd=ROOT):return subprocess.check_output(argv,cwd=cwd,text=True,stderr=subprocess.STDOUT)

def paired_delta(a,b,session):
    delta=b-a;unique=sorted(set(session));groups=[np.flatnonzero(session==s) for s in unique]
    sums=np.array([delta[i].sum() for i in groups]);counts=np.array([len(i) for i in groups])
    draws=np.random.default_rng(0).integers(0,len(groups),(20000,len(groups)))
    samples=sums[draws].sum(1)/counts[draws].sum(1)
    return {'delta_PROGRESS_minus_DIRECT':float(delta.mean()),'session_CI95':np.quantile(samples,[.025,.975]).tolist(),
        'sessions_improved':int((sums<0).sum()),'session_count':len(unique),
        'limitation':'Conditional paired resampling of the same reused 11 DEV sessions, not independent test evidence.'}

def compare(step,write=True,run_suffix=''):
    arrays={};result={'step':step,'arms':{},'interpretation':'fixed terminal' if step==UPDATES else 'planned intermediate on reused V0'}
    keys=None;gt=None
    for arm in ARMS:
        run=RUNS/f'{arm}-s1{run_suffix}';path=run/f'predictions_step{step}.json';payload=json.loads(path.read_text());records=payload['records']
        current=[(r['row'],r['session'],r['scenario'],r['frame']) for r in records]
        truth=np.array([r['gt_abs_xy'] for r in records],np.float64)
        if keys is None:keys=current;gt=truth
        assert current==keys and len(current)==1998 and np.array_equal(gt,truth)
        pred=np.array([r['pred_abs_xy'] for r in records],np.float64);assert np.isfinite(pred).all()
        error=np.linalg.norm(pred-gt,axis=-1);score=error@W
        assert abs(float(score.mean())-payload['report']['official_d3'])<1e-6
        diagnostic=nominal.diagnostics(records)
        result['arms'][arm]={'PREFIX':float(score.mean()),'L2_1s':float(error[:,:2].mean()),
            'L2_2s':float(error[:,:4].mean()),'L2_3s':float(error.mean()),'per_waypoint_L2':error.mean(0).tolist(),
            'diagnostics':diagnostic,'auxiliary':{k:payload['report'][k] for k in ('occ_iou','lane_iou',
                'state_mae_vx_vy_ax_ay_yawrate','history_position_mae_by_offset')},
            'checkpoint':str(run/f'ckpt_step{step}.pth'),'checkpoint_sha256':sha(run/f'ckpt_step{step}.pth'),
            'predictions':str(path),'predictions_sha256':sha(path)}
        arrays[arm]=score
    session=np.array([k[1] for k in keys]);result['comparison']=paired_delta(arrays[DIRECT],arrays[PROGRESS],session)
    gd=result['arms'][DIRECT]['diagnostics']['groups'];gp=result['arms'][PROGRESS]['diagnostics']['groups']
    result['group_deltas']={k:{'mean_PREFIX':gp[k]['PREFIX']-gd[k]['PREFIX'],
        'overall_contribution':gp[k]['overall_PREFIX_contribution']-gd[k]['overall_PREFIX_contribution']} for k in gd}
    result['no_automatic_server_conversion']=True
    if write:atomic(REPORT/f'result_step{step}.json',result)
    return result

def row_hashes():
    values=[]
    for arm in ARMS:
        path=RUNS/f'{arm}-s1'/'metrics.jsonl';out={}
        for line in path.read_text().splitlines() if path.exists() else []:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if 'sample_order_sha256' in r:out[r['step']]=r['sample_order_sha256']
        values.append(out)
    a,b=values;shared=sorted(set(a)&set(b));bad=[s for s in shared if a[s]!=b[s]]
    return {'logged_steps_compared':len(shared),'last_compared_step':shared[-1] if shared else None,
        'mismatch_steps':bad,'all_compared_equal':not bad,
        'unlogged_tail':'Digest covers logging steps; both use the checked deterministic sampler.'}

def summarize(completed):
    rows=sorted([json.loads(p.read_text()) for p in REPORT.glob('result_step*.json')],key=lambda r:r['step'])
    sample=row_hashes();assert not sample['mismatch_steps'],sample
    terminal=next((r for r in rows if r['step']==UPDATES),None)
    selected={arm:({'step':min(rows,key=lambda r:r['arms'][arm]['PREFIX'])['step'],
        'PREFIX':min(r['arms'][arm]['PREFIX'] for r in rows),'independent_validation':False} if rows else None) for arm in ARMS}
    value={'status':'completed' if completed else 'running','primary_step':UPDATES,'planned_steps':STEPS,
        'available_steps':[r['step'] for r in rows],'terminal':terminal,'selected_on_reused_V0':selected,
        'sample_stream':sample,'new_input_policy':'five H4/current poses only, common scene query',
        'automatic_training_FULL_submission':False}
    atomic(REPORT/'results.json',value)
    fields=['step','arm','PREFIX','L2_1s','L2_2s','L2_3s','nonstop','depart','steady',
        'first2s_contribution','long_abs','lat_abs','delta_PROGRESS_minus_DIRECT']
    with (REPORT/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for r in rows:
            for arm in ARMS:
                a=r['arms'][arm];d=a['diagnostics'];g=d['groups']
                writer.writerow(dict(step=r['step'],arm=arm,**{k:a[k] for k in ('PREFIX','L2_1s','L2_2s','L2_3s')},
                    **{k:g[k]['PREFIX'] for k in ('nonstop','depart','steady')},first2s_contribution=d['first2s_PREFIX_contribution'],
                    long_abs=d['weighted_longitudinal_abs_m'],lat_abs=d['weighted_lateral_abs_m'],
                    delta_PROGRESS_minus_DIRECT=r['comparison']['delta_PROGRESS_minus_DIRECT']))
    lines=['# H4 status + 구간 진행량/방향 비교','',
        '두 arm 모두 20,554 update 완료.' if completed else '학습 중. 예정 중간 평가를 고정 terminal 결과로 해석하지 않는다.',
        '동일 TemporalRead·공개 초기값·5시점 status·loss·sample stream·budget에서 출력 구조를 비교한다.',
        '기존 11시점 status 모델은 이 구조의 matched control이 아니다. 초기 trainable tensor는 같지만 초기 궤적은 다르다.',
        '', '|Step|DIRECT PREFIX|PROGRESS PREFIX|차이|DIRECT 일반주행|PROGRESS 일반주행|',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        a,b=r['arms'][DIRECT],r['arms'][PROGRESS]
        lines.append(f"|{r['step']}|{a['PREFIX']:.9f}|{b['PREFIX']:.9f}|{r['comparison']['delta_PROGRESS_minus_DIRECT']:+.9f}|{a['diagnostics']['groups']['nonstop']['PREFIX']:.9f}|{b['diagnostics']['groups']['nonstop']['PREFIX']:.9f}|")
    if terminal:
        c=terminal['comparison'];lines+=['',f"고정 terminal 차이 {c['delta_PROGRESS_minus_DIRECT']:+.9f}; session paired 95% CI {c['session_CI95']}."]
        baseline=json.loads((REPORT/'policy_CONT.json').read_text())['new']['PREFIX']
        p=terminal['arms'][PROGRESS]['PREFIX'];d=terminal['arms'][DIRECT]['PREFIX']
        verdict=('matched control과 보존된 H4 입력 CONT 단일 후보보다 모두 낮다. DEV 후보로 보존한다.'
            if p<d and p<baseline else 'matched control 차이와 기존 단일 후보 대비를 구분한다. 이 값만으로 새 최선 또는 제출 승격을 주장하지 않는다.')
        lines+=['',f"PROGRESS 판정: {verdict}",f"H4 입력의 frozen CONT 참고값 {baseline:.9f}; 훈련 계보·예산은 다르다."]
    lines+=['',f"Sample stream: {sample['logged_steps_compared']}개 로그 비교, 불일치 {sample['mismatch_steps']}.",
        '인지·state/history·구간 진행량·첫2초·종횡 지표는 result_step*.json에 기록한다.',
        '서버 점수 예측이나 0.12 달성을 주장하지 않는다. 자동 추가 학습·FULL·공식 제출은 없다.']
    (REPORT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    return value

def publish():
    receipt={}
    try:
        if command(['git','diff','--cached','--name-only']).strip():raise RuntimeError('Concurrent staging; defer publication')
        files=[str(p.relative_to(ROOT)) for p in sorted(REPORT.glob('result_step*.json'))]
        files += [str((REPORT/n).relative_to(ROOT)) for n in ('results.json','results.csv','RESULTS_KO.md')]
        command(['git','add','--',*files]);command(['git','diff','--cached','--check'])
        command(['git','commit','-m','Record matched H4-status progress-heading training results'])
        work=command(['git','rev-parse','HEAD']).strip();receipt['work_commit']=work
        mirror=Path('/home/korea_sdv01/edrive_mirror')
        if command(['git','status','--porcelain'],mirror).strip():raise RuntimeError('Concurrent mirror changes; defer publication')
        assert command(['git','branch','--show-current'],mirror).strip()=='motiondrive-v2-20260910'
        command(['git','fetch','github','motiondrive-v2-20260910'],mirror);command(['git','merge','--ff-only','FETCH_HEAD'],mirror)
        command(['git','fetch',str(ROOT),work],mirror);command(['git','cherry-pick',work],mirror)
        receipt['mirror_commit']=command(['git','rev-parse','HEAD'],mirror).strip()
        command(['git','push','github','HEAD:motiondrive-v2-20260910'],mirror);receipt['push_succeeded']=True
        receipt['remote_head']=command(['git','ls-remote','github','refs/heads/motiondrive-v2-20260910'],mirror).split()[0]
        receipt['remote_verified']=receipt['remote_head']==receipt['mirror_commit'];receipt['status']='pushed'
    except BaseException:receipt.update(status='publication_incomplete',error=traceback.format_exc())
    atomic(REPORT/'completion_receipt.json',receipt);return receipt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--publish',action='store_true');ap.add_argument('--check-smoke',action='store_true');args=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    if args.check_smoke:
        result=compare(2,write=False,run_suffix='-smoke')
        atomic(REPORT/'collector_preflight.json',{'status':'passed','actual_two_smoke_predictions_compared':True,
            'n':1998,'comparison':result['comparison'],'collector_sha256':sha(Path(__file__))})
        print('Collector smoke comparison passed');return
    status=REPORT/'runtime/watcher_status.json';state={'status':'watching','pid':os.getpid(),
        'training_started_by_collector':False,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic(status,state);deadline=time.monotonic()+8*3600;collected=set()
    try:
        while True:
            manifests={}
            for arm in ARMS:
                path=RUNS/f'{arm}-s1'/'manifest.json'
                if path.exists():manifests[arm]=json.loads(path.read_text())
            for step in STEPS:
                if step not in collected and all((RUNS/f'{arm}-s1'/f'predictions_step{step}.json').exists()
                    and (RUNS/f'{arm}-s1'/f'ckpt_step{step}.pth').exists() for arm in ARMS):
                    compare(step);collected.add(step)
            finished=len(manifests)==2 and all(m['status']=='completed' and
                (RUNS/f'{arm}-s1'/'experiment.json').exists() for arm,m in manifests.items())
            if finished:assert all(m['step']==UPDATES and m['nonfinite_count']==0 for m in manifests.values()) and len(collected)==len(STEPS)
            summary=summarize(finished);state.update(available_steps=sorted(collected),
                arms={a:{k:m.get(k) for k in ('status','step','pid','nonfinite_count')} for a,m in manifests.items()},
                sample_stream=summary['sample_stream'],last_checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            atomic(status,state)
            if finished:
                state.update(status='completed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                if args.publish:state['publication']=publish()
                atomic(status,state);return
            for arm,m in manifests.items():
                if m['status'] in ('failed','stopped'):raise RuntimeError(arm+' '+m['status'])
                if m['status']!='completed':
                    proc=Path(f"/proc/{m['pid']}/stat")
                    if not proc.exists() or proc.read_text().split()[2]=='Z':raise RuntimeError(arm+' ended without completion')
            if time.monotonic()>deadline:raise TimeoutError('8h collection deadline; training untouched')
            time.sleep(30)
    except BaseException:state.update(status='collection_failed',error=traceback.format_exc());atomic(status,state);raise
if __name__=='__main__':main()
