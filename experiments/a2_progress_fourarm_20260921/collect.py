"""CPU-only scheduled result collection; no optimizer or model selection by in-fit FULL."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,subprocess,time,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/a2_progress_fourarm_20260921'
RUNS=ROOT/'work_dirs/a2_progress_fourarm_20260921'
PARENT=ROOT/'work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json'
ARMS=('P-CTRL','P-VECTOR','P-FINE','P-SHARED768');STEPS=(0,1142,2284,3426)
W=np.asarray([11,11,5,5,2,2],np.float64)/36
def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):
    payload=json.loads(path.read_text());r=payload['records']
    key=[(v['row'],v['session'],v['scenario'],v['frame']) for v in r]
    gt=np.asarray([v['gt_abs_xy'] for v in r],np.float64)
    pred=np.asarray([v['pred_abs_xy'] for v in r],np.float64)
    bucket=np.asarray([v['stop_bucket'] for v in r]);session=np.asarray([v['session'] for v in r])
    assert len(r)==1998 and len(set(key))==1998 and np.isfinite(pred).all()
    return payload,key,gt,pred,bucket,session
def stats(gt,p,bucket):
    err=p-gt;e=np.linalg.norm(err,axis=-1);d=e@W
    dp=np.diff(np.concatenate((np.zeros((len(p),1,2)),p),1),axis=1)
    dg=np.diff(np.concatenate((np.zeros((len(p),1,2)),gt),1),axis=1)
    lp=np.linalg.norm(dp,axis=-1);lg=np.linalg.norm(dg,axis=-1)
    valid=lg>.05;u=dg/np.maximum(lg[...,None],1e-12)
    long=(err*u).sum(-1);lat=err[...,0]*u[...,1]-err[...,1]*u[...,0]
    mask=valid*W;den=mask.sum()
    heading=np.arctan2(dg[...,1],dg[...,0]);predheading=np.arctan2(dp[...,1],dp[...,0])
    hd=np.abs(np.arctan2(np.sin(predheading-heading),np.cos(predheading-heading)))
    pred_degenerate=(lp<=.01)&valid
    groups={}
    for name in ('nonstop','depart','steady'):
        ix=bucket==name
        groups[name]={'n':int(ix.sum()),'PREFIX':float(d[ix].mean()),'contribution':float(d[ix].sum()/len(d)),
            'L2_1s':float(e[ix,:2].mean()),'first1s_contribution':float((e[ix,:2]@W[:2]).sum()/len(d)),
            'signed_interval_length_m':(lp[ix]-lg[ix]).mean(0).tolist()}
    curve=(valid.all(1)&(lg.sum(1)>=3)&(np.abs(heading).max(1)>=np.deg2rad(15)))
    return dict(PREFIX=float(d.mean()),L2_1s=float(e[:,:2].mean()),L2_2s=float(e[:,:4].mean()),L2_3s=float(e.mean()),
        point_L2=e.mean(0).tolist(),pair_PREFIX_contributions=[float((e[:,i:i+2]@W[i:i+2]).mean()) for i in (0,2,4)],
        groups=groups,interval_vector_MAE=np.linalg.norm(dp-dg,axis=-1).mean(0).tolist(),
        interval_length_MAE=np.abs(lp-lg).mean(0).tolist(),signed_interval_length_error=(lp-lg).mean(0).tolist(),
        weighted_longitudinal_abs=float((np.abs(long)*mask).sum()/den),
        weighted_lateral_abs=float((np.abs(lat)*mask).sum()/den),
        GT_tangent_mask='GT interval length > .05m',GT_mask_count_by_interval=valid.sum(0).tolist(),
        predicted_degenerate_on_GT_mask_by_interval=pred_degenerate.sum(0).tolist(),
        heading_MAE_deg_including_zero_pred_convention=(hd*valid).sum(0).__truediv__(valid.sum(0)).__mul__(180/np.pi).tolist(),
        zero_prediction_heading_convention='atan2(0,0)=0; counted separately, never excluded from PREFIX',
        curve_n=int(curve.sum()),curve_PREFIX=float(d[curve].mean()),
        curve_signed_interval_length=(lp[curve]-lg[curve]).mean(0).tolist()),d
def compare(delta,session):
    names=sorted(set(session));idx=[np.flatnonzero(session==s) for s in names]
    sums=np.asarray([delta[i].sum() for i in idx]);count=np.asarray([len(i) for i in idx])
    draw=np.random.default_rng(0).integers(0,len(names),(20000,len(names)))
    boot=sums[draw].sum(1)/count[draw].sum(1)
    influential=int(np.argmax(np.abs(sums)))
    return {'mean_delta':float(delta.mean()),'session_CI95':np.quantile(boot,[.025,.975]).tolist(),
        'session_delta':{s:{'n':int(count[i]),'mean':float(sums[i]/count[i]),'overall_contribution':float(sums[i]/len(delta))} for i,s in enumerate(names)},
        'largest_contribution_session':names[influential],
        'delta_excluding_largest_contribution_session':float((sums.sum()-sums[influential])/(count.sum()-count[influential])),
        'limitation':'Conditional resampling of reused DEV sessions, not hidden-test proof'}
def result(step):
    parent,key,gt,pp,bucket,session=read(PARENT);parent_stats,parent_d=stats(gt,pp,bucket)
    arrays={};out={'step':step,'primary':step==3426,'parent':parent_stats,'arms':{}}
    for arm in ARMS:
        path=REPORT/f'{arm}_step0.json' if step==0 else RUNS/f'{arm}-s1'/f'predictions_step{step}.json'
        raw,k,g,p,b,s=read(path);assert k==key and np.array_equal(g,gt) and np.array_equal(b,bucket)
        a,arr=stats(gt,p,bucket);arrays[arm]=arr
        assert abs(a['PREFIX']-raw['report']['official_d3'])<1e-6
        a.update(predictions=str(path),predictions_sha256=sha(path),
            auxiliary={k:raw['report'][k] for k in ('occ_iou','lane_iou','state_mae_vx_vy_ax_ay_yawrate','history_position_mae_by_offset')})
        if step:
            ckpt=RUNS/f'{arm}-s1'/f'ckpt_step{step}.pth'
            a.update(checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt))
        out['arms'][arm]=a
    for arm in ARMS:
        a=out['arms'][arm];a['delta_ctrl']=compare(arrays[arm]-arrays['P-CTRL'],session)
        a['delta_parent']=compare(arrays[arm]-parent_d,session)
    atomic(REPORT/f'result_step{step}.json',out);return out
def stream():
    values={};manifests={}
    for arm in ARMS:
        p=RUNS/f'{arm}-s1/metrics.jsonl';v={}
        for line in p.read_text().splitlines() if p.exists() else []:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('kind')=='train':v[r['step']]=r['sample_order_sha256']
        values[arm]=v
        p=p.parent/'manifest.json'
        if p.exists():manifests[arm]=json.loads(p.read_text())
    common=set.intersection(*(set(v) for v in values.values()))
    bad=[s for s in sorted(common) if len({v[s] for v in values.values()})!=1]
    assert not bad,('Sample streams diverged',bad)
    return {'steps_compared':len(common),'last_compared_step':max(common,default=0),'mismatch_steps':bad},manifests
def summarize(rows,completed):
    st,manifests=stream();rows=sorted(rows,key=lambda r:r['step'])
    terminal=next((r for r in rows if r['step']==3426),None)
    out={'status':'completed' if completed else 'running','terminal_primary_step':3426,
        'available_steps':[r['step'] for r in rows],'sample_stream':st,'terminal':terminal,
        'official_server_baseline':.1336848279459137,'server_conversion_or_prediction':None,
        'selected_intermediate_on_reused_V0':{a:min(({'step':r['step'],'PREFIX':r['arms'][a]['PREFIX']} for r in rows if r['step']),key=lambda v:v['PREFIX'],default=None) for a in ARMS}}
    if completed:
        assert st['last_compared_step']==3426 and st['steps_compared']==3426
        audit=[manifests[a]['stream_audit'] for a in ARMS]
        assert all(v==audit[0] for v in audit)
        out['terminal_augmentation_audit_equal']=True
        best=min(ARMS,key=lambda a:terminal['arms'][a]['PREFIX'])
        improve=terminal['arms'][best]['PREFIX']<terminal['parent']['PREFIX']
        decision={'status':'DEV_complete','lowest_terminal_arm':best,'candidate_for_FULL':best if improve else None,
            'reason':'lowest terminal PREFIX below frozen parent' if improve else 'no terminal improved frozen parent',
            'primary':terminal['arms'][best],'parent_PREFIX':terminal['parent']['PREFIX'],
            'FULL_transfer_updates':4156 if improve else None,'FULL_started':False,'uploaded':False,
            'tradeoffs_require_reading':'group, signed length, auxiliary and session deltas retained in primary',
            'no_fixed_minimum_gain_filter':True,'CI_not_hard_gate':True}
        atomic(REPORT/'decision.json',decision);out['decision']=decision
        atomic(REPORT/'candidate_registry.json',{'DEV':decision,'preserved_official_FULL':{
            'checkpoint':'work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1/ckpt_step24931.pth',
            'sha256':'dae99f86f29e92296b1d036db3f7933d41bf9a5c3dd3af5153416a8eaa676b14',
            'official_PREFIX':.1336848279459137},'new_FULL':None})
    atomic(REPORT/'results.json',out)
    with (REPORT/'results.csv').open('w') as f:
        fields=['step','arm','PREFIX','delta_ctrl','delta_parent','L2_1s','L2_2s','L2_3s','nonstop','depart','steady']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:
            for arm,a in r['arms'].items():
                w.writerow(dict(step=r['step'],arm=arm,**{k:a[k] for k in ('PREFIX','L2_1s','L2_2s','L2_3s')},
                    delta_ctrl=a['delta_ctrl']['mean_delta'],delta_parent=a['delta_parent']['mean_delta'],
                    **{k:a['groups'][k]['PREFIX'] for k in ('nonstop','depart','steady')}))
    lines=['# PROGRESS 네 arm 동일 조건 continuation','',
        '완료: terminal3,426이 주 비교다.' if completed else '실행 중. 예정 중간 결과를 terminal로 해석하지 않는다.',
        'DEV 부모0.151178860, 공식 FULL0.133684828은 서로 다른 평가다. 서버 점수 환산 없음.','',
        '|Stage step|CTRL|VECTOR|FINE|SHARED768|','|---|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+str(r['step'])+'|'+'|'.join(f"{r['arms'][a]['PREFIX']:.9f}" for a in ARMS)+'|')
    lines+=['',f"동일 sample stream {st['steps_compared']}개 update 확인.",
        '시점·주행군·signed 구간 길이·횡/종 오차·인지 지표·세션 집중도는 result_step*.json에 보존한다.',
        'FINE 추가 정보를 쓰는 경로와 VECTOR loss 변경은 독립 비교이며 효과를 더해서 예측하지 않는다.']
    if completed:lines+=['',f"FULL 검토 후보: {out['decision']['candidate_for_FULL']}. FULL은 별도 실행이며 아직 시작 상태가 아니다."]
    (REPORT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    return out,manifests
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');args=ap.parse_args()
    seen={};start=time.monotonic()
    while True:
        for step in STEPS:
            if step in seen:continue
            paths=[REPORT/f'{a}_step0.json' if not step else RUNS/f'{a}-s1'/f'predictions_step{step}.json' for a in ARMS]
            if all(p.exists() for p in paths) and (not step or all((p.parent/f'ckpt_step{step}.pth').exists() for p in paths)):
                seen[step]=result(step)
        _,m=stream()
        done=len(m)==4 and all(v.get('status')=='completed' and v.get('step')==3426 and
            (RUNS/f'{a}-s1/experiment.json').exists() for a,v in m.items())
        if done:assert len(seen)==4 and all(v['nonfinite_count']==0 for v in m.values())
        summary,m=summarize(list(seen.values()),done)
        atomic(REPORT/'runtime/collector_status.json',{'pid':os.getpid(),'status':summary['status'],
            'available_steps':sorted(seen),'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        if done or args.once:return
        if any(v.get('status') in ('failed','stopped') for v in m.values()):raise RuntimeError('An arm stopped/failed; no automatic continuation')
        if time.monotonic()-start>6*3600:raise TimeoutError('Collection deadline; jobs untouched')
        time.sleep(30)
if __name__=='__main__':main()
