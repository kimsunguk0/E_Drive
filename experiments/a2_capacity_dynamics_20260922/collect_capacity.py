"""Scheduled CPU-only matched-pair results, using the established row metric producer."""
from pathlib import Path
import argparse,csv,datetime,json,os,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/a2_splitread_lanegeom_20260921'))
from collect_next import read,stats,compare,sha
REPORT=ROOT/'reports/a2_capacity_dynamics_20260922';RUNS=ROOT/'work_dirs/a2_capacity_dynamics_20260922'
PARENT=ROOT/'work_dirs/a2_native_detail_20260921/M-NATIVE-s1/predictions_step10277.json'
ARMS=('C-CTRL','C-R101','C-DECSPLIT','C-AGENT');STEPS=(0,3426,6852,10277)
def atomic(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def result(step):
    _,key,gt,pp,bucket,session=read(PARENT);parent,pd=stats(gt,pp,bucket)
    out={'step':step,'primary':step==STEPS[-1],'parent':parent,'arms':{}};arrays={}
    for arm in ARMS:
        path=RUNS/f'{arm}-step0.json' if step==0 else RUNS/f'{arm}-s1'/f'predictions_step{step}.json'
        raw,k,g,p,b,s=read(path)
        assert k==key and np.array_equal(g,gt) and np.array_equal(b,bucket) and np.array_equal(s,session)
        a,arrays[arm]=stats(gt,p,bucket)
        assert abs(a['PREFIX']-raw['report']['official_d3'])<1e-6
        a.update(predictions=str(path),predictions_sha256=sha(path),
            auxiliary={k:raw['report'][k] for k in ('occ_iou','lane_iou','state_mae_vx_vy_ax_ay_yawrate','history_position_mae_by_offset')})
        if step:
            ckpt=RUNS/f'{arm}-s1'/f'ckpt_step{step}.pth'
            a.update(checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt))
        out['arms'][arm]=a
    for arm in ARMS:
        out['arms'][arm]['delta_control']=compare(arrays[arm]-arrays['C-CTRL'],session)
        out['arms'][arm]['delta_parent']=compare(arrays[arm]-pd,session)
    atomic(REPORT/f'result_step{step}.json',out);return out
def stream():
    streams={};manifests={}
    for arm in ARMS:
        run=RUNS/f'{arm}-s1';p=run/'metrics.jsonl';v={}
        for line in p.read_text().splitlines() if p.exists() else []:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            if r.get('kind')=='train':v[r['step']]=r['sample_order_sha256']
        streams[arm]=v
        if (run/'manifest.json').exists():manifests[arm]=json.loads((run/'manifest.json').read_text())
    common=set.intersection(*(set(v) for v in streams.values()))
    bad=[s for s in common if len({v[s] for v in streams.values()})!=1]
    assert not bad,('Different sample streams',bad)
    return {'steps_compared':len(common),'last_compared_step':max(common,default=0),'mismatch_steps':bad},manifests
def summarize(rows,done):
    st,m=stream();rows=sorted(rows,key=lambda r:r['step'])
    terminal=next((r for r in rows if r['step']==STEPS[-1]),None)
    out={'status':'completed' if done else 'running','primary_step':STEPS[-1],
        'available_steps':[r['step'] for r in rows],'sample_stream':st,'terminal':terminal,
        'parent_DEV_PREFIX':.14459766188205064,'official_FULL_PREFIX':.1336848279459137,
        'server_conversion_or_prediction':None,'FULL_started':False,'uploaded':False,
        'selected_intermediate_on_reused_V0':{a:min(({'step':r['step'],'PREFIX':r['arms'][a]['PREFIX']}
            for r in rows if r['step']),key=lambda v:v['PREFIX'],default=None) for a in ARMS}}
    if done:
        assert st['steps_compared']==STEPS[-1] and st['last_compared_step']==STEPS[-1]
        audit=[m[a]['stream_audit'] for a in ARMS]
        for key in ('microcalls','rows_sha256','baseline_augmentation_sha256','detail_pixels_sha256'):
            assert len({v[key] for v in audit})==1,key
        out['terminal_baseline_and_sample_stream_equal']=True
        best=min(ARMS,key=lambda a:terminal['arms'][a]['PREFIX'])
        decision={'lowest_terminal_arm':best,'PREFIX':terminal['arms'][best]['PREFIX'],
            'improves_parent':terminal['arms'][best]['PREFIX']<terminal['parent']['PREFIX'],
            'treatment_minus_control':{a:terminal['arms'][a]['delta_control'] for a in ARMS[1:]},
            'requires_review_of_cost_auxiliary_groups':True,'automatic_FULL':False,'automatic_sweep':False}
        atomic(REPORT/'decision.json',decision);out['decision']=decision
    atomic(REPORT/'results.json',out)
    with (REPORT/'results.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(('step','arm','PREFIX','delta_control','delta_parent','L2_1s','L2_2s','L2_3s','nonstop','depart','steady'))
        for r in rows:
            for arm,a in r['arms'].items():w.writerow((r['step'],arm,a['PREFIX'],a['delta_control']['mean_delta'],a['delta_parent']['mean_delta'],
                a['L2_1s'],a['L2_2s'],a['L2_3s'],*(a['groups'][g]['PREFIX'] for g in ('nonstop','depart','steady'))))
    lines=['# Backbone capacity / full decoder split / agent future matched continuations','',
        'DEV terminal 10,277 완료.' if done else 'DEV 진행 중. 현재 평가를 terminal 성과로 해석하지 않는다.',
        '부모 DEV 0.144597662 / 제출 FULL 0.133684828. 서로 다른 평가이며 환산하지 않는다.','',
        '|Stage step|CTRL|R101|DECSPLIT|AGENT|','|---|---:|---:|---:|---:|']
    for r in rows:lines.append('|'+str(r['step'])+'|'+'|'.join(f"{r['arms'][a]['PREFIX']:.9f}" for a in ARMS)+'|')
    lines+=['',f"같은 sample 순서 {st['steps_compared']} update 확인.",
        '각 treatment−CTRL이 주 비교다. 부모 대비는 추가 학습 효과를 포함한다.',
        '시점·일반주행·종횡·구간 벡터·인지·세션 paired CI는 result_step*.json에 보존한다.',
        'FULL 및 공식 업로드는 실행하지 않는다.']
    (REPORT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    return out,m
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');args=ap.parse_args();seen={};start=time.monotonic()
    while True:
        for step in STEPS:
            if step in seen:continue
            paths=[RUNS/f'{a}-step0.json' if not step else RUNS/f'{a}-s1'/f'predictions_step{step}.json' for a in ARMS]
            if all(p.exists() for p in paths) and (not step or all((p.parent/f'ckpt_step{step}.pth').exists() for p in paths)):
                seen[step]=result(step)
        _,m=stream()
        done=len(m)==4 and all(v.get('status')=='completed' and v.get('step')==STEPS[-1] and
            (RUNS/f'{a}-s1/experiment.json').exists() for a,v in m.items())
        if done:assert len(seen)==len(STEPS) and all(v['nonfinite_count']==0 for v in m.values())
        summary,m=summarize(list(seen.values()),done)
        atomic(REPORT/'runtime/collector_status.json',{'pid':os.getpid(),'status':summary['status'],
            'available_steps':sorted(seen),'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        if done or args.once:return
        if any(v.get('status') in ('failed','stopped') for v in m.values()):raise RuntimeError('An arm failed/stopped; no automatic continuation')
        if time.monotonic()-start>24*3600:raise TimeoutError('Collector deadline; training jobs untouched')
        time.sleep(30)
if __name__=='__main__':main()
