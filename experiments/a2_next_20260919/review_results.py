"""Terminal paired comparisons only, with exact data/order checks and no fitting."""
import argparse
import csv
import datetime
import hashlib
import json
import numpy as np
from common import *
from terminal_review import metrics as existing_metrics

def read(path):return json.loads(path.read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--scope',choices=('G','all'),required=True)
    args=parser.parse_args()
    runs={'PARENT':PARENT.parent,'BASE':BASE_CONTROL,**{a:RUNS/f'{a}-s1' for a in G_ARMS}}
    if args.scope=='all':runs[S_ARM]=RUNS/f'{S_ARM}-s1'
    models={};scores={};orders={};reference=None;artifacts=[]
    for arm,run in runs.items():
        m=read(run/'manifest.json');assert m['status']=='completed' and m['nonfinite_count']==0
        expected=3426 if arm in G_ARMS else 20554
        assert m['step']==expected
        e=read(run/'final_eval.json');records=e['records'];assert len(records)==1998
        keys=[(r['row'],r['session'],r['scenario'],r['frame']) for r in records]
        gt=np.asarray([r['gt_abs_xy'] for r in records],np.float64)
        pred=np.asarray([r['pred_abs_xy'] for r in records],np.float64)
        buckets=np.asarray([r['bucket'] for r in records]);sessions=np.asarray([r['session'] for r in records])
        if reference is None:reference=(keys,gt,buckets,sessions)
        else:assert keys==reference[0] and np.array_equal(gt,reference[1]) and np.array_equal(buckets,reference[2])
        dg=np.diff(np.concatenate([np.zeros((len(gt),1,2)),gt],1),axis=1)
        stats,scores[arm]=existing_metrics(pred,gt,buckets,np.linalg.norm(dg,axis=-1)>.05)
        assert abs(stats['PREFIX']-e['report']['official_d3'])<1e-7
        log=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
        orders[arm]={r['step']:r['sample_order_sha256'] for r in log if r['kind']=='train'}
        curve=[{'step':r['step'],'PREFIX':r['official_d3']} for r in log if r['kind']=='eval']
        probe=REPORT/f'posttrain_probe_{arm}.json'
        train=read(probe)['datasets']['train']['PREFIX'] if probe.exists() else None
        models[arm]={'run':str(run),'step':m['step'],**stats,'train_probe_PREFIX':train,
          'state_mae':e['report']['state_mae_vx_vy_ax_ay_yawrate'],
          'history_mae':e['report']['history_position_mae_by_offset'],
          'occupancy_IoU':e['report']['occ_iou'],'lane_IoU':e['report']['lane_iou'],
          'learning_curve':curve,'best_on_reused_V0':min(curve,key=lambda z:z['PREFIX']),
          'best_selection_warning':'development selection, not independent validation',
          'diagnostics':nominal.diagnostics(records)}
        for name in ['manifest.json','experiment.json','metrics.jsonl','final_eval.json',f'ckpt_step{m["step"]}.pth']:
            p=run/name
            if p.exists():artifacts.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
    assert orders['A2-G0']==orders['A2-G1']
    g0=read(runs['A2-G0']/'manifest.json');g1=read(runs['A2-G1']/'manifest.json')
    assert g0['initial_model_state_sha256']==g1['initial_model_state_sha256']==read(REPORT/'CURRENT_BASE.json')['model_state_sha256']
    if args.scope=='all':
        assert orders[S_ARM]==orders['BASE']
        s=read(runs[S_ARM]/'experiment.json')
        assert s['initial_load']['shared_base_state_sha256']==nominal.BASE_INITIAL_SHA
    unique=sorted(set(sessions));indices=[np.flatnonzero(sessions==u) for u in unique]
    counts=np.asarray([len(i) for i in indices]);draws=np.random.default_rng(0).integers(0,len(unique),(20000,len(unique)))
    pairs=[('A2-G0','A2-G1'),('PARENT','A2-G0'),('PARENT','A2-G1')]
    if args.scope=='all':pairs += [('BASE',S_ARM),('PARENT',S_ARM)]
    comparisons={}
    for baseline,arm in pairs:
        delta=scores[arm]-scores[baseline];sums=np.asarray([delta[i].sum() for i in indices])
        boot=sums[draws].sum(1)/counts[draws].sum(1);lo,hi=np.percentile(boot,[2.5,97.5])
        comparisons[f'{arm} minus {baseline}']={'delta':float(delta.mean()),'relative_percent':float(100*delta.mean()/scores[baseline].mean()),
          'session_cluster_ci95':[float(lo),float(hi)],'ci_includes_zero':bool(lo<=0<=hi),
          'sessions_improved':int((sums<0).sum()),'session_count':len(unique),
          'session_delta':{u:float(delta[i].mean()) for u,i in zip(unique,indices)},
          'group_delta':{b:float(delta[buckets==b].mean()) for b in sorted(set(buckets))},
          'group_contribution_delta':{b:float(delta[buckets==b].sum()/len(delta)) for b in sorted(set(buckets))}}
    out={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':args.scope,
       'models':models,'comparisons':comparisons,
       'verified':{'same_1998_rows_GT_buckets':True,'G_logged_sample_orders_match':len(orders['A2-G0']),
                   'S_logged_sample_orders_match':len(orders[S_ARM]) if args.scope=='all' else None,'FULL_excluded':True},
       'limitations':['Repeatedly used DEV; one seed per recipe.','Session bootstrap on 11 observed sessions does not estimate training-seed variance or test score.',
                      'Intermediate best is selected on the same V0.','These scores are not official submissions.']}
    name='terminal_results.json' if args.scope=='all' else 'G_terminal_results.json'
    (REPORT/name).write_text(json.dumps(out,indent=2)+'\n')
    (REPORT/('run_artifact_index.json' if args.scope=='all' else 'G_artifact_index.json')).write_text(json.dumps(artifacts,indent=2)+'\n')
    fields=['run','step','train_probe_PREFIX','PREFIX','L2_1s','L2_2s','L2_3s','first2s_PREFIX_contribution','nonstop_PREFIX','occupancy_IoU','lane_IoU','RTX4090_ms']
    with (REPORT/('results.csv' if args.scope=='all' else 'G_results.csv')).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for a,m in models.items():writer.writerow({'run':a,**{k:m[k] for k in fields[1:8]},'nonstop_PREFIX':m['groups']['nonstop']['PREFIX'],
            'occupancy_IoU':m['occupancy_IoU'],'lane_IoU':m['lane_IoU'],'RTX4090_ms':None})
    text='# A2 결과 / '+args.scope+'\n\n동일 V0 1,998행의 고정 terminal 비교. 공식 서버 점수가 아니다.\n\n'
    text+='|Run|Step|PREFIX|일반 주행|첫2초 기여|\n|---|---:|---:|---:|---:|\n'
    for a,m in models.items():text+=f"|{a}|{m['step']}|{m['PREFIX']:.9f}|{m['groups']['nonstop']['PREFIX']:.9f}|{m['first2s_PREFIX_contribution']:.9f}|\n"
    text+='\n|비교|Delta|Session bootstrap95%|개선 session|\n|---|---:|---|---:|\n'
    for k,c in comparisons.items():text+=f"|{k}|{c['delta']:+.9f}|{c['session_cluster_ci95']}|{c['sessions_improved']}/11|\n"
    text+='\nG1−G0는 보조 비중 비교, 각 parent 대비는 추가학습까지 포함한 실제 후보 비교다.\n'
    text+='반복 사용한 DEV/단일 seed의 한계가 있다. CI가0을 포함하면 확실한 이득이라고 주장하지 않는다.\n'
    text+='S는 완료된 BASE와 공통 초기값/로그 sample-order를 대조한다. FULL score는 순위에 사용하지 않는다.\n'
    text+='Raw 배포/RTX4090 latency/공식 업로드 완료를 이 보고서가 대신하지 않는다. 자동 FULL 시작 또는 업로드는 수행하지 않았다.\n'
    (REPORT/('RESULTS_KO.md' if args.scope=='all' else 'G_RESULTS_KO.md')).write_text(text)
    print(json.dumps({'PREFIX':{a:m['PREFIX'] for a,m in models.items()},'comparisons':comparisons}),flush=True)

if __name__=='__main__':main()
