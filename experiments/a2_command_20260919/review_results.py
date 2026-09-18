"""Frozen terminal comparison; command/geometry groups never alter the fit."""
import argparse
import csv
import datetime
import json
import numpy as np
from command_common import *
from command_data import LABELS

WEIGHTS = np.array([11,11,5,5,2,2],dtype=np.float64)/36
QREFINE = ROOT/'work_dirs/md_a2_scene_extensions_20260918/A2-QREFINE-NOM-s1'

def read(path):
    return json.loads(path.read_text())

def group_stats(distance,score,mask,sessions):
    n = int(mask.sum())
    if not n:
        return {'n':0,'sessions':0,'PREFIX':None,'score_contribution':0.}
    return dict(n=n,sessions=len(set(sessions[mask])),PREFIX=float(score[mask].mean()),
        score_contribution=float(score[mask].sum()/len(score)),
        L2_1s=float(distance[mask,:2].mean()),L2_2s=float(distance[mask,:4].mean()),
        L2_3s=float(distance[mask].mean()))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke-check',action='store_true')
    args=parser.parse_args()
    run = RUNS/f'{ARM}-s1'
    if args.smoke_check:
        smoke=RUNS/f'{ARM}-s1-smoke'
        m=read(smoke/'manifest.json')
        assert m['status']=='completed' and m['step']==2 and m['nonfinite_count']==0
        r=read(smoke/'final_eval.json')
        assert len(r['records'])==1998
        old=[json.loads(v) for v in (CONTROL/'metrics.jsonl').read_text().splitlines()][0]
        first=[json.loads(v) for v in (smoke/'metrics.jsonl').read_text().splitlines()][0]
        assert old['sample_order_sha256']==first['sample_order_sha256']
        for k in ('total','plan_d3','occ_bce','lane_bce','motion','plan_interval_length'):
            assert old[k]==first[k],k
        payload=torch.load(smoke/'ckpt_step2.pth',map_location='cpu',weights_only=False)
        norm=float(payload['model']['shared_command_query.weight'].norm())
        assert norm>0
        result=dict(status='passed',updates=2,eval_rows=1998,nonfinite_count=0,
            exact_first_batch_loss_and_order_vs_BASE=True,command_weight_norm_after_two_steps=norm,
            full_effective_batch_normalization='unchanged existing producer; batch16/micro8',
            checkpoint_sha256=sha(smoke/'ckpt_step2.pth'),
            eval_is_smoke_not_performance_result=True)
        (REPORT/'smoke_summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    runs={'BASE':CONTROL,ARM:run,'QREFINE':QREFINE}
    models,scores,orders,artifacts = {},{},{},[]
    reference=None
    cache=np.load(CACHE/'provided_commands.npz',allow_pickle=False)
    for name,directory in runs.items():
        manifest=read(directory/'manifest.json')
        assert manifest['status']=='completed' and manifest['step']==20554 and manifest['nonfinite_count']==0
        result=read(directory/'final_eval.json');rec=result['records']
        rows=np.asarray([r['row'] for r in rec]);assert len(rows)==1998
        identity=[(r['row'],r['scenario'],r['frame'],r['session']) for r in rec]
        gt=np.asarray([r['gt_abs_xy'] for r in rec],np.float64)
        pred=np.asarray([r['pred_abs_xy'] for r in rec],np.float64)
        sessions=np.asarray([r['session'] for r in rec]);buckets=np.asarray([r['bucket'] for r in rec])
        if reference is None:
            reference=(identity,gt,buckets)
        else:
            assert identity==reference[0] and np.array_equal(gt,reference[1]) and np.array_equal(buckets,reference[2])
        pos=np.searchsorted(cache['row'],rows);assert np.array_equal(cache['row'][pos],rows)
        command=cache['command'][pos]
        distance=np.linalg.norm(pred-gt,axis=-1);score=distance@WEIGHTS;scores[name]=score
        assert abs(score.mean()-result['report']['official_d3'])<1e-7
        geometry={'left_3s_y_ge_2m':gt[:,-1,1]>=2,'right_3s_y_le_minus2m':gt[:,-1,1]<=-2,
                  'other_3s':abs(gt[:,-1,1])<2}
        all_stats=group_stats(distance,score,np.ones(len(rows),bool),sessions)
        log=[json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
        orders[name]={r['step']:r['sample_order_sha256'] for r in log if r['kind']=='train'}
        curve=[{'step':r['step'],'PREFIX':r['official_d3']} for r in log if r['kind']=='eval']
        models[name]=dict(run=str(directory),step=20554,**all_stats,
            semantic={label:group_stats(distance,score,command==i,sessions) for i,label in enumerate(LABELS)},
            geometry={label:group_stats(distance,score,mask,sessions) for label,mask in geometry.items()},
            groups={label:group_stats(distance,score,buckets==label,sessions) for label in sorted(set(buckets))},
            per_point_L2=distance.mean(0).tolist(),learning_curve=curve,
            selected_best_on_reused_V0=min(curve,key=lambda v:v['PREFIX']),
            best_warning='Selected from scheduled evaluations on repeatedly used DEV, not independent validation',
            diagnostics=nominal.diagnostics(rec),
            state_mae=result['report']['state_mae_vx_vy_ax_ay_yawrate'],
            occupancy_IoU=result['report']['occ_iou'],lane_IoU=result['report']['lane_iou'])
        for filename in ['manifest.json','experiment.json','metrics.jsonl','final_eval.json','ckpt_step20554.pth']:
            p=directory/filename;artifacts.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    assert orders[ARM]==orders['BASE']
    d=read(run/'experiment.json');base=read(CONTROL/'experiment.json')
    assert d['initial_load']['shared_base_state_sha256']==nominal.BASE_INITIAL_SHA
    for k in ('train_data','tune_data','initializer','recipe','nominal_input','expected_optimizer_groups'):
        assert d[k]==base[k],k
    unique=sorted(set(sessions));ids=[np.flatnonzero(sessions==u) for u in unique]
    counts=np.array([len(i) for i in ids]);draws=np.random.default_rng(0).integers(0,len(ids),(20000,len(ids)))
    comparisons={}
    for baseline in ('BASE','QREFINE'):
        delta=scores[ARM]-scores[baseline]
        sums=np.array([delta[i].sum() for i in ids]);boot=sums[draws].sum(1)/counts[draws].sum(1)
        ci=np.percentile(boot,[2.5,97.5])
        comparisons[baseline]=dict(delta=float(delta.mean()),relative_percent=float(100*delta.mean()/scores[baseline].mean()),
            session_cluster_CI95=ci.tolist(),CI_includes_zero=bool(ci[0]<=0<=ci[1]),
            sessions_improved=int((sums<0).sum()),sessions=len(ids),
            session_delta={u:float(delta[i].mean()) for u,i in zip(unique,ids)},
            semantic_delta={label:dict(n=int((command==i).sum()),
                mean=float(delta[command==i].mean()) if (command==i).any() else None,
                overall_contribution=float(delta[command==i].sum()/len(delta))) for i,label in enumerate(LABELS)})
    counterfactual=read(REPORT/'command_counterfactual.json') if (REPORT/'command_counterfactual.json').exists() else None
    payload=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),models=models,
        comparisons_command_minus_control=comparisons,command_counterfactual=counterfactual,
        verified=dict(same_rows_GT_buckets=True,logged_sample_orders_match=len(orders[ARM]),
            shared_initial_tensors_identical=True,same_existing_recipe=True,FULL_excluded=True),
        limitations=['Repeated DEV and one seed; session CI does not include training-seed variation.',
            'Semantic TURN_LEFT: 39 rows from one session. V0 has no U_TURN.',
            'GT geometry groups include bends/lane changes, not semantic turn labels.',
            'No official submission or RTX4090 latency result.'])
    (REPORT/'terminal_results.json').write_text(json.dumps(payload,indent=2)+'\n')
    (REPORT/'run_artifact_index.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    text='# A2 semantic command 실험 결과\n\n동일 V0 1,998행, 고정20,554 update terminal 비교. 공식 서버 점수가 아니다.\n\n'
    text+='|Run|PREFIX|L2_1s|L2_2s|L2_3s|일반 주행|\n|---|---:|---:|---:|---:|---:|\n'
    for name,m in models.items():
        text+=f"|{name}|{m['PREFIX']:.9f}|{m['L2_1s']:.9f}|{m['L2_2s']:.9f}|{m['L2_3s']:.9f}|{m['groups']['nonstop']['PREFIX']:.9f}|\n"
    text+='\n|Command − 비교군|Delta|Session95%CI|개선 session|\n|---|---:|---|---:|\n'
    for name,c in comparisons.items():
        text+=f"|{name}|{c['delta']:+.9f}|{c['session_cluster_CI95']}|{c['sessions_improved']}/{c['sessions']}|\n"
    text+='\n|제공 command|행 수|BASE|COMMAND|전체 점수 변화 기여|\n|---|---:|---:|---:|---:|\n'
    for label in LABELS:
        a=models['BASE']['semantic'][label];b=models[ARM]['semantic'][label]
        if b['n']:
            text+=f"|{label}|{b['n']}|{a['PREFIX']:.9f}|{b['PREFIX']:.9f}|{b['score_contribution']-a['score_contribution']:+.9f}|\n"
        else:
            text+=f'|{label}|0|N/A|N/A|0|\n'
    c=comparisons['BASE'];q=comparisons['QREFINE']
    text+='\n고정 terminal 판정: '+('BASE 대비 개선 관측. ' if c['delta']<0 else 'BASE 대비 개선 없음. ')
    text+=('기존 QREFINE보다도 낮음. ' if q['delta']<0 else '기존 QREFINE 최선 terminal을 넘지 못함. ')
    if c['CI_includes_zero']:
        text+='BASE 비교 CI가0을 포함하므로 확정적 개선으로 해석하지 않는다. '
    text+='\n\n좌회전은 V0 한 session뿐이고 유턴은 없어 일반화를 별도로 보장하지 않는다. '
    text+='일반 주행에는 회전/차선변경이 포함된다. 세 prefix 항은 누적 평균이며 3초 endpoint가 아니다.\n'
    if counterfactual:
        text+=f"\n같은 학습 모델에서 command를 LANE_KEEP으로 고정한 진단: PREFIX {counterfactual['all_LANE_KEEP_PREFIX']:.9f}, 원래 지시 {counterfactual['actual_command_PREFIX']:.9f}. 독립 학습 대조가 아니다.\n"
    text+='\n추가 FULL/스윕/공식 제출은 자동 실행하지 않았다.\n'
    (REPORT/'RESULTS_KO.md').write_text(text)
    with (REPORT/'results.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['run','PREFIX','L2_1s','L2_2s','L2_3s','nonstop_PREFIX'])
        for name,m in models.items():
            writer.writerow([name,*[m[k] for k in ('PREFIX','L2_1s','L2_2s','L2_3s')],m['groups']['nonstop']['PREFIX']])
    print(json.dumps({'PREFIX':{name:m['PREFIX'] for name,m in models.items()},'comparisons':comparisons}),flush=True)

if __name__=='__main__':
    main()
