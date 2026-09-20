"""Read-only terminal diagnosis plus ONE fixed equal-weight output average.

No model fitting, coefficient sweep, label-dependent selection, FULL or upload.
The average is computed from saved DEV predictions, not a deployed ensemble.
"""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np

ROOT=Path('/NHNHOME/data/sukim/adcl')
REPORT=ROOT/'reports/a2_motion_fresh_20260919'
CONTROL=ROOT/'work_dirs/md_a2_scene_extensions_20260918/A2-QREFINE-NOM-s1'
FRESH=ROOT/'work_dirs/a2_motion_fresh_20260919/A2-FRESH-NUIM-s1'
W=np.array([11,11,5,5,2,2],dtype=np.float64)/36


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=json.loads((CONTROL/'final_eval.json').read_text())
    fresh=json.loads((FRESH/'final_eval.json').read_text())
    a,b=base['records'],fresh['records']
    key=lambda records:[(r['session'],r['scenario'],int(r['frame'])) for r in records]
    assert key(a)==key(b)
    gt=np.array([r['gt_abs_xy'] for r in a],dtype=np.float64)
    assert np.array_equal(gt,np.array([r['gt_abs_xy'] for r in b],dtype=np.float64))
    bp=np.array([r['pred_abs_xy'] for r in a],dtype=np.float64)
    fp=np.array([r['pred_abs_xy'] for r in b],dtype=np.float64)
    buckets=np.array([r['bucket'] for r in a]);sessions=np.array([r['session'] for r in a])
    unique=sorted(set(sessions));ix=[np.flatnonzero(sessions==s) for s in unique]
    draws=np.random.default_rng(0).integers(0,len(ix),(20000,len(ix)))
    counts=np.array([len(i) for i in ix]);stats={};scores={}
    for name,pred in [('QREFINE',bp),('FRESH',fp),('FIXED_50_50',.5*(bp+fp))]:
        e=np.linalg.norm(pred-gt,axis=-1);d=e@W;scores[name]=d
        stats[name]={'PREFIX':float(d.mean()),'L2_1s':float(e[:,:2].mean()),
            'L2_2s':float(e[:,:4].mean()),'L2_3s':float(e.mean()),'endpoint_3s':float(e[:,-1].mean()),
            'waypoint_L2':e.mean(0).tolist(),
            'first2s_contribution':float((e[:,:4]*W[:4]).sum(1).mean()),
            'last_two_points_contribution':float((e[:,4:]*W[4:]).sum(1).mean()),
            'groups':{g:{'n':int((buckets==g).sum()),'PREFIX':float(d[buckets==g].mean()),
                           'contribution':float(d[buckets==g].sum()/len(d))} for g in sorted(set(buckets))},
            'sessions':{s:{'n':int((sessions==s).sum()),'PREFIX':float(d[sessions==s].mean())} for s in unique}}
    comparisons={}
    for left,right in [('FRESH','QREFINE'),('FIXED_50_50','QREFINE'),('FIXED_50_50','FRESH')]:
        delta=scores[left]-scores[right];sums=np.array([delta[i].sum() for i in ix])
        boot=sums[draws].sum(1)/counts[draws].sum(1)
        comparisons[f'{left}_minus_{right}']={'delta':float(delta.mean()),
            'CI95':np.quantile(boot,[.025,.975]).tolist(),'sessions_improved':int((sums<0).sum()),
            'group_contribution_delta':{g:float(delta[buckets==g].sum()/len(delta)) for g in sorted(set(buckets))}}
    evaluations=[r for r in map(json.loads,(FRESH/'metrics.jsonl').read_text().splitlines()) if r.get('kind')=='eval']
    result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'same1998 DEV rows; both completed20554-update models; no retraining',
        'fixed_average':{'weights':[.5,.5],'weight_sweep_performed':False,'deployed_or_server_scored':False,
            'raw_adapter_parity_and_target_device_latency':'not measured for this pair',
            'GT_or_status_based_candidate_selection':False,
            'test_or_FULL_predictions_used':False,'timing':'post-terminal exploratory saved-prediction check'},
        'metrics':stats,'comparisons':comparisons,
        'XY_error_flattened_correlation':float(np.corrcoef((bp-gt).ravel(),(fp-gt).ravel())[0,1]),
        'fresh_eval_curve':[{'step':r['step'],'PREFIX':r['official_d3']} for r in evaluations],
        'source_predictions':{str(p.relative_to(ROOT)):file_sha(p) for p in (CONTROL/'final_eval.json',FRESH/'final_eval.json')},
        'limitations':['Same repeatedly used DEV11sessions; bootstrap is conditional.',
            'Fresh initialization alters total historical ETRI exposure; it does not prove a local minimum.',
            'Lower late-stage points are evidence for considering more training, not a guarantee.',
            'The fixed average has two independently trained backbones; inference costs are not shared automatically.']}
    (REPORT/'terminal_interpretation_20260920.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 2026-09-20 terminal 해석 및 고정 1:1 평균','',
        'C2F/FRESH는 각각20,554 update 완료. 본 보고의 1:1 평균은 완료 후 저장 예측으로 계산한',
        '추가 분석이며 신규 학습이나 배포 완료 결과가 아니다. 계수 스윕·GT별 선택을 하지 않았다.',
        '', '|후보|PREFIX|L2_1s|L2_2s|L2_3s|', '|---|---:|---:|---:|---:|']
    for name,v in stats.items():lines.append('|'+name+'|'+ '|'.join(f"{v[k]:.9f}" for k in ('PREFIX','L2_1s','L2_2s','L2_3s'))+'|')
    lines += ['', 'L2_1s/2s/3s는 앞2/4/6점 누적 평균이다. 마지막 열은3초 endpoint가 아니다.',
        '', '## 실제 변화', '',
        '- C2F 단독0.164204741은 QREFINE0.164251767과 거의 같다. 일반 주행은0.168445113→0.168979542로 소폭 악화해 추가 확대를 우선하지 않는다.',
        '- FRESH는 전체3.38%, 일반 주행7.82% 개선. 첫2초 기여0.122951391→0.114141415로0.008809976 감소했다.',
        '- 대신 마지막 두 포인트 기여가0.041300376→0.044565844로0.003265468 증가해 순개선은0.005544508이다.',
        '- 일반 주행 기여 개선0.012360238 중 depart/steady 악화가합계0.006815731을 상쇄했다. 작은 집단의 회귀를 숨기지 않는다.',
        '- GT tangent 동일 mask 종방향 절대오차0.143405238→0.119448767, 횡방향0.060901533→0.071342948. 두 값은 PREFIX의 가산 성분이 아니다.',
        '- 인지 지표도 악화: occupancy IoU0.458116→0.380721, lane IoU0.479192→0.356890. 입력 경계는 같지만 표현 학습이 전반적으로 우월하다는 결론은 아니다.',
        '', '## 고정 평균과 불확실성', '']
    for name,c in comparisons.items():
        lines.append(f"- {name}: Δ{c['delta']:+.9f}, 11-session paired95%CI {c['CI95']}, 개선{c['sessions_improved']}/11.")
    lines += ['',f"XY 오차 상관은{result['XY_error_flattened_correlation']:.6f}. 이번 두 모델의 오차가 달라 평균에서 추가 이득이 관찰됐다.",
        '평균은 정지/출발의 기존 대조 수준을 완전히 회복하지 못한다. DEV 가중치 선택·두 모델 실행비용·raw parity 검증을 별도로 다뤄야 한다.',
        '', '## 다음 판단', '',
        'FRESH 계보를 다음 주력 후보로 보존한다. 후반0.184436→0.162706→0.158707로 내려왔지만',
        'cosine이 끝난 terminal 이후 추가 학습 이득을 보장하지 않는다. 정지/출발·횡방향·인지 지표 회귀를',
        '함께 추적해야 한다. 이번 확인으로 추가 학습·FULL·공식 제출을 새로 시작하지 않았다.']
    (REPORT/'INTERPRETATION_20260920_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'metrics':stats,'comparisons':comparisons,'error_correlation':result['XY_error_flattened_correlation']},indent=2))


if __name__=='__main__':main()
