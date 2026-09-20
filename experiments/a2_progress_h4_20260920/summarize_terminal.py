"""Read saved terminal predictions and explain group/time tradeoffs; no training."""
from pathlib import Path
import json
import numpy as np

ROOT=Path('/NHNHOME/data/sukim/adcl')
REPORT=ROOT/'reports/a2_progress_h4_20260920'
RUNS=ROOT/'work_dirs/a2_progress_h4_20260920'
ARMS=('A2-H4-DIRECT','A2-H4-PROGRESS')
W=np.array([11,11,5,5,2,2],np.float64)/36

def main():
    result=json.loads((REPORT/'results.json').read_text())
    assert result['status']=='completed' and result['sample_stream']['all_compared_equal']
    summary={};identity=None;truth=None
    for arm in ARMS:
        run=RUNS/f'{arm}-s1'
        manifest=json.loads((run/'manifest.json').read_text())
        assert manifest['status']=='completed' and manifest['step']==20554 and manifest['nonfinite_count']==0
        records=json.loads((run/'predictions_step20554.json').read_text())['records']
        ids=[(r['row'],r['session'],r['scenario'],r['frame']) for r in records]
        pred=np.array([r['pred_abs_xy'] for r in records],np.float64)
        gt=np.array([r['gt_abs_xy'] for r in records],np.float64)
        if identity is None:identity=ids;truth=gt
        assert ids==identity and np.array_equal(gt,truth) and np.isfinite(pred).all()
        bucket=np.array([r['bucket'] for r in records]);error=np.linalg.norm(pred-gt,axis=-1)
        stats={}
        for name,mask in [('all',np.ones(len(error),bool))]+[(n,bucket==n) for n in ('nonstop','depart','steady')]:
            e=error[mask];first=e[:,:4]@W[:4];last=e[:,4:]@W[4:]
            stats[name]={'n':int(mask.sum()),'PREFIX':float((e@W).mean()),'per_point_L2':e.mean(0).tolist(),
                'first2s_group_contribution':float(first.mean()),'last_two_group_contribution':float(last.mean()),
                'first2s_total_contribution':float(first.sum()/len(error)),
                'last_two_total_contribution':float(last.sum()/len(error))}
        assert abs(stats['all']['PREFIX']-result['terminal']['arms'][arm]['PREFIX'])<1e-12
        summary[arm]=stats
    a,b=summary.values()
    delta={group:{k:b[group][k]-a[group][k] for k in ('PREFIX','first2s_group_contribution',
        'last_two_group_contribution','first2s_total_contribution','last_two_total_contribution')} for group in a}
    out={'arms':summary,'PROGRESS_minus_DIRECT':delta,'optimizer_updates_added':0,'new_GPU_inference':False,
        'finding':'Overall gain is from depart/steady; nonstop worsens in both first four and last two points.',
        'promotion':'Preserve single DEV candidate; no automatic main-model replacement, FULL or submission.',
        'session_comparison':result['terminal']['comparison']}
    (REPORT/'terminal_group_time_breakdown.json').write_text(json.dumps(out,indent=2)+'\n')
    terminal=result['terminal'];direct,progress=(terminal['arms'][arm] for arm in ARMS)
    percent=lambda old,new:100*(new/old-1)
    lines=['# H4 PROGRESS 최종 판정 — 2026-09-21','',
        '두 arm 모두20,554 update를 완료했다. Nonfinite0,412개 sample-order 로그 일치. 새 학습·GPU 재평가 없이 저장 예측을 분해했다.',
        '',f"전체 PREFIX {direct['PREFIX']:.9f} → **{progress['PREFIX']:.9f}**, {percent(direct['PREFIX'],progress['PREFIX']):+.2f}%. 새 단일 DEV 후보는 확보했지만 일반 주행 병목의 개선은 확인하지 못했다.",
        '', '|그룹|행 수|DIRECT|PROGRESS|그룹 평균 변화|전체 점수 변화 기여|',
        '|---|---:|---:|---:|---:|---:|']
    for name,label in [('nonstop','일반 주행'),('depart','출발'),('steady','정지 유지')]:
        aa,bb=a[name],b[name];contribution=(bb['PREFIX']-aa['PREFIX'])*aa['n']/1998
        lines.append(f"|{label}|{aa['n']}|{aa['PREFIX']:.9f}|{bb['PREFIX']:.9f}|{percent(aa['PREFIX'],bb['PREFIX']):+.2f}%|{contribution:+.9f}|")
    lines+=['',
        '전체 개선 −0.003034는 출발 −0.001383, 정지 유지 −0.008102의 이득에서 일반 주행 +0.006450의 손해를 합한 결과다.',
        '', '|누적 평균 L2|DIRECT|PROGRESS|', '|---|---:|---:|']
    for key in ('L2_1s','L2_2s','L2_3s'):lines.append(f"|{key}|{direct[key]:.9f}|{progress[key]:.9f}|")
    lines+=['',
        'L2_3s는 첫6점 평균이다. 3초 endpoint 자체는0.420691→0.455502로 악화했다.',
        f"전체 첫2초의 PREFIX 기여 {a['all']['first2s_group_contribution']:.9f}→{b['all']['first2s_group_contribution']:.9f}는 감소했다. 그러나 일반 주행 안에서는 {a['nonstop']['first2s_group_contribution']:.9f}→{b['nonstop']['first2s_group_contribution']:.9f}로 증가했다. 첫2초 전체 개선을 일반 주행 개선으로 해석하면 안 된다.",
        f"일반 주행의2.5/3초 점 기여도 {a['nonstop']['last_two_group_contribution']:.9f}→{b['nonstop']['last_two_group_contribution']:.9f}로 증가했다.",
        '',
        '동일 GT tangent mask의 종방향 절대오차는0.121123→0.122230, 횡방향은0.059775→0.068343이다. 이 두 값은 더해서 PREFIX가 되는 분해가 아니다.',
        '일반 주행 첫2초 공통 진행속도 오차 MAE는0.091975→0.089525m/s로 소폭 감소했으나 시간 변화 성분 RMS는0.092476→0.104884m/s로 증가했다. 단일 진행량 보조값의 개선만으로 실제 XY 개선을 주장할 수 없다.',
        '',f"Session paired95%CI {terminal['comparison']['session_CI95']}, 개선5/11session. 재사용한 V0의 한 seed 결과이며 독립 검증·서버 성능 보장이 아니다.",
        '',
        '판정: 정지·출발 성능이 좋은 단일 후보로 보존한다. 이번 결과를 일반 주행을 해결한 주력 변경으로 승격하거나 자동 FULL 학습하지 않는다. 이 예산에서의 결과로 출력 구조 전체의 성능 상한을 선언하지도 않는다.',
        '새 PROGRESS의 일반 주행 전체 기여만0.145355다. 출발·정지 오차를 모두 없애더라도0.12에 도달하지 못하므로 일반 주행 개선은 여전히 필요하다.',
        '기존 H4 입력 frozen CONT0.153511500보다 전체 점수는 낮지만 부모 계보·예산은 다르다. 다른 입력 정책의 기존3모델 평균0.146197790과는 단일/앙상블·학습 조건 차이를 구분한다.',
        '제공 status의5시점 input coverage와 query-only 경계는 이전 검사대로 유지한다. 운영국 개별 승인, 새로운 공식 제출 또는 서버0.15/0.12 결과를 주장하지 않는다.',
        '', '근거: result_step20554.json, terminal_group_time_breakdown.json, results.json, 각 run manifest.json.']
    (REPORT/'TERMINAL_REVIEW_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'reviewed','DIRECT':direct['PREFIX'],'PROGRESS':progress['PREFIX'],
        'overall_relative_percent':percent(direct['PREFIX'],progress['PREFIX']),'nonstop_relative_percent':percent(a['nonstop']['PREFIX'],b['nonstop']['PREFIX'])}))

if __name__=='__main__':main()
