"""Summarize the completed, fixed-budget experiment and deployment artifacts."""
import json,hashlib,datetime,subprocess
from pathlib import Path
R=Path('/NHNHOME/data/sukim/adcl');D=R/'reports/a2_visual_teacher_20260919'

def item(path):
    path=Path(path);h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}

def main():
    s=json.loads((D/'state_on_off.json').read_text());v=json.loads((D/'visual_results.json').read_text())
    m=v['models'];t=m['VIS-TERMINAL'];parent=m['PARENT'];control=m['CONTROL']
    both=t['PREFIX']<parent['PREFIX'] and t['PREFIX']<control['PREFIX']
    decision=('관측상 control과 부모를 모두 개선한 DEV 후보로 보존한다. 반복 사용 V0의 불확실성을 함께 표시하고 FULL 이전은 별도 판단한다.' if both else
        '이번 교사 continuation을 새 최선 후보나 FULL 레시피로 승격하지 않는다. 기존 QREFINE fixed terminal과 G1 선택 중간점은 유지한다.')
    lines=['# A2 state/history 개입·RGB 교사 실행 결과 — 2026-09-19','',
        '고정 QREFINE DEV 부모의 state ON/OFF 평가와 RGB 교사 한 종류의 matched continuation을 마쳤다. 완료된 A2 FULL의 raw 배포 패키지도 별도로 확보했다.','',
        '## 1. 최종 수치','',
        '|조건|PREFIX|L2_1s|L2_2s|L2_3s|일반 주행 PREFIX|첫 2초 기여|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ['PARENT','CONTROL','VIS-1142','VIS-2284','VIS-TERMINAL']:
        x=m[name];lines.append(f"|{name}|{x['PREFIX']:.9f}|{x['L2_1s']:.9f}|{x['L2_2s']:.9f}|{x['L2_3s']:.9f}|{x['groups']['nonstop']['PREFIX']:.9f}|{x['first2s_PREFIX_contribution']:.9f}|")
    lines+=['',decision,'','주 비교는 예정된 3426 update terminal이다. 중간 두 점은 고정된 평가 시점의 관측이며 독립 검증이 아니다.','',
        '|비교|PREFIX 차이|11-session paired 95% CI|개선 session 수|',
        '|---|---:|---|---:|']
    for name,c in v['comparisons'].items():
        lines.append(f"|{name}|{c['PREFIX_delta']:+.9f}|{c['session_CI95']}|{c['sessions_improved']}/11|")
    lines+=['','CI는 같은 11개 session의 조건부 재표집이다. 학습 seed 분산, 숨은 test 성능 또는 DEV 선택 편향을 제거한 추정이 아니다.','',
        '## 2. 예측 state/history ON/OFF','',
        f"ON **{s['models']['ON']['PREFIX']:.9f}**, OFF **{s['models']['OFF']['PREFIX']:.9f}**. OFF−ON {s['OFF_minus_ON']['PREFIX']:+.9f}, CI {s['OFF_minus_ON']['session_CI95']}.",
        '제공 status와 continuous motion은 유지하고, planner의 sample별 예측 state/history 숫자만 MLP 직전에 0으로 만들었다. Bias constant token은 남는다. 내부 scene/motion/state/history 출력과 parameter/buffer는 같았다.',
        'OFF에서 일반 주행도 악화했다. 이번 고정 가중치 개입으로 제거 이득을 확인하지 못해 ON을 유지한다. OFF 학습의 원리적 실패를 입증한 것은 아니며 자동 제거 조합 실험은 하지 않는다.','',
        '## 3. 시각 학습의 실제 변경과 검사','',
        '- DINOv2 ViT-B/14 registers 하나를 frozen RGB teacher로 사용했다. 출처·revision·가중치 SHA·라이선스는 teacher_manifest.json과 TEACHER_NOTICE_KO.md에 있다.',
        '- 현재 6-camera의 조건화 전 상위 FPN 한 level만 비교했다. 같은 crop/flip/photometric 입력에서 dense patch token을 추출하고, 좌표·마스크를 맞췄다. CLS/register는 제외했다.',
        '- Teacher와 train-only projection 외에 A2 입력 경계, state ON, scene/motion/planner, 기존 loss 및 데이터 계보는 유지했다.',
        '- λ_vis=.25는 train 4batch의 gradient norm으로 미리 정한 단일 규칙의 상한값이다. V0로 계수를 조정하지 않았다. 초기 visual backbone/FPN gradient norm은 PREFIX+LEN의 약 1.16%였다. 최적 강도를 찾았다는 뜻은 아니다.',
        '- G0와 동일 부모 tensor·nominal 정책·추가 3426 update·fresh optimizer·LR/BN/증강을 확인했다. 모든 기록 step의 sample-order SHA와 LR가 일치했다. CUDA backward의 bitwise 재현은 주장하지 않는다.',
        f"- Train visual loss: {v['training_first_visual_loss']:.6f} → {v['training_last_logged_visual_loss']:.6f} (마지막 train log step {v['last_logged_train_step']}). 실제 상위 FPN RMS 변화 {v['parameter_updates']['student_upper_FPN_RMS_from_parent']:.8g}, projection RMS 변화 {v['parameter_updates']['projector_RMS_from_private_initialization']:.8g}.",
        '- Visual-only gradient가 student까지 흐르고 teacher에는 없음을 확인했다. Student의 실제 parameter 변화에는 기존 loss의 영향도 포함된다. Loss 감소를 planning 성공으로 대신하지 않는다.',
        '- Whole/microbatch loss·gradient, 부분 invalid/전체 invalid, spatial ramp, 상태 입력 의존성 검사를 통과했다. 최종 teacher state는 변하지 않았다.',
        '- 실제 저장한 teacher/projector 없는 student를 strict load했다. B1 최종 XY 차이는 0이고 inference tensor 구성은 QREFINE 부모와 같다. Single-head A2 FULL의 FLOPs를 이 학생에게 부여하지 않았다.',
        '- 최초 본 실행은 normalizer key guard에서 update 전에 종료했다. 기존 loss용 정규화와 visual 정규화를 분리한 r2가 본 결과다.',
        '', '구간 길이 MAE/부호 편향, 기존 producer의 진행량 그룹 분석, 종·횡 오차, occupancy/lane 및 state/history 지표는 visual_results.json에 모두 있다.',
        '이 결과는 완성된 QREFINE에서의 짧은 추가 학습 한 종류에 대한 것이다. 모든 RGB teacher·새 backbone·처음부터의 사전학습 효과나 A2 성능 상한을 검증한 것은 아니다. Teacher/층/λ/seed 스윕이나 새 FULL을 자동 시작하지 않는다.','',
        '## 4. 기존 A2 FULL 배포','',
        '- 이미 완료된 step24931을 사용했다. 재학습이나 DEV 역류는 없었다.',
        '- 8 raw fixture 입력/XY bitwise 일치, 실제 1125 clip 추론, finite/shape/clip isolation, 공식 방식의 **729.815616192G** counter, ZIP 검사를 완료했다.',
        '- 깨끗한 Docker에서 raw 두 clip 추론을 확인했다. 첫 clip 입력은 B200과 모두 일치하고, BF16 환경 간 최대 XY 차이는 약 2.829mm였다.',
        '- 로컬 제출 파일: ~/Downloads/A2-FULL-NOM-s1_submission_20260919/submission.zip. ZIP에는 submission.json 하나만 있다. 공식 업로드는 하지 않았다.',
        '- 이번 A2의 RTX4090 실측은 장비 접속 정보 확인이 필요하다. B200/4060 결과나 과거 MR 4090 값을 대신 사용하지 않았다.',
        '', '## 5. 추적 경로','',
        '- CURRENT_BASE.json / INPUT_POLICY.json: 부모·split·입력 정책.',
        '- control_reuse.json / COST_CHECK.json: 대조군과 행 순서·비용.',
        '- visual_results.json / results.csv / run_artifact_index.json: 최종 수치·가중치·예측 경로와 SHA.',
        '- reports/a2_full_submission_20260919/: FULL 별도 배포 검사.',
        '- 구현의 첫 work commit b71d6b6, GitHub mirror 12bd250. 종점 결과는 후속 커밋으로 기록한다.','']
    (D/'RESULTS_KO.md').write_text('\n'.join(lines))
    run=Path(v['checkpoint']).parent
    paths=[run/f for f in ['ckpt_step1142.pth','ckpt_step2284.pth','ckpt_step3426.pth','student_step3426.pth','manifest.json','experiment.json','metrics.jsonl','final_eval.json','predictions_step1142.json','predictions_step2284.json','teacher_unchanged.json']]
    index={'status':'completed','created_at_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'teacher_run':str(run),'artifacts':[item(p) for p in paths],
        'decision':decision,'new_FULL_started':False,'official_uploaded':False,
        'deployment_index':'reports/a2_visual_teacher_20260919/deployment_artifact_index.json',
        'RTX4090_latency':'pending actual hardware access'}
    (D/'run_artifact_index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(decision)

if __name__=='__main__':main()
