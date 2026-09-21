"""Render the completed frozen error/cost audit; no model fitting."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();out=a.directory
    data=json.loads((out/'STATISTICS.json').read_text());cost=json.loads((out/'RTX4090_INPUT_COST.json').read_text());reuse=json.loads((out/'RTX4090_SHARED_HISTORY_COST.json').read_text())
    assert cost['completed'] and reuse['completed']
    full=data['models']['FULL_infit_V0'];dev=data['models']['DEV_PROGRESS_heldout'];direct=data['models']['DEV_DIRECT_heldout'];ut=data['models']['FULL_infit_UTURN_train375'];all_=full['all']
    cm=lambda v:f'{100*v:.2f}'
    pct=lambda v:f'{100*v:.2f}%'
    geom=[('near_straight_lt5deg','직진에 가까움 <5°'),('gentle_bend_5to15deg','완만한 굽음 5~15°'),('left_bend_ge15deg','좌측 굽음 ≥15°'),('right_bend_ge15deg','우측 굽음 ≥15°'),('low_motion_or_partial_stop','저속·일부 정지')]
    sem=[('LANE_KEEP','차로 유지'),('TURN_LEFT','좌회전 command'),('TURN_RIGHT','우회전 command'),('LANE_CHANGE_L','좌 차로 변경'),('LANE_CHANGE_R','우 차로 변경')]
    curve=data['FULL_training']['learning_curve'];server=data['server_block_decomposition'];train=data['FULL_training']
    lines=[
    '# FULL 잔여 오차와 영상 입력 확대 비용 — 2026-09-21',
    '',
    '**판정:** 종방향이 여전히 가장 크지만 횡방향도 다음 개선 대상이다. 회전 한 건의 오차는 크고, 전체 횡오차의 대부분은 표본이 많은 직진에 가까운 일반 주행에서 발생한다. 모든 현재 카메라는 이미 사용 중이다. 더 세밀한 과거 전방 특징을 공통 scene에 전달하는 저비용 후보와, 원본 RGB에서 현재 scene 해상도를 높이는 후보가 남아 있다.',
    '',
    '이번 작업은 저장 예측 재계산, FULL 고정 가중치의 유턴 진단 375행, RTX4090 비용 시제품 평가다. 신규 학습·checkpoint 변경·공식 제출은 하지 않았다.',
    '',
    '## 1. 통계의 범위',
    '',
    '|대상|행/세션|PREFIX|해석|','|---|---:|---:|---|',
    f'|공식 FULL 서버|1,125 clips|{data["server_user_supplied"]["score"]:.9f}|사용자 전달 집계값. test GT·행별 오차 없음|',
    f'|동일 FULL, 기존 V0|1,998 / 11|{all_["PREFIX"]:.9f}|FULL 학습에 포함된 **in-fit** 진단|',
    f'|DEV H4-PROGRESS|1,998 / 11|{dev["all"]["PREFIX"]:.9f}|해당 DEV 모델이 학습하지 않은 V0. 반복 개발에 사용됨|',
    f'|FULL 유턴 command probe|375 / 3|{ut["all"]["PREFIX"]:.9f}|train310 내 해당 command 전 행, **in-fit**. V0와 합치지 않음|',
    '',
    f'FULL checkpoint SHA256: `{data["server_user_supplied"]["provenance"]["checkpoint_sha256"]}`. 기준 source `{data["inspected_source_commit"]}`. FULL은 376 scene / 101,520행 / 24,931 update / effective batch16, 약 3.93회 노출. 학습 시간 {train["elapsed_seconds"]/3600:.2f}시간, nonfinite {train["nonfinite_count"]}회.',
    '',
    'FULL의 V0 학습 내 곡선: '+' → '.join(f'{v["step"]:,} step **{v["official_d3"]:.6f}**' for v in curve)+'. 계속 하락했으나 학습 내 곡선만으로 추가 FULL 학습의 서버 이득을 판단할 수 없다.',
    '',
    '## 2. 공식 서버 수치에서 확실히 분해되는 것',
    '',
    '`L2_1s/2s/3s`는 첫 2/4/6점 누적 평균이다. 공식 가중치는 `[11,11,5,5,2,2]/36`이다. 아래 쌍 평균은 공식 집계에서 정확히 역산했다. 개별 0.5초 시점 오차는 역산할 수 없다.',
    '',
    '|평가 포인트|두 포인트 평균 L2 (m)|전체 PREFIX 기여|비율|','|---|---:|---:|---:|']
    for name,mean,c in zip(['0.5·1.0초','1.5·2.0초','2.5·3.0초'],server['pair_point_mean_L2'],server['PREFIX_contribution']):
        lines.append(f'|{name}|{mean:.6f}|{c:.6f}|{pct(c/data["server_user_supplied"]["score"])}|')
    lines += ['',f'첫 2초가 **{pct(server["first2s_score_share"])}**를 차지한다. 0.12까지 필요한 감소량은 **0.013684828, 현재의 10.24%**다. 마지막 두 포인트도 29.81%라 개선 가치가 있지만 첫 2초를 계속 추적해야 한다. 아래 종·횡/command 통계를 공식 test의 분해로 해석하지 않는다.',
    '', '## 3. 횡방향이 큰가', '',
    'GT 각 0.5초 구간의 chord 방향으로 위치 오차를 종·횡 투영했다. GT 구간 길이 >0.05m인 포인트만 사용하며, 공식 시점 가중치의 유효 합으로 나눈 MAE다. 정지에 가까워 tangent가 불안정한 점은 방향 통계에서 제외하되 PREFIX에는 포함한다.',
    '', '|모델/범위|종방향 MAE|횡방향 MAE|','|---|---:|---:|',
    f'|FULL in-fit V0|{cm(all_["longitudinal_MAE"])}cm|{cm(all_["lateral_MAE"])}cm|',
    f'|DEV H4-PROGRESS|{cm(dev["all"]["longitudinal_MAE"])}cm|{cm(dev["all"]["lateral_MAE"])}cm|',
    f'|DEV H4-DIRECT|{cm(direct["all"]["longitudinal_MAE"])}cm|{cm(direct["all"]["lateral_MAE"])}cm|',
    '',
    'FULL에서 횡 MAE는 종 MAE의 55.88%다. L2를 `long²/L2 + lat²/L2`로 기술적으로 배분하면 종 68.76%, 횡 29.93%, tangent 무효 1.32%다. 이것은 현재 오차 벡터의 기술적 분해이며, 각 성분을 고쳤을 때 회수할 수 있는 점수나 독립 원인 기여도가 아니다. 종·횡 MAE를 더해 PREFIX로 해석하지 않는다.',
    '',
    f'부호 평균은 종 {cm(all_["longitudinal_signed_mean"])}cm, 횡 {cm(all_["lateral_signed_mean"])}cm로 절대오차보다 작다. 전체에 같은 상수 offset을 주면 해결된다는 근거는 없다.',
    '',
    '|시점|FULL 점별 L2|종 MAE, 유효점|횡 MAE, 유효점|','|---|---:|---:|---:|']
    for i,t in enumerate(np.arange(1,7)/2):lines.append(f'|{t:.1f}s|{cm(all_["point_L2"][i])}cm|{cm(all_["longitudinal_point_MAE"][i])}cm|{cm(all_["lateral_point_MAE"][i])}cm|')
    lines += ['', '횡오차는 뒤쪽에서 빠르게 커진다. 직진에 가까운 구간만 보아도 1초 2.46cm → 2초 7.17cm → 3초 16.02cm다. 작은 방향 오차가 누적되는 패턴과 맞지만, 영상 해상도·lane 인식·decoder 결합 중 어느 것이 원인인지 이 통계만으로 확정할 수 없다.',
    '', '## 4. 직진, 좌·우회전, 차로 변경', '',
    '**실제 GT 궤적의 굽음별 통계**. 현재 ego 전방을 0°로 두고 미래 6구간의 최대 절대 방향을 사용한다. 모든 구간 길이 >0.05m, 총 길이 ≥3m인 행을 분류했다. 좌/우는 마지막 구간 방향 부호다. 분류 문턱은 점수를 보기 전에 고정했으며 도로의 법적 maneuver 종류를 뜻하지 않는다.',
    '', '|GT 기하|행|PREFIX|종 MAE|횡 MAE|전체 횡 절대오차 중 비중|','|---|---:|---:|---:|---:|---:|']
    for k,label in geom:
        v=full['geometry'][k];lines.append(f'|{label}|{v["n"]:,}|{v["PREFIX"]:.6f}|{cm(v["longitudinal_MAE"])}cm|{cm(v["lateral_MAE"])}cm|{pct(v["lateral_abs_total_share"])}|')
    lines += ['', '**회전은 건당 어렵지만, 누적 횡오차는 직진에 가까운 구간이 76.11%로 가장 크다.** 같은 비중이 held-out DEV에서도 76.29%여서 FULL의 학습 내 표본에서만 생긴 순위는 아니다. 좌·우 굽음 50행은 3초 횡 MAE가 각각 1.35m/0.88m로 크지만 해당 표본은 2/3개 세션에 집중되어 있다.',
    '', '**제공 command 기준**은 별도로 집계했다. `LANE_KEEP`에는 굽은 도로도 포함되므로 이를 직진이라고 부르지 않는다.',
    '', '|제공 command|행/세션|PREFIX|횡 MAE|전체 횡 절대오차 중 비중|','|---|---:|---:|---:|---:|']
    for k,label in sem:
        v=full['semantic'][k];lines.append(f'|{label}|{v["n"]}/{v["sessions"]}|{v["PREFIX"]:.6f}|{cm(v["lateral_MAE"])}cm|{pct(v["lateral_abs_total_share"])}|')
    lines += ['', '현재 V0의 좌/우회전 command 중 실제 굽음 ≥15°인 21/24행을 따로 보면, 마지막 구간 회전량이 GT보다 작은 비율은 80.95%/100%, 회전 방향으로 부호를 맞춘 평균 부족량은 9.72°/14.48°다. **덜 도는 현상은 관측됐지만 이유는 아직 미확정**이다. 같은 세션의 인접 행이므로 21/24개의 독립 시험으로 보지 않는다.',
    '', '## 5. 유턴: V0가 보지 못한 항목', '',
    'V0에는 제공 U_TURN command가 0개다. 이번에는 FULL 가중치를 고정하고 train310에서 해당 command인 375행을 모두 추론했다. 오류를 보고 행을 고르지 않았으며 모델 parameter/buffer hash는 평가 전후 동일했다.',
    '', '|유턴 probe 집단|행|PREFIX|','|---|---:|---:|',f'|전체|375|{ut["all"]["PREFIX"]:.6f}|']
    for k,label in [('nonstop','일반 주행'),('depart','출발'),('steady','정지 유지')]:lines.append(f'|{label}|{ut["groups"][k]["n"]}|{ut["groups"][k]["PREFIX"]:.6f}|')
    lines += ['', f'움직임이 충분하고 좌측 굽음 ≥15°인 119행은 PREFIX **{ut["geometry"]["left_bend_ge15deg"]["PREFIX"]:.6f}**, 횡 MAE **{cm(ut["geometry"]["left_bend_ge15deg"]["lateral_MAE"])}cm**다. 유턴 쪽에 약점은 있다. 다만 375행은 3세션이며 인접 행들이고 모두 학습에 포함됐다. 세션 평균은 0.552697 / 0.047162 / 0.385956로 큰 차이가 있다.',
    '', 'U_TURN은 현재 제공된 maneuver label이다. 이번 분류에서 충분한 움직임 조건을 만족하면서 3초 마지막 GT 구간 방향이 135° 이상 돌아선 행은 0개다. 유턴 전 과정/완성도 평가로 해석하지 않는다. Test 입력에는 U_TURN 12/1,125개가 있지만 정답이 없어 test 유턴 오차나 그 개선 기여를 계산할 수 없다. 이 비율로 train 통계를 재가중하지 않았다.',
    '', '## 6. 아직도 일반 주행이고, 감가속 타이밍만의 문제는 아니다', '',
    f'FULL V0에서 nonstop 1,875행이 전체 점수의 **{pct(full["groups"]["nonstop"]["score_share"])}**다. 정지 유지 99행은 PREFIX {full["groups"]["steady"]["PREFIX"]:.6f}, 출발 24행은 {full["groups"]["depart"]["PREFIX"]:.6f}다.',
    '', '미래 첫 2초의 네 구간 평균 진행속도 범위가 0.5m/s 이하인 일반 주행은 1,306행, PREFIX **0.101952**, 전체 기여 **0.066642 (62.51%)**다. 범위가 더 큰 569행은 **0.133395**, 기여 **0.037989 (35.63%)**다. 변화가 큰 행이 건당 어렵지만, 변화가 작은 행의 정밀도도 개선해야 한다. 이는 미래 구간 GT로 만든 진단이며 현재 실제 vx나 외부 status 입력이 아니다.',
    '', '오류 상위 1%가 전체의 4.43%, 상위 10%가 26.58%를 차지한다. 몇 개 최악의 회전만 고치면 전체 문제가 끝나는 분포는 아니다. 첫 구간 진행속도 10m/s 이상인 행은 전체 횡오차의 약 63.75%다. 고속 여부만으로 원인을 확정할 수 없지만 일반 주행의 미세 방향·진행량 오차가 넓게 남아 있다.',
    '', '동일 예산 DEV에서 PROGRESS는 DIRECT보다 전체는 좋지만 nonstop은 0.148017→0.154891로 악화했고, 횡 MAE도 5.98→6.83cm로 증가했다. 기존 저장 예측의 PROGRESS 길이 + DIRECT 방향 결합은 0.145104였으므로 길이/방향 표현의 절충을 검증할 이유가 있다. 이 결합값은 학습된 새 모델이나 서버 예측이 아니다.',
    '', '## 7. 현재 영상 입력을 실제 코드/shape로 확인', '',
    '|경로|현재 설정|','|---|---|',
    '|Backbone|공유 ResNet50 + 128D FPN|',
    '|현재 영상|front, front-left/right, rear-left/right, rear-wide **6개 모두**, 768×432|',
    '|과거 영상|front만 −0.1/−0.2/−0.5/−1.0초, 4장|',
    '|과거 scene 경로|384×216로 축소한 front 4장|',
    '|Motion 경로|현재+과거 front 5장, 768×432|',
    '|실제 backbone 호출|현재6장 고해상도 + scene용5장 저해상도 + motion용5장 고해상도|',
    '|FPN 공간 해상도|768 입력은 96×54 / 48×27, 384 입력은 48×27 / 24×14|',
    '|Motion memory|4시점 × 12×16 token. 추가 temporal read가 각 waypoint별로 읽음|',
    '|Scene|64×48 BEV, 세 높이/두 FPN level의 영상 투영, QREFINE|',
    '', '고유 RGB는 10장이지만 여러 경로에서 16장 분량의 인코딩을 한다. 저해상도 scene stack의 현재 front는 인코딩 후 사용하지 않고, 현재 front 고해상도도 current6/motion5에서 중복된다. 이번에는 성능 의미가 분명한 과거 특징 공유 한 가지만 비용 시제품으로 확인했다.',
    '', '원본 fixture의 6-camera JPEG는 모두 1920×1536이고, 실제 입력은 undistort → 1920×1080 crop → 768×432 캐시다. 원본 train tar 376개가 존재하고 현재 cache는 etri_768만 확인했다. 따라서 **원본에서 더 세밀하게 읽을 정보는 남아 있다.** 768 캐시를 1152로 보간하는 것은 새로운 세부 정보를 복원하지 않는다. 1152 실험은 원본에서 동일 crop/geometry로 별도 생성해야 한다. H4가 −0.1/−0.2초를 사용하므로 과거 VAD용 2Hz cache 레시피를 그대로 쓰면 안 된다.',
    '', '## 8. RTX4090에서 직접 측정한 비용', '',
    '동일 FULL checkpoint, B1 BF16 + FP32 planner, raw train fixture 2개, 각각 warmup30/repeat100. 표는 두 clip 중 큰 median/p95다. 모든 backbone을 포함한 전체 forward이며 파일읽기/undistort/crop/전송은 제외한다. 제출 graph와 같은 FlopCounterMode FP32 Global 합을 사용했다.',
    '', '확대 입력은 계산 비용을 위한 시제품이다. 입력 크기/필요한 correlation radius를 변경했으며 radius가 바뀐 fuse는 초기화했다. **정확도 평가나 신규 학습을 수행한 결과가 아니다.**',
    '', '|비용 시제품|현재6 / 과거scene / motion 너비|GFLOPs|median ms|p95 ms|','|---|---|---:|---:|---:|']
    names={'BASE':'현재 제출 graph','SCENE_HISTORY_768':'과거 scene만 확대, 별도 인코딩','CURRENT_SCENE_1152':'현재6 scene만 확대','MOTION_1152_R6':'Motion만 확대, radius6','ALL_1152_R6':'전체1.5배, radius6','ALL_1536_R8':'전체2배, radius8'}
    for k,v in cost['cases'].items():lines.append(f'|{names[k]}|{v["image_hw"][1]} / {v["scene_history_hw"][1]} / {v["motion_hw"][1]}|{v["gflops"]:.1f}|{v["max_clip_median_ms"]:.2f}|{v["max_clip_p95_ms"]:.2f}|')
    lines += [f'|**768 과거 특징을 공유**|768 / 768 / 768|**{reuse["gflops"]:.1f}**|**{reuse["max_clip_median_ms"]:.2f}**|**{reuse["max_clip_p95_ms"]:.2f}**|',
    '', '전체 1.5배는 약 55ms라 계산 여유가 있다. 전체 2배/radius8은 약 101ms로 100ms 기준을 이미 넘는다. FLOPs가 cut-off 7,053G 이내여도 latency 조건은 별도다. 두 fixture 측정이 모든 clip/실행 환경의 최대 시간을 보장하지 않는다. 서버 elapsed_ms303은 이 표와 다른 harness 값이다.',
    '', '**추가로 확인한 유용한 구조:** motion에 이미 있는 조건화 전 768 과거 FPN을 shared scene에도 전달하면, 384 scene 인코딩을 없앨 수 있다. 768 scene을 따로 계산한 시제품과 scene/motion/state/history/plan의 FP32 최대 차이가 두 fixture 모두 0이었다. 따라서 비용은 730.0→658.0G, 26.1→23.9ms로 줄면서 scene이 더 세밀한 과거 특징을 읽을 수 있다.',
    '', '이 parity는 **새 768-scene 방식끼리**의 비교다. 384 scene으로 학습한 제출 baseline과의 출력 parity가 아니며, 재학습/적응 없이 정확도가 유지된다는 의미도 아니다. 더 풍부한 정보가 motion에 이미 있었음을 scene에 직접 전달하는 변경이고, 전체 모델에 새 RGB를 추가하는 것은 아니다. 공유되는 것은 raw status 조건화 전 FPN이므로 status→motion/state의 새 경로를 만들지 않는다.',
    '', '## 9. 다음 실험의 우선순위', '',
    '|순서|한 가지 변경|확인할 가설/주의|','|---|---|---|',
    '|1|**기존 768 과거 FPN을 shared scene에 공유**|더 선명한 과거 scene 근거가 일반 주행의 방향/진행량을 개선하는가. cache 재생성 없이 준비 가능. 같은 DEV parent·추가 예산 control과 비교|',
    '|2|**현재 6-camera scene을 원본 1152×648로**|차선/멀리 있는 도로·차량의 세부 정보가 부족한가. motion·history는 먼저 유지. 현재 비용 37.9ms로 여유 확인|',
    '|별도 축|길이/방향 readout 분리|이미 저장 예측에서 관측한 상보성을 학습 모델로 회수하는가. 해상도 변경과 처음부터 결합하지 않음|',
    '|그다음|측면 과거 또는 −2초 관측 한 종류|현재6cam은 이미 전부 사용. front-only history에 새 시점/방향을 추가하는 변경. 과거 SIDE 결과와 구분하고 대조 필요|',
    '', '과거 SIDE-SCENE은 20,554 update 완료 후 BASE0.165511→0.172406으로 악화했고, C2F도0.164205로 큰 이득을 보이지 않았다. 그래서 측면 4장을 그대로 반복하거나 단순 correlation 확대를 첫 순위로 두지 않는다. 이 결과가 현재 H4-PROGRESS의 모든 측면/고해상도 방법을 기각하는 것은 아니다. 현재 기록에서 H4-PROGRESS의 native1152/1536 동일 예산 완료 비교는 확인되지 않았다.',
    '', '더 긴 history는 가능하지만 현재 WaypointTemporalRead가 `[B,4,192,128]`을 고정 검사한다. 이미지·시간/pose 정렬·history supervision·memory를 함께 고쳐야 한다. 기존 −0.1/−0.2초를 버리기보다 −2초 한 시점을 추가하는 가설이 낫지만, 그것이 미래 감속/조향을 더 잘 예측하는지는 미확인이다. 제공 status의 기존 H4 producer는 유지하고, pose는 해당 영상의 정렬 범위로만 사용한다.',
    '', '새 run은 DEV 가중치로 검증하고 FULL 가중치/feature를 DEV로 반입하지 않는다. 기존 FULL 제출은 보존한다. 단순한 in-fit 개선·FLOPs 여유·낮은 oracle을 서버0.12 달성 보장으로 사용하지 않는다.',
    '', '## 10. 재현 파일', '',
    '- `STATISTICS.json`, `GROUP_SUMMARY.csv`, `*_rows.csv`: 같은 row/GT 재계산, command/기하/속도/세션 통계.',
    '- `UTURN_train_probe.json`: 375행 고정 FULL 예측과 hash. 원격 진단 산출물로 보존.',
    '- `RTX4090_INPUT_COST.json`, `RTX4090_SHARED_HISTORY_COST.json`: 측정 원문/shape/parity.',
    '- `ERROR_AND_COST.png` / `.pdf`: 정적 요약 그림.',
    '- `experiments/a2_full_error_audit_20260921/`: 재계산·유턴 평가·비용 측정·보고서 생성 소스.',
    '',
    '기존 근거: `models/motiondrive_v2_inputs.py`의 원본 기하/리사이즈, `matching_resolution.py`와 `factorized_model.py`의 별도 인코딩, `temporal_model.py`의 시점별 memory, `progress_model.py`의 구간 출력, `reports/md_a2_scene_extensions_20260918/TERMINAL_REVIEW_KO.md`의 SIDE 결과, `reports/a2_after_submission_20260921/RESULTS_KO.md`의 길이/방향 재조합.',
    '']
    (out/'RESULTS_KO.md').write_text('\n'.join(lines))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    t=np.arange(1,7)/2
    for key,label,c in [('point_L2','L2, all points','#333333'),('longitudinal_point_MAE','Longitudinal MAE, valid tangent','#2468aa'),('lateral_point_MAE','Lateral MAE, valid tangent','#e17c24')]:axes[0,0].plot(t,np.array(all_[key])*100,'o-',label=label,c=c)
    axes[0,0].set(title='FULL in-fit V0: errors grow over the horizon',xlabel='Future time (s)',ylabel='Error (cm)');axes[0,0].legend(fontsize=9);axes[0,0].grid(alpha=.2)
    keys=[g[0] for g in geom[:4]];labels=['Near straight\nn=1,648','Gentle bend\nn=149','Left bend\nn=23','Right bend\nn=27']
    vals=[100*full['geometry'][g]['lateral_MAE'] for g in keys]
    bars=axes[0,1].bar(labels,vals,color=['#2468aa','#3e9b8f','#e17c24','#b95451']);axes[0,1].bar_label(bars,fmt='%.2f');axes[0,1].set(title='Bends are harder per row',ylabel='PREFIX-weighted lateral MAE (cm)',ylim=(0,28))
    vals=[full['geometry'][k]['lateral_abs_total_share']*100 for k in keys]+[full['geometry'][geom[-1][0]]['lateral_abs_total_share']*100]
    labels=['Near straight','Gentle bend','Left bend','Right bend','Low motion / stop']
    bars=axes[1,0].barh(labels[::-1],vals[::-1],color=['#8a8a8a','#b95451','#e17c24','#3e9b8f','#2468aa']);axes[1,0].bar_label(bars,fmt='%.1f%%',padding=3)
    axes[1,0].set(title='Most lateral error comes from near-straight rows',xlabel='Share of weighted absolute lateral error (%)',xlim=(0,91))
    items=[('Shared 768 past FPN',reuse['max_clip_median_ms']),('Submitted BASE',cost['cases']['BASE']['max_clip_median_ms']),('Past scene 768',cost['cases']['SCENE_HISTORY_768']['max_clip_median_ms']),('Current scene 1152',cost['cases']['CURRENT_SCENE_1152']['max_clip_median_ms']),('Motion 1152 / r6',cost['cases']['MOTION_1152_R6']['max_clip_median_ms']),('All 1.5x / r6',cost['cases']['ALL_1152_R6']['max_clip_median_ms']),('All 2x / r8',cost['cases']['ALL_1536_R8']['max_clip_median_ms'])]
    bars=axes[1,1].barh([v[0] for v in items][::-1],[v[1] for v in items][::-1],color=['#b95451']+['#7898b5']*4+['#333333','#3e9b8f']);axes[1,1].bar_label(bars,fmt='%.1f',padding=3)
    axes[1,1].axvline(100,ls='--',c='#b95451',lw=1);axes[1,1].set(title='RTX4090 cost prototypes: accuracy untested',xlabel='Whole B1 forward median (ms), preprocessing excluded',xlim=(0,116))
    fig.suptitle('A2-H4-PROGRESS-FULL | In-fit V0 diagnostics and measured input costs\nOfficial server PREFIX: 0.133685; row-level test errors are unavailable',fontsize=14)
    for ext in ('png','pdf'):fig.savefig(out/f'ERROR_AND_COST.{ext}',dpi=180)
    plt.close(fig)
    print(out/'RESULTS_KO.md')

if __name__=='__main__':main()
