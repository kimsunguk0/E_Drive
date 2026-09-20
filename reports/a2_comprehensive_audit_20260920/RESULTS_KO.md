# A2의 0.12 목표: 잔여 병목 종합 점검

2026-09-20. 기준 작업 HEAD `03b51aad517c7cadbb3ec946f34812ebeecefc47`.
이 문서는 완료 실험·소스 검토, 저장 예측 재분석, 새 고정 가중치 진단 결과다. 새 학습·optimizer update·FULL·공식 제출은 실행하지 않았다. GPU 0/1/2에서 진단을 마쳤고 0–3은 유휴다. GPU 4–7은 사용하거나 중단하지 않았다.

**현재 가장 큰 측정된 병목은 일반 주행의 첫 2초 진행량 정밀도다. 가장 유력한 원인 설명은 영상 기반 거리/진행량 표현의 정밀도와 새로운 session으로의 일반화가 함께 부족하다는 것이다. 어느 한 모듈이 유일한 원인이라고 입증한 것은 아니다.**

별도로 실제 배포 adapter에서 **status를 만드는 pose 11시점과 모델이 읽는 영상 5시점의 차이**를 확인했다. 이는 query-only 위치와 다른 규정 쟁점이며, 이전 점검이 놓친 항목이다. 아래 8절을 제출 후보 판단과 함께 읽어야 한다.

## 1. 무엇을 실제로 다시 확인했나

- 최신 FRESH / continuation / temporal-read 구현·커밋·완료 곡선과, MH4/QREFINE/SIDE/G/S/command/teacher/C2F/precision/residual의 기존 완료 기록.
- 5개 단일 후보의 동일 V0 1,998행, 37 scenes / 11 sessions 저장 예측·GT·row 일치.
- CONT step5710, CONT terminal6852, TEMPORAL terminal20554의 새 frozen forward. Strict load, 저장 예측 parity, parameter/buffer SHA 전후 일치. Update 0.
- 이전과 동일 seed20260920의 무증강 uniform train 1,024행을 새 3개 모델에서 평가. QREFINE/FRESH의 기존 결과와 row·GT가 동일하다.
- DEV train83,700행 + V0 1,998행의 미래/goal 좌표를 원본 pose에서 재계산. 347 scene의 원본 timestamp·pose SHA를 기록.
- 배포형 nominal status와 실측 시간 state target의 V0 전수 재현, train/V0 session 격리, 실제 raw adapter의 image getter 호출 시점.

모든 과거 모델을 재학습하거나 모든 원본 영상을 사람이 시각 판독한 것은 아니다. GT 센서 정확도·카메라 exposure 동기화는 독립적인 ground truth가 없어 이번 검사로 인증하지 못한다.

## 2. 현재 점수와 이번에 새로 확인한 평균

|후보|DEV PREFIX|조건|
|---|---:|---|
|QREFINE|0.164251767|과거 계보 fixed terminal|
|FRESH|0.158707259|공개 trunk 초기값, 20,554 update|
|FRESH-CONT step5710|0.153684060|같은 V0의 예정 중간점에서 선택|
|FRESH-CONT terminal6852|0.156334059|고정 추가학습 종점|
|TEMPORAL-READ terminal|0.154119411|FRESH와 동일 초기값·예산의 구조 비교|
|QREFINE + CONT5710, 고정 1:1|0.148555967|기존 저장 예측 평균|
|QREFINE + TEMPORAL, 고정 1:1|0.148557242|이번 상호 보완성 진단|
|CONT5710 + TEMPORAL, 고정 1:1|0.150232602|이번 상호 보완성 진단|
|QREFINE + CONT5710 + TEMPORAL, 고정 1:1:1|**0.146197790**|이번 상호 보완성 진단|

새 평균은 `PROTOCOL.json`에 고정한 조합만 계산했고 비율 최적화나 parameter 평균을 하지 않았다. 세 모델 평균은 기존 두 모델 평균보다 -0.002358177, 10/11 session 개선, session paired95%CI [-0.004564897,-0.000792710]이다. 반복 사용한 DEV이고 학습 seed 분산/선택 편향을 제거한 검증이 아니다.

0.15 미만의 **저장 예측 DEV 수치**는 확보했다. 세 모델 raw adapter·전체 FLOPs·RTX4090 시간·FULL 이전·서버 점수는 이 계산으로 확보된 것이 아니다. 현재 확인된 공식 서버 기준은 사용자 전달 MR FULL 0.185968928이다. DEV에서 서버 0.15나0.12를 약속할 수 없다.

## 3. 0.12까지 무엇을 줄여야 하나

기존 두 모델 평균을 기준으로:

|구분|행 수|그룹 PREFIX|전체 PREFIX 기여|
|---|---:|---:|---:|
|일반 주행 nonstop|1,875|0.149234990|**0.140047851**|
|출발 depart|24|0.378548679|0.004547131|
|정지 유지 steady|99|0.079939871|0.003960985|
|합계|1,998|—|0.148555967|

일반 주행은 전체 오차의94.3%다. 출발·정지를 완벽히 해결해도0.140048이 남는다. 세 모델 평균에서도 같은 가정의 floor는0.135638이다. 정지 회귀는 고쳐야 하지만 이것만으로0.12가 되지 않는다.

기존 평균의 첫2초 포인트 기여는0.109273944, 전체의73.56%다. 세 모델 평균도0.107229915,73.35%다.
0.146197790→0.12는 **0.026197790, 약17.92%** 추가 감소가 필요하다. 후반 두 포인트만으로 해결하려면 그 기여를67.23% 줄여야 하고, 첫2초만 개선한다면24.43%가 필요하다. 개선폭을 단순 더한 달성 전망이 아니라 필요한 오차 예산이다.

첫2초의 네 GT interval speed 최대–최소가0.5m/s 이하인 일반 주행이1,306행이다. 이 집단이 기존 평균 오차의60.84%, 세 모델 평균의59.53%를 만든다. 큰 가속·감속 사건만이 주원인이라는 해석은 맞지 않는다. 작은 가감속/시간 위상 오차까지 없다는 뜻도 아니다.

세 모델 평균에서 future heading 변화5도 미만 일반 주행1,633행의 기여는0.109913이다. 15도 이상57행의 기여는0.009037이다. Command의 의미 회전 분류와 이 GT 기하 분류는 다르며 서로 합쳐 쓰지 않는다.

세 모델 평균의 최악5% 행을 GT로 완벽히 고쳐도0.123418이 남는다. 극소수 실패 사례 제거만으로0.12를 설명할 수도 없다.

## 4. 진행량의 어떤 오차인가

일반 주행에서 첫 네 interval의 예측 길이 오차를0.5초로 나눈 값은 실제 현재 vx가 아니라 **예측 미래 계획의 구간 속도 오차**다.
CONT5710의 네 구간 공통 오차 크기는 평균절대0.09052m/s, 네 구간 오차의 제곱합 중 공통 성분은65.64%다. Temporal은0.09231m/s,65.12%로 더 좋아지지 않았다. 세 모델 평균도0.09275m/s,69.40%다.

이는 여러 구간을 같은 방향으로 조금 길거나 짧게 예측하는 문제가 크다는 근거다. 세 모델 평균의 전역 signed mean은+0.00378m/s에 불과해 sample별 부호가 상쇄된다. 한 개 전역 bias/scale을 빼는 해결책으로 볼 근거가 약하다. MSE 성분 비율을 공식 PREFIX 기여 비율로 읽지 않는다.

GT를 활용한 기하 성분 교체 진단:

|기존 두 모델 평균의 수정|PREFIX|
|---|---:|
|원래 출력|0.148555967|
|모델 방향 유지 + GT 구간 길이|0.055704486|
|모델 구간 길이 유지 + GT 방향|0.118660287|

진행량을 정확히 맞히면 남은 오차를 크게 줄일 표현 여지가 있다. 이것은 GT를 사용하는 구성적 진단이며 배포 가능한 모델, PREFIX 최적 oracle, 학습 난이도나 달성 예상이 아니다. 회전의 chord/방향 상호작용도 있어 두 감소량을 더할 수 없다.

## 5. 원인 판단을 바꾸는 새 frozen 진단

|고정 가중치 개입|CONT5710|TEMPORAL|
|---|---:|---:|
|원래|0.153684068|0.154119412|
|Planner의 예측 state/history 숫자만 0|0.153564181|0.154014964|
|Pooled continuous motion만0|0.194481592|0.175328544|
|새 temporal read 경로 우회|해당 없음|0.527783546|
|두 continuous motion 경로 모두 제거|0.194481592|0.551690981|
|Native motion 과거 영상만 현재 영상으로 반복|0.226024813|0.242692158|

제공 status/goal/scene history는 마지막 개입에서도 유지했다. Native history 개입 때 scene은 정확히 같고 motion 정보만 바뀐다. 제공 status의 vx를±0.1m/s 바꾸는 별도 검사에서는 motion/state/history 직접 출력이 정확히 같았다. 입력 경로 분리를 다시 확인했다.

**해석:** 모델은 영상 시간 정보를 실제로 사용한다. 새 temporal read도 장식이 아니다. 그러나 그 경로에 강하게 의존한다는 사실과 대조군 대비 진행량 정확도가 좋아졌다는 사실은 다르다. Temporal의 실제 matched 개선은 주로 횡방향(-13.74%)이고 종방향 절대오차는+0.88%다.

예측 숫자 token의 영향은 작은 반면 continuous representation은 중요하다. 따라서 vx MAE만 줄이거나 numeric token을 빼는 것을 주력으로 삼을 근거는 약하다. Auxiliary supervision 자체가 불필요하다는 실험은 아니다.

Status vx -0.1m/s는 전체 PREFIX를 각각0.0000127/0.0000061만 낮추며, 일반 주행은 오히려 악화한다. +0.1은 약0.0049/0.0052 악화한다. 이는 query 간접 영향의 민감도 검사이지 status 보정 추천이나 직접 적분 경로 인증이 아니다.

모든 OFF/입력 개입은 학습 때의 분포를 바꾸는 frozen 검사다. OFF 재학습의 결과나 각 경로의 독립적인 인과 기여량으로 해석하지 않는다.

## 6. 표현·최적화와 일반화가 함께 남아 있다

같은 무증강 train1,024행:

|모델|Train probe PREFIX|V0 PREFIX|
|---|---:|---:|
|QREFINE|0.092860878|0.164251762|
|FRESH|0.129735925|0.158707248|
|CONT5710|0.125723775|0.153684068|
|CONT6852|0.123638390|0.156334052|
|TEMPORAL|0.121652970|0.154119412|

세 모델 평균 train0.103159201/V0 0.146197790. Train/V0는 다른 scene이므로 차이를 순수 과적합량으로 분해할 수 없다. 하지만 각 일반 주행 속도 bin에서도 train보다 V0 오차가 크며 단순 정지 빈도 차이만은 아니다.

- QREFINE은 train을 더 잘 맞혀도 V0는 나쁘다. 기존 ETRI 계보의 잘 맞는 표현을 계속 보존한 것이 항상 새로운 세션에 유리하지 않다는 관측이다. 초기화 자체가 유일 원인이라는 통제 실험은 아니다.
- FRESH continuation의 마지막1,142 update는 train0.12572→0.12364 개선과 V0 0.15368→0.15633 악화를 함께 보였다. 더 오래 돌리는 것만으로0.12가 된다는 근거가 없다.
- Temporal은 train fitting도 개선했지만 train에서도0.12165가 남는다. 일반화 문제 하나로 모든 잔여 오차를 설명하지 않는다.
- 모델 간 상관: CONT5710/FRESH의 row 오차 상관0.989, Temporal/CONT5710은0.873. 같은 계보 continuation을 많이 평균하는 것보다 일부 서로 다른 실패 패턴이 유용하되, 세 모델 평균도0.1462다.

정지 상태 해석도 동일하다. CONT는 steady99행 모두, Temporal은98행을 현재 정지로 분류한다. 그런데 미래1초 변위는 각각0.107m/0.164m, GT는0.00146m다. Direct future regression에서 정지 유지와 출발의 구분/일반화가 남아 있다는 증거다. 현재 정지≠미래 정지이므로 current-stop hard gate를 만들 근거는 아니다.

**우선 원인 가설:** 영상에서 미세한 metric progress를 읽고 미래 계획으로 옮기는 연속 표현이 충분히 정밀하지 않고, 학습한 관계가 새로운 세션에 안정적으로 옮겨지지 않는다. Direct XY의 출력 구조/학습 동역학도 기여할 수 있다. 정확히 image matching, depth/scale, decoded representation 중 어느 것이 한계인지 이번 결과만으로 분리하지 못했다.

## 7. 데이터·수치 문제는 어디까지 배제했나

- 83,700+1,998행의 미래6점과 goal을 원본 좌표·자세에서 재계산하면 cache와 float32 bitwise 일치. 변환 전 반올림 최대 차이3.815e-6m. Frame+50 convention도 실제 frame join과 일치했다.
- V01,998행의 real-time state target와 nominal provided status 각각 재계산 차이0. 두 정의를 섞거나 clone으로 덮어쓰지 않는다.
- Train93session과 V0 11session의 overlap0. FULL 가중치/feature를 이번 DEV 진단에 사용하지 않았다.
- Train5scene/1,350행(1.61%)은 실제 cadence 약0.10624s,30프레임은약3.1875s. V0에는 이런 persistent slow scene이 없다. V0 미래3초 timestamp 편차 최대절대0.000768s, 모든 horizon10ms 초과행0.
- 이5scene에서는 nominal 입력/프레임 기준 미래 supervision과 real-time state 보조 supervision의 의미 차이가 커질 수 있다. 기록할 정합성 항목이지만 현재 V0 오차 대부분의 원인이라고 할 수 없다. 공식 frame-index target을 임의로 retime하지 않았다.
- PREFIX 구현의[11,11,5,5,2,2]/36, full-effective-batch normalizer와 LEN 정의를 확인했고 기존 numerical preflight를 재사용했다. 출력은 absolute XY다.
- 이전 BF16→FP32 sampling/전체 FP32 frozen 검사가 개선을 주지 않았다는 결과를 유지한다. 모든 FP32 학습 가능성을 기각한 것은 아니다.
- GT 센서 오차, 카메라 exposure 동기화, 숨은 test 분포, 영상으로 미래 행동을 식별할 수 있는 한계는 이번 재현만으로 확인 불가다. 이를 근거 없이0.15의 노이즈 바닥이라고 선언하지 않는다.

## 8. 이번에 발견한 입력 시점 정합성 쟁점

실제 train raw fixture에 배포 adapter를 호출하고 image getter를 추적했다.

|항목|현재 코드가 사용한 상대 frame|
|---|---|
|Status causal fit pose|**-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0**|
|실제로 읽는 영상 시점|**-10,-5,-2,-1,0**|
|Pose fit에는 있으나 소비 영상이 없는 시점|**-9,-8,-7,-6,-4,-3**|

현재·과거 영상을 중복 인코딩하는 native/scene 두 pass가 있어도 고유 시점은5개다. Parser가 파일을 열 수 있다는 것과 모델이 해당 영상을 소비한다는 것은 다르다.

`OPEN_ISSUE.md` 질문2 공지의 문구는 “과거 정보를 활용하는 모든 시점(프레임)에 대하여, 해당 프레임의 영상 정보가 반드시 모델의 입력으로 함께 사용”이다. 질문4도 과거 사용 시 해당 영상 사용을 반복한다. 질문7/8의 공통 feature 간접 사용 허용은 이 시점 조건을 명시적으로 면제하지 않는다.

따라서 **query-only 목적지, causal/no-future, nominal parser parity를 통과했다는 사실만으로 전체 규정 정합성을 완료했다고 말하면 안 된다.** 이것은 기존 A2 FULL에도 적용되는 현재 producer/adapter의 구체적인 점검 항목이다. 운영국의 최종 위반 판정을 대신하지 않지만, 제출 전에 해소해야 할 정합성 공백으로 기록한다.

가능한 구현 방향은 두 가지다. (1) status fit에 실제 사용하는 모든 과거 시점의 영상을 공통 perception이 유효하게 소비하도록 하거나, (2) status producer를 실제 소비 영상 시점의 pose로 제한하고 train/DEV/배포를 동일하게 다시 맞춘다. 후자는 현재 fit의 최소6점 guard와5시점 계약을 함께 재설계해야 하므로 시점만 조용히 잘라 적용할 수 없다. 두 변경 모두 기존 점수를 그대로 승계할 수 없다. 이번 진단에서 어느 것도 수정·학습·제출하지 않았다.

## 9. 내 판단과 다음 우선순위

1. **제출 후보의 시점 계약을 먼저 닫는다.** 이는 A3 gate를 다시 여는 문제가 아니다. 선택한 수정의 성능 변화는 동일 입력 정책의 DEV에서 확인해야 한다.
2. **0.15 미만 확보와0.12 공격을 구분한다.** 고정 equal-output 평균의0.14620은 보존할 실용적인 DEV 결과다. 실제 raw graph/비용과 시점 계약이 해결되어야 제출 후보가 된다. 새 FULL을 자동 시작하지 않았다.
3. **주력 공격 대상은 일반 주행의 첫2초 sample별 진행량이다.** 작은 scene head/radius/lambda 스윕이나 numeric state head 제거를 반복할 근거는 약하다. 신경망이 직접 미래 구간의 진행량과 방향을 다루도록 하는 planner 출력/표현의 대조를 한 종류 검토할 가치가 있다. Raw status를 적분하지 않고 이미지 유래 continuous scene/motion을 유지하며 최종 PREFIX로 end-to-end 학습하는 범위다. 이는 scalar 후처리 residual이나 P×V 후보 선택의 반복과 구분되어야 한다. 기존 LEN은 loss 추가였고, 이번 후보는 출력 구조 변경이라는 차이가 있다. 다만 새 정보가 생기지 않으므로 개선을 보장하지 않는다.
4. **일반화를 실험의 중심에 둔다.** 동일 train probe와 session 단위 V0, ordinary-drive signed/common progress, stop/turn 손익을 함께 판단한다. Train fitting만 좋아지는 변경을0.12 경로로 승격하지 않는다. 이미 반복 사용한 V0이므로 작은 차이의 후보 순위와 실제 숨은 test 성능을 분리한다.

QREFINE/CONT/Temporal 중 GT로 매 row 최선 하나를 고르면0.117843이지만, 이것은 선택기의 가능한 정보/학습 성공을 확인하지 않은 oracle이다. “selector만 잘 만들면0.12”라는 과거의 과도한 해석을 반복하지 않는다.

**0.12가 불가능하다는 증거는 없다. 도달할 실행이 이미 입증됐다는 증거도 없다.** 이번에 좁힌 것은 오차의 위치·크기, 시간 정보의 실제 사용, 숫자 state token의 작은 영향, 일반화/학습의 분리, 그리고 해결해야 할 입력 시점 계약이다.

## 재현 파일

- `PROTOCOL.json`, `launch.json`: 실행 전 고정 범위와 GPU/PID.
- `probe_*.json`: 저장 모델 SHA, parameter/buffer 불변, same-forward parity, frozen 개입과 paired CI.
- `saved_prediction_analysis.json`: 5단일+고정 평균, 그룹/시점/진행량/성분 교체/scene별 결과.
- `train_and_intervention_analysis.json`: 동일1,024 train cohort, speed bin, 정지 및 query 민감도.
- `fixed_average_comparison.json`: 고정 세 모델 평균의 paired 차이.
- `target_and_time_audit.json`, `timing_distribution.json`: 전수 정답·원본 hash·시점 정합성.
- `input_frame_coverage.json`: 실제 raw adapter image 호출과 status fit 시점.
- `artifact_index.json`: 대형 NPZ는 서버 보존, path/size/SHA만 Git에 기록.
- 소스: `experiments/a2_comprehensive_audit_20260920/`.

진단 스크립트 최초 실행의 import 경로 누락과 threaded lazy NPZ 읽기를 수정했다. 후자는 배열을 메모리에 먼저 읽어 해결했고 원본 cache SHA/재계산 검사는 통과했다. 이 개발 오류를 데이터 손상이나 모델 실패로 기록하지 않는다.
