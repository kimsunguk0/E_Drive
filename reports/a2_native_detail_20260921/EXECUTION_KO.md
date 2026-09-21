# Native1152 motion / scene: 두 개의 독립 대조

사용자 지시: “둘 다 ㄱ”. GPU 0, 1, 2, 3에서 DEV 실험 두 쌍을 실행한다.
최종 제출·FULL 학습은 이 실행에 포함하지 않는다.

## 질문과 현재 근거

동결한 부모는 `P-SPLITREAD-s1/ckpt_step6852.pth`다.
DEV V0 PREFIX **0.1473937337**, checkpoint SHA-256
`8252ae12063b0e684899ae22cfac2f3a29fedc0ea06df41849bab3d47c2767f6`.
공식 제출 FULL의 **0.1336848279**와 평가 범위가 다르며 환산하지 않는다.

직전 분석에서 일반 주행의 점수 기여는 약 96%다. 안정적인 진행량의 일반 주행도
상당한 오차를 남겼으므로, 모든 오류를 가감속 타이밍으로 설명하지 않는다.
이번에는 기존 영상에서 이미 줄어든 공간 정보를 더 많은 attention으로 읽는 대신,
원본에서 만든 1152×648 영상의 세부 정보가 유효한지를 시험한다.
새 정보의 유효성 및 예상 개선 폭은 미측정이다.

## GPU와 동일 조건

|GPU|Arm|추가 관측|대조|
|---|---|---|---|
|0|M-LOW|Front 현재+H4, 1152→768→1152|동일 motion 추가 분기, 낮은 영상 세부 정보|
|1|M-NATIVE|Front 현재+H4, 원본 유래 1152|M-NATIVE − M-LOW|
|2|S-LOW|현재 6-camera, 1152→768→1152|동일 scene 추가 분기, 낮은 영상 세부 정보|
|3|S-NATIVE|현재 6-camera, 원본 유래 1152|S-NATIVE − S-LOW|

H4는 −0.1/−0.2/−0.5/−1.0초다. 카메라·시간 수를 늘리는 실험은 아니다.
LOW/NATIVE는 동일한 1152 JPEG에서 출발한다. LOW만 down/up을 적용한다.
각 쌍의 graph·연산 shape·새 파라미터 초기값·부모·row 순서·기존 영상·증강은 같다.
LOW와 옛 768 cache의 pixel이 같다고 주장하지 않는다.

모두 train310 **83,700행**, V0 **1,998행/37scene/11session**을 사용한다.
FULL 계보의 weight, feature, teacher, 적응 통계를 반입하지 않는다.

## 실제 구조

기존 768 current6 / 384 front scene history / 768 front motion 경로를 보존한다.
추가 관측도 같은 image backbone/FPN을 사용하며 별도 backbone 가중치는 없다.

M: 추가 1152 front pair의 FPN 두 level에서 radius6 correlation과 시각 feature를
fuse한다. 그 map을 기존 pair map 크기로 맞춘 뒤 **zero-initialized 1×1 projection**을
거쳐 기존 `correlation_fuse` 출력에 더한다. 기존 pooling, state/history,
continuous motion, SplitRead memory와 길이·방향 출력은 그대로 연결된다.

S: 추가 1152 현재6-camera와 기존 low-resolution history를 기존 scene encoder로
읽는다. 얻은 공통 raster를 zero-initialized projection으로 기존 raster에 더한다.
**동일 최종 raster를 occupancy, lane, planner 모두 사용한다.**
기존 scene 계산과 추가 scene 계산의 비용을 모두 측정한다.

기존 768 calibration의 normalized 좌표는 같은 crop/FOV의 1152 feature를 가리킨다.
등가 pixel-center 변환은 `u_hi=1.5*(u_lo+0.5)-0.5`이며, 좌우 반전도 같은 정책이다.

새 분기 기여가 0일 때 원래 함수가 보존된다. 과거 SHARED768의 큰 step-0 변화를
그대로 받아들이는 방식과 다르다. 이후 shared backbone의 joint 학습은 허용되므로
학습 내내 base 출력이 고정되는 것은 아니다.

## A2 입력 경계

제공 nominal causal status는 기존 shared scene query에만 들어간다.
기존 goal의 scene 조건과 pose alignment는 유지한다.
Raw status/pose/goal을 motion, predicted state/history, planner, XY 보정식의
새 입력으로 연결하지 않는다. S 결과를 motion/state 경로로 되먹이지 않는다.
전체 모델을 goal-independent라고 설명하지 않으며, 이 검사가 운영국의 개별 승인이라는
주장도 하지 않는다.

## 고정 학습 계획

|설정|값|
|---|---|
|Stage update|10,277; 약 1.965회 train row 노출|
|Effective / micro batch|16 / 8, 네 arm 동일|
|기존 backbone LR|1e-6|
|기존 head LR|1e-5|
|신규 분기 LR|5e-5|
|Optimizer|Fresh AdamW, 기존 weight decay/clip 유지|
|Schedule|100 warmup + 새 stage 전체 cosine|
|정밀도·BN|기존 BF16 image / FP32 planner·loss, fixed BN|
|Loss|기존 PREFIX + LEN 0.25 + 기존 인지/motion 묶음|
|평가|0 / 3,426 / 6,852 / 10,277|
|Primary|동일 10,277 terminal끼리 비교, 부모 대비도 함께 보고|

5-update 검사는 실행·gradient·export 검증이며 성능 screening이 아니다.
본 학습은 smoke weight/optimizer를 쓰지 않고 동결 DEV 부모에서 다시 시작한다.
추가 학습의 이득과 영상 detail의 이득을 구분한다.
결과를 보고 자동 λ/head/해상도 sweep, 업데이트 연장, FULL을 시작하지 않는다.

## 검증과 판정

초기 실제 checkpoint FP32/BF16 비교에서 plan, scene, motion, pair memory,
state/history, occupancy/lane 출력 차이가 모두 0이었다.
4개 arm의 V0 step-0는 모두 **0.14739373617304918**, 기존 parent와 차이 2.5e-9다.
추가 projection을 0.01I로 활성화한 검사에서도 provided status/goal만 바꿀 때
motion/state/history의 직접 출력은 불변이었다.

모든 영상과 status/target의 double-flip은 bitwise 동일하다.
기존 float32 lidar2img 반사 계산만 최대 3.052e-5 round-trip 반올림이 발생하여
그 필드에만 6.2e-5 absolute 허용오차를 적용했다. 처음 실패한 검사 로그도 보존한다.
캐시 launcher의 completed/complete 문자열 불일치도 수정하고 실패 로그를 보존했다.
이 두 초기 검사 실패 중에는 본 학습이 실행되지 않았다.

본 학습 전에 5 update, 신규 분기 실제 update, fresh-process strict reload/export,
실제 row·기존 증강 동일성과 detail pixel 차이, RTX4090 전체 graph 비용을 확인한다.
비용은 파일 읽기/전처리/전송 제외 model forward이며 서버 elapsed_ms와 다르다.
모델 구조가 추가 분기를 포함하므로 과거 replacement graph의 41/38ms를 승계하지 않는다.

`collect_native.py`가 기존 metric producer로 시점, nonstop/depart/steady,
종·횡, 구간 길이·벡터, 인지 지표, 같은 11 session의 paired CI를 보존한다.
CI는 재사용 DEV 조건부 결과이며 hidden-test 보장이 아니다.
개선 주장은 실제 PREFIX와 같은 계열 LOW·부모 대비를 함께 보고 판단한다.

## 산출물 위치

- 코드: `experiments/a2_native_detail_20260921/`
- 설명/검증/요약: `reports/a2_native_detail_20260921/`
- 가중치·row 예측·학습 로그: `work_dirs/a2_native_detail_20260921/`
- 캐시: `data/etri/motiondrive_v2/native_detail_1152_20260921/`

대형 cache·weights·예측은 Git에 넣지 않는다. Git에는 protocol, source/weight/row
식별자, 검사 및 집계 결과를 기록한다. 실행 상태는 `runtime/main_orchestrator.json`,
결과는 `results.json`, `RESULTS_KO.md`가 기준이다.

## RTX4090 실측 완료

5-update strict student, 실제 DEV 입력 두 개에서 전체 B1 graph를 측정했다.
Motion median **49.36/49.42ms**, p95최대49.51ms, **1,444.924G FLOPs**.
Scene median **51.75/51.77ms**, p95최대51.91ms, **1,516.674G FLOPs**.
추가 분기를 포함한 비용이며 파일 읽기·전처리·전송은 제외했다.
최종 후보의 raw 제출 adapter와 동일 가중치 배포 검증은 DEV 결과 뒤 별도 작업이다.

## 본 학습 착수

2026-09-21 22:03:00 KST에 네 arm을 모두 시작했다.
22:05:57 KST snapshot에서149~150update, nonfinite0, 동일149update sample stream을 확인했다.
실측1.01~1.03초/update 기준 첫 결과9/21 23:05전후, terminal9/22 01:05~01:15예상이다.
실제 PID·개별 ETA·optimizer group은 `launch_health.json`에 기록했다.
