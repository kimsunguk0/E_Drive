# 공개 영상 운동 모델의 비용 검증 — 연구 후보, 미채택

2026-09-07. B200 P2 학습 대기 중 별도 RTX3090에서 측정했다.
P2 모델·데이터·학습 인자와 기존 배포 모델은 변경하지 않았다.

## 실제 비용

Torchvision0.22.1+cu128의 `raft_small`, 공개 `C_T_V2` 가중치,
FP32/TF32 off, RGB384×216, 조건별 warmup20·반복50회다.
현재→과거−1frame 한 쌍 또는 현재→과거−1/−2/−5/−10frame 네 쌍을 batch로 처리했다.

| 쌍 수 | recurrent 갱신 | CUDA median ms | p90 | p95 |
|---|---:|---:|---:|---:|
| 1 | 4 | 8.5601 | 8.62 | 9.0808 |
| 1 | 8 | 13.5400 | 13.60 | 13.7184 |
| 1 | 12 | 18.5206 | 18.75 | 19.3251 |
| 4 | 4 | 9.5485 | 9.56 | 9.6602 |
| 4 | 8 | 14.8934 | 14.92 | 15.0133 |
| 4 | 12 | 20.2152 | 20.28 | 20.5940 |

4pair peak PyTorch allocated/reserved는149.02/250MiB다. 각 전체 forward의 영상
feature encoder, context encoder, all-pairs correlation pyramid 생성, recurrent
갱신과 모든 반복별 upsampling을 포함했다. feature cache/warm-start flow는 없다.
전처리·H2D·출력 검사·가중치 로드는 CUDA event 경계 밖이며 매회 synchronize했다.

이는 **부품 비용만**이다. 이전 V2 측정값에 단순히 더해 최종 Tinfer로 사용하지 않는다.
성능 비교/학습/metric velocity 평가/최종 채택/공식 제출은 하지 않았다.
실제 종료0, GPU PID72040만 관측됐고 종료 후130MiB/compute PID없음으로 복귀했다.
0.5초 간격 검사보다 짧은 외부 작업의 미관측 가능성은 남는다.

## 입력·가중치·실행 계보

이미 검증된 immutable train8 export의 첫 클립만 사용했다. forward에는 현재 전방과
과거 전방 영상만 전달했다. 원 export의 GT·status·goal·pose는 모델에 전달하지 않았다.
현재768 영상은 V2와 같은 `bilinear/align_corners=False/antialias=True`로384에 맞췄다.
ImageNet 정규화를 역변환한 뒤 RAFT의 RGB[-1,1]로 변환했다. 추가 crop/pad/clamp는 없다.

- 측정기: `scripts/benchmark_motiondrive_v2_flow_candidate.py`
  SHA `8937da911dc685f11523fb05a604ae5c6c22a4ac97a99656a41a9907bc88457d`
- 결과: `reports/motiondrive_v2_raft_small_3090_fp32.json`
  SHA `ddcb0e940c7ef9616dd5d26826902acb0ac39ca592179a4faaacb82359a54d0f`
- 실제 argv/rc/runtime 소스 증거: `reports/motiondrive_v2_raft_small_3090_fp32.execution.json`
  SHA `9fcbcd225f8eceae23083687d62be0ac6a22b215b51bf8bb76f8341122a29760`
- weight SHA `01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27`
- reference SHA `4fb192ff6dd80ad44c8374b64afe55d49112327174ee8f7534ef4591cd2de1f3`
- 새3090 stage: `/home/intern/adcl_motiondrive_v2/measurements/raft_small_fp32.UIZiGPNu`

공식 weight URL에서만 다운로드하고 prefix/full SHA·strict load를 검사했다.
기존 runtime은 읽기 전용, package 설치/기존 stage 변경/B200 변경은 없었다.
별도 CPU6tests와 루트 독립 재실행이 통과했다. 실제 보고서의6조건×50 event,
전후 소스/reference/weight SHA, GPU PID 기록도 루트가 교차 확인했다.

## 다음 의사결정

이 비용 결과는 공개 사전학습 영상 대응을 **검토 가능한 후보**로 남긴다.
전체 neural integration의 추가 비용과 L2 개선은 따로 검증해야 한다.
우선 P2 C1/T1의 상태 추정과 궤적 오차를 함께 관찰한다.

이를 위해 planning evaluator에 opt-in `--include-motion-predictions`를 준비했다.
기본 off일 때 기존 records·지표·forward/RNG는 유지한다. on이면 **같은 full forward**의
FP32 state/history를 기록한다. stop은 확률이 아닌 logit이며 sin/cos도 보정 없이 보존한다.
이 기록은 GT로 neural state를 교체하거나 새로운 경로를 만들지 않는다.
평가기와 P2 분석기 관련 CPU101tests가 함께 통과했다. 실제 P2 출력은 아직 없다.

판단을 구분한다:

- 실제 neural motion/state 추정이 부정확하면 영상 대응 표현 강화가 다음 실험 후보다.
- 추정은 정확한데 planner의 횡방향 표현이 약하면 정보 사용/시간별 출력 학습을 먼저 점검한다.
- 둘 중 어느 원인도 P1의 GT 상태↔GT 미래 상관만으로 확정하지 않는다.

기존 P1 normal1998의 사후 상관에서 현재GT yaw_rate와 미래y의2차 계수는
Pearson.842, vx×yaw_rate는.863이었으나, 정지123개에서는 관계가 약했다.
이는 미래 경로를 현재 상태로 생성하라는 처방이나 회수 가능한 개선량이 아니다.
새 근거 없이 P2 조건·체크포인트 선택·미사용 final-val을 바꾸지 않는다.

## 공개 근거와 한계

[Torchvision0.22 모델 문서](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.optical_flow.raft_small.html),
[실제 연산 구조의 공식 구현](https://docs.pytorch.org/vision/0.22/_modules/torchvision/models/optical_flow/raft.html),
[공식 입출력 예제](https://docs.pytorch.org/vision/0.22/auto_examples/others/plot_optical_flow.html),
[RAFT 원논문](https://arxiv.org/abs/2003.12039)을 확인했다.
RAFT의 flow는 픽셀 변위이지 metric 속도가 아니며 깊이·회전·동적 객체의 모호성이 남는다.

코드의 BSD 라이선스와 공개 사전학습 가중치 사용 조건을 동일시하지 않는다.
[Torchvision 사전학습 모델 고지](https://github.com/pytorch/vision/blob/v0.22.1/README.md#pre-trained-model-license)는
학습 데이터 유래 별도 조건의 가능성을 명시한다. 사용 조건 검토는 미완결이며,
현재 결과는 연구용 비용 검증이다. 대회 규정의 최종 승인을 뜻하지 않는다.
