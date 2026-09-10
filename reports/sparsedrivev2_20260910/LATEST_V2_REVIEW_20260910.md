# VADv2 / SparseDriveV2 추가 검토 — 2026-09-10

앞선 분석은 두 V2 연구의 실제 공개 범위와 최신 논문을 충분히 반영하지 못했다. 이번에는 공식 논문, 구현, 가중치 링크를 확인했다. 원격 학습·제출은 실행하지 않았다.

## VADv2

- 최신 arXiv 본문은 2026-04-17 v2다. HF paper markdown은 확인 시점에 2024 v1 본문이어서 최신 arXiv HTML로 보완했다.
- 장면을 map/agent/traffic/image token으로 표현하고, 기본 4,096개 궤적 vocabulary에 대한 확률을 학습한다.
- 최신 논문 Table 12에서 deterministic / probabilistic의 Town05 Long 3초 L2는 0.223 / 0.225m, Driving Score는 74.6 / 85.1이다. 확률적 계획의 closed-loop 이득이 이 대회 open-loop L2의 이득이라는 근거는 아니다.
- 공식 VADv2 폴더에는 config와 head 두 파일만 있다. 코드가 별도 모델·데이터셋·PIPP 유틸·vocabulary 파일을 참조하므로 그 폴더만으로 완결된 학습/추론 재현이 되지 않는다. 공식 README에서 확인된 checkpoint는 VAD v1 Tiny/Base이며, VADv2 공식 checkpoint 다운로드는 찾지 못했다.

출처: [최신 논문](https://arxiv.org/html/2402.13243v2), [공식 VADv2 코드](https://github.com/hustvl/VAD/tree/main/VADv2), [공식 README](https://github.com/hustvl/VAD). 공개 파일·API 응답은 `materials/vadv2/`에 보존했다.

## SparseDriveV2

- 2026-03-31 논문 공개. 경로 모양과 시간별 속도를 분리한 vocabulary 및 단계적 scoring이 핵심이다.
- NAVSIM 설정은 1,024개 path × 256개 velocity = 262,144개 조합이다. 단계적으로 줄여 NAVSIMv1에서는 400개, v2에서는 200개를 최종 scoring한다. 경로별 영상 샘플링 및 최종 궤적별 재평가를 쓴다.
- 논문의 NAVSIM PDMS/EPDMS 및 Bench2Drive Driving Score는 ETRI D3와 다른 지표다.
- 공식 NAVSIMv1/v2 checkpoint와 Bench2Drive stage2 checkpoint는 실제 HTTP 200 및 바이너리 크기를 확인했다. 전체 파일 다운로드·모델 재현을 완료한 것은 아니다.

출처: [논문](https://arxiv.org/html/2603.29163v1), [공식 저장소](https://github.com/swc-17/SparseDriveV2), [공식 가중치](https://huggingface.co/wenchaosun/SparseDriveV2/tree/main), [HTTP 확인 기록](sparsedrivev2_review/ckpt_http_head.json).

코드상 주의점:

- NAVSIM main은 현재 시점의 3개 카메라를 사용한다. 과거 영상으로 속도를 추정하는 temporal encoder가 공개 checkpoint에 이미 학습되어 있다고 가정하면 안 된다.
- NAVSIM은 command/velocity/acceleration의 8차원 status를 임베딩하여 path/velocity query에 더한다.
- velocity anchor는 구간별 절대 속도이며 제공 현재 속도로 좌표를 스케일하는 방식이 아니다. 최종 출력은 고정된 `traj_vocab`의 행 선택이다. re-conditioning은 특징을 다시 읽는 과정이며 궤적 좌표를 회귀·수정하는 연산이 아니다.
- Bench2Drive는 별도 브랜치다. 6카메라 R50, temporal instance queue, perception/motion 학습 구성이므로 NAVSIM의 R34 현재 영상 모델과 구분해야 한다.

출처: [입력 처리](https://github.com/swc-17/SparseDriveV2/blob/main/navsim/agents/sparsedrive/sparsedrive_features.py), [모델](https://github.com/swc-17/SparseDriveV2/blob/main/navsim/agents/sparsedrive/sparsedrive_model.py), [scoring 및 최종 행 선택](https://github.com/swc-17/SparseDriveV2/blob/main/navsim/agents/sparsedrive/custom_decoder.py), [Bench2Drive 브랜치](https://github.com/swc-17/SparseDriveV2/tree/bench2drive).

## 대회에 대한 판단과 제안

**이번 추가 검토로 SparseDriveV2의 설계 검증 우선순위를 높인다.** 현재 모델의 종방향 오차와 맞닿는 아이디어는 경로 형상 p(s)와 진행량 s(t)를 구분하여 같은 경로에 서로 다른 가감속·정지/출발 시점을 조합하는 것이다. 그러나 이런 분해만으로 영상에서 필요한 상태와 미래 상호작용을 더 잘 읽게 된다는 보장은 없다.

기존 M13 oracle 0.1551은 그 고정 shortlist의 한계다. 더 조밀한 factorized vocabulary의 한계까지 정하지 않는다. 기존 PV head 실패도 후보 구성, 학습, 영상과의 상호작용이 다른 SparseDriveV2 전체를 반증하지 않는다. 반대로 큰 vocabulary 자체를 성공으로 간주해서도 안 된다.

`OPEN_ISSUE.md` Q7–Q10을 함께 읽으면, 제공 상태/goal의 직접 궤적 생성 입력은 제한되지만 완성된 후보 중 선택에는 예외가 있다. 따라서 원본이 status를 사용한다는 이유만으로 scoring 계열을 일괄 불허라고 할 수 없다. 고정 후보를 선택하는 실제 코드 경로는 이 예외와 관련된 근거다. 다만 대회용 구현의 후보 출력 정의·선택 경계·영상 기여는 별도로 확인해야 한다. 이후 status/goal로 좌표를 수정하는 refinement를 추가하면 동일한 선택 전용 근거를 그대로 적용할 수 없다. [운영 답변](snapshot/OPEN_ISSUE.md).

실행 제안은 다음 두 비교를 작게 시작하는 것이다.

1. **구조 검증:** ETRI train fold만으로 3초 path/velocity bank를 구성한다. 같은 held-out 표본에서 전체 bank oracle → 학습 coarse 단계 후 shortlist oracle → 실제 D3를 각각 측정한다. 예를 들어 0.10 → 0.12 → 0.15는 개발 목표 예시이며 예상 성능이 아니다. bank가 충분해도 pruning이나 마지막 선택이 실패하는지 구분한다.
2. **공개 가중치 활용 검증:** 공개 SparseDriveV2 초기화를 대회 카메라·좌표·6개 시점·허용 입력에 맞춰 적응시킨 모델을 직접 예측 기준과 비교한다. 기존 표현에 factorized head를 붙인 비교와 함께 보면 공개 표현의 이득과 head 설계의 이득을 분리해 조사할 수 있다. NAVSIM PDMS용 최종 scoring을 그대로 두지 않고 대회 D3에 맞는 지도·선택 기준을 학습한다.

모든 bank, teacher, pretraining, fine-tuning 계보는 검증 세션을 제외해야 한다. B200에서 정확도 실험을 하더라도 초기에 전체 추론 비용을 측정하여 최종 지연 제약에서 실현 가능한 후보인지 확인한다.

‘공개 checkpoint로 고점을 만든다’의 정확한 의미는 **재사용 가능한 학습 표현을 출발점으로, 이 대회에서 측정한 강한 기준 모델을 확보한다**는 것이다. 공개 점수나 checkpoint 존재 자체가 0.15 달성 또는 성능 상한을 증명하지 않는다. 이번 근거상 실행 가능한 공개 가중치 출발점은 SparseDriveV2이며, VADv2는 우선 설계 참고 대상으로 두는 편이 합리적이다.
