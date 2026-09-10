# 0.13075 주장에 대한 반증 중심 CPU 감사

2026-09-10. 결론은 **“고정된 모델이 원 tune 1,998행에서 정답을 이용한 후보 선택 없이 D3 0.13074861을 얻었다”는 주장은 지지된다. “미노출 데이터에서 0.15 이하의 일반화가 확정됐다”는 주장은 아직 지지되지 않는다.** 이번 감사에서는 GPU·새 모델 실행·held 역할의 예측/정답 분석을 하지 않았다. primary train203의 입력 특징 감사에는 당연히 이후 held32로 지정된 원 학습 시나리오도 포함되지만, 그 집합의 정답·성능을 조회하거나 따로 분석하지 않았다.

**정답 유입 가설을 점검한 결과**

- 실제 `cache_selection_features.py:154`의 특징 입력은 후보 좌표·원 점수·유효성, causal 상태, 제공 목표점뿐이다. `gt_plan`으로 만든 D3는 159행에서 별도 `costs.npy`에 기록된다. `relative_selector.py:46`의 32D 함수에는 정답 인자가 없다.
- 선언만 확인하지 않고, 저장된 **train 10,962행 + tune 1,998행 전체, 각 200후보**의 특징을 입력 파일만으로 CPU에서 다시 만들었다. 이 작업은 `gt_xy.npy`와 `costs.npy`를 열지 않았다. 저장 GPU 특징과의 최대 차이는 **2.3842×10⁻⁷**이며, 후보 속도·가속도 22차원은 차이가 0이다. 소비한 입력 파일 SHA도 캐시 manifest와 일치했다. 따라서 정답이 별도로 직렬화된 특징에 섞였다는 가설을 강하게 반박한다.
- `train_cached_selector.py:220`은 train 특징만 head에 넣고, 다음 행에서 train costs를 손실에 넣는다. 평가의 `argmax(scores)`는 137행, 선택한 cost 조회는 그 뒤다. GT로 최적 후보를 고르는 oracle은 성능 진단값이며 실제 선택값과 분리돼 있다. 학습 행과 tune 행은 겹치지 않고, 캐시 생성 시 session 경계도 검사한다.
- 측정 당시 `evaluate_relative_selector.py:50`은 입력을 이미지·투영행렬·이미지 크기·상태·목표점의 5개 키로 제한한다. 모델 호출 이후에 GT를 오차 계산에 사용한다. 전체 이미지 추론 결과의 `rows/pred/candidate_id/d3/shortlist_oracle/point_l2/error_xy`가 캐시 terminal 결과와 **1,998행 전부 bitwise 동일함을 이번 CPU 감사에서 다시 확인했다.** 학습용 특징/정답 NPY를 배포 head 로더가 읽지 않는 것도 `relative_artifact.py:34`에서 확인했다.

이는 모델 정답 유입을 찾지 못했다는 구체적인 계산 경로 감사이며, 모든 과거 프로젝트 작업에 대한 무제한 무누수 증명은 아니다.

**causal 상태와 제공 목표점은 다른 정보다**

상태는 과거 frame −10..0의 11개 pose를 nominal 10Hz로 이차 적합한 vx,vy,ax,ay이다. overlay 생성기는 이 프레임만 계산 함수에 넘기며, 현재 loader는 status8의 4:8에 넣는다. 실제 `motiondrive_v2_shared_status_data.py:27`과 overlay builder의 `required_frames` 구성을 읽었다. 원 pose 파일 전체를 identity join 용도로 읽는 것과 미래 값을 상태 계산에 사용하는 것은 다르다.

목표점은 **제공되는 frame +50 위치**를 현재 ego 좌표로 바꾼 값이다. `deployment.py`는 현재 pose와 +50의 XYZ만 사용하고, 예측 정답인 +5,+10,…,+30은 pose 입력 whitelist에서 허용하지 않는다. 이 목표점은 과거 관측으로부터 추정한 causal 정보가 아니다. 따라서 “모든 입력이 causal” 또는 “영상만으로 0.13을 달성”이라고 쓰면 틀린다. 정확한 표현은 **현재 3-camera 영상 + 과거 pose 기반 상태 + 제공 목표점으로 고정 후보를 선택**한다는 것이다. 제공 상태/목표점의 완성 후보 선택 사용이 허용된다는 대회 규정 해석과, GT6의 계산상 누수 여부는 구분해야 한다.

원 base의 goal mode는 `none`이지만 새 상대 head에는 목표점 특징이 추가된다. 이번 개선은 상대 운동 특징·추가 MLP·목표점 입력을 함께 바꾼 결과다. 운동 특징 하나의 인과 효과나 특정 손실 함수의 우수성을 따로 증명하지 않는다.

**수치가 지지하는 범위와 남은 불확실성**

| 확인된 사실 | 넘어서는 주장 |
|---|---|
| 고정 terminal D3 0.193903→0.130749, fine regret 0.103409→0.040255 | 선택 regret의 원인이 하나로 규명됐다는 주장 |
| 동일 후보/좌표에서 fine regret 61.07% 감소, 11/11 tune session 개선 | 새로운 도로·날씨·센서 분포에 대한 보장 |
| 11-session paired bootstrap ΔD3 CI [−0.070657, −0.059074] | 방법 개발의 적응적 선택까지 보정한 확증 검정 |
| 개별 tune mean CI [0.111638, 0.153115] | 95% 신뢰도로 항상 ≤0.15라는 주장 |
| 3초 평균 L2 0.785202→0.442827 | 후반 오차가 모든 집단·극단 사례에서 해결됐다는 주장 |

seed는 0 하나다. tune은 프로젝트 전체에서 반복적으로 방법·은행·설정 선택에 사용됐다. 개별 행을 독립으로 bootstrap하지 않은 것은 적절하지만, 11개 세션의 재표집만으로 이 적응성이나 공간 중복을 제거하지는 못한다. CE와 centered D3의 차이 CI [−0.001887,+0.002639]도 우열을 확정하지 않는다. CE의 3초 p99는 2.5764m로 MotionDrive 2.4554m보다 높고, GT 감속 집단 및 일부 속력 구간에도 후반 약점이 남는다.

추가로 확인한 실제 배치1 추론은 D3 **0.13052215**였다. 평균은 유사하지만 배치8 대비 후보 ID 227행, 좌표 213행이 달라졌다. 따라서 배치·BF16 수치 조건에 무관한 bitwise 동일 결과라는 주장은 안 된다. 배치8의 0.13074861과 배치1의 0.13052215를 구별해 기록하고 최종 확인/배포 조건을 고정해야 한다.

**12-session 확인의 분리 수준**

240개 실제 timestamp parquet을 다시 읽고 기존 감사의 SHA와 대조했다. held32의 12세션은 train171의 60세션 및 tune37의 11세션과 scene/session이 겹치지 않는다. 과거 여유까지 포함한 전체 raw frame −50..349 구간에서도 겹침은 없다.

| 경계 | 원 입력/정답 support frame0..349 최소 간격 | 전체 raw −50..349 최소 간격 |
|---|---:|---:|
| train171 ↔ held32 | 65.099878초 | **60.099855초** |
| tune37 ↔ held32 | 65.099974초 | **60.100049초** |
| train171 ↔ tune37 | 65.099995초 | 60.100064초 |

이것은 시간·session 분리의 증거이며 도로/공간 분리의 증거는 아니다. 더 중요한 한계는 **held32가 원래 primary train203에 속했다는 사실**이다. 과거 프로젝트와 primary 모델·은행은 이 집합의 정답을 이미 사용했다. 새 train171 run의 가중치·은행에서 제외하더라도 과거 방법 개발의 정보가 사라지는 것은 아니다. 과거 일부 holdout 수치에 대한 논의도 있었으므로 “역사적으로 한 번도 보지 않은 독립 lockbox”라는 표현은 부적절하다.

권장 명칭은 **“그룹을 제외하고 다시 학습한 모델의 12-session 추가 확인”**이다. 새 모델과 은행이 해당 그룹을 직접 학습하지 않고도 성능을 유지하는지 점검하는 유용한 증거가 된다. 반면 이것만으로 전체 연구 과정과 독립인 미노출 일반화 성능을 확정할 수는 없다. 현재 0.13075의 정직한 tune 주장에는 충분한 근거가 있지만, 새로운 챌린지 평가 분포에서의 목표 달성까지 확정하는 근거는 아직 아니다.

**최종 확인 전에 지켜야 할 실질적 조건**

현재 primary base·은행·상대 head를 held32에 그대로 적용해 일반화 확인이라고 부르면 누수가 된다. 확인용 bank는 fresh train171/46,170행으로 만들어졌고 `confirmation2000_none_s0_v1`도 동일 집합의 공개 초기화 학습 manifest를 가지고 있음을 확인했다. 여기에 붙이는 head 역시 해당 train171 모델의 9,234개 train 캐시 행만으로 새로 적합해야 한다. 이번 감사는 그 최종 composite 후보의 학습 완료나 held 평가를 대신 승인한 것이 아니다.

held 정답을 열기 전에 base·은행·head·소스 SHA, objective, terminal step, 배치/precision과 평가 행 SHA를 하나의 후보 기록으로 고정하고 각 조상의 fit row가 train171에만 속하는지 거부 테스트까지 통과해야 한다. 이 경계가 충족된다면 분할 때문에 추가 확인을 막을 구체적인 장애는 찾지 못했다. 확인 결과를 보고 다시 설정을 고르면 그 결과는 추가 개발용 결과로 재분류해야 한다. 새로운 학습 실험의 확장은 제안하지 않는다.

**측정 소스 보존 확인**

감사 중 현재 evaluator와 artifact loader가 confirmation 지원 때문에 수정돼 측정 receipt SHA와 달라지는 것을 감지했다. 당시 소스는 `reports/sparsedrivev2_20260910/public_init/relative_v1_measured_source/`에 보존돼 있었고 세 파일 모두 측정 receipt와 일치하는 SHA를 재확인했다. 따라서 나중 코드를 과거 실행 코드로 제시하는 문제는 해소됐다. core `relative_selector.py`는 계속 SHA `7fa75415a40691f555cca27ab9fe88331725177f19887c01a0e887121e2ac59f`다.

근거는 같은 보고서 디렉터리의 `independent_013075_claim_audit.json`, `original_live_source_audit.json`, `relative2000_terminal_analysis.json` 및 `public_init/relative_soft_ce_live_tune1998/result.json`, `public_init/relative_soft_ce_live_tune1998_b1/result.json`에 있다. 감사 스크립트는 `audit_claim_cpu.py`이며 frozen 학습/모델 소스를 변경하지 않았다.
