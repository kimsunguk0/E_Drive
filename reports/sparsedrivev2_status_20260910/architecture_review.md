**SDV2 status 경로 및 다음 대조 학습의 독립 검토 — 2026-09-10**

검토 범위는 `/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910`의 요청 시점 commit `5c8ba90`이다. 읽는 동안 HEAD가 `63c220a3143c1001c210552eebc255ac595c86f2`로 이동했으나, 두 commit의 차이는 root가 추가한 status sensitivity 진단 파일 3개뿐이고 이 문서의 모델·규정·P7/A2 근거 파일은 동일함을 `git diff --exit-code`로 확인했다. GPU 실행, 학습 코드 수정, 외부 메시지/제출은 하지 않았다. 아래 경로와 줄 번호는 모두 이 원격 worktree 기준이다. 초안은 진단 결과가 없는 상태에서 작성했으며, 마지막에 root가 전달한 frozen sensitivity 수치와 3-arm 검토를 구분해 추가했다. 새 학습의 효과를 측정한 결과는 아니다.

현재 raw status는 **앞단 후보 축소와 마지막 재선택 두 곳에 별도로** 들어간다. 앞단은 단순한 공통 perception 입력이라고 부르기 어렵다. 상태 임베딩이 path/velocity planning query에 직접 더해져 이미지 샘플 위치, attention 잔차, top-k 후보 집합과 최종 기본 점수를 바꾼다. 뒤의 relative head는 이미 나온 고정 200개 후보의 좌표를 변경하지 않고 선택만 바꾼다. 이 차이를 먼저 진단해야 한다. 과거 결과는 이미지 상태 추정치를 정확한 원시 status 대신 꽂으면 성능이 유지된다는 전제를 지지하지 않는다.

다음 대조의 우선 후보는 **원시 status를 coarse/fine 모두에서 제거하고, 이미지 시간 특징을 후보 점수 손실로 직접 학습하는 구조**다. 다만 P7도 이미 continuous scene/motion 특징을 사용했으므로, “temporal 특징을 추가한다” 자체는 새로운 근거가 아니다. 실질적 차이는 **공개 NAVSIM의 후보 점수 디코더 재사용, dense fixed bank, top-k 이전 velocity supervision, D3에 맞춘 coarse/fine loss, 상태 추정값을 planner 입력으로 강제하지 않는 것**이다. 공통 perception에 raw 정보를 쓰는 두 번째 안은 규정의 간접 활용 예시에 가까운 설계를 만들 수 있지만, A2가 이미 수행한 query-only + occupancy/lane/motion 학습을 그대로 반복해서는 새 가설이 되지 않는다.

**규정 원문에서 확실한 것과 남는 해석**

| 원문 위치 | 확인된 내용 | 설계에 미치는 영향 |
|---|---|---|
| `OPEN_ISSUE.md:172` | 과거 pose로 현재 ego status를 계산할 수 있고, 이 계산에 쓴 과거 영상까지 입력할 필요는 없다. | status를 계산할 수 있다는 허용과 planner에 넣을 수 있다는 허용은 별개다. |
| `OPEN_ISSUE.md:200`–202 | raw/단순 임베딩 과거 정보의 planner 직접 입력은 금지. 여러 task의 공통 특징 개선 등 간접 활용은 허용. 코드로 판단. | `_status_encoding`을 scene encoder라고 이름만 바꾸는 것으로 경계가 바뀌지 않는다. |
| `OPEN_ISSUE.md:220`, 226 | 미래 정보도 같은 기준. 모델 여러 출력 중 선택에만 쓰는 예외 및 BEV/scene 형성 단계의 간접 활용을 확인. | 완성 후보의 마지막 선택은 근거가 있다. 앞단의 모든 planning 연산까지 예외라고 자동 단정하지 않는다. |
| `OPEN_ISSUE.md:247`–275, 특히 273 | 팀이 제시한 image-only value, goal/command query, 영상 0이면 정확히 출력 0인 planner도 불허. | 영상 ablation, value 경로의 분리만으로 planner 직접 조건 입력을 정당화할 수 없다. |
| `OPEN_ISSUE.md:285` | 이미지에서 추론한 ego history/status를 planning에 쓰는 것은 허용. | 이미지 추론 상태의 정확도가 충분하다는 뜻은 아니다. |
| `OPEN_ISSUE.md:233`–240 | 우회 확대해석을 인정하지 않으며 최종 코드 심사로 판단. | 이 보고서는 공식 사전 승인 판정이 아니다. |
| `OPEN_ISSUE.md:145`–149 | 과거 3초 영상의 간격/개수 선택 가능. 초기화 이후 과거와 현재 모든 model forward 시간 누적. | temporal 안은 마지막 프레임만 재서 latency를 주장할 수 없다. |

현재 SDV2가 **항상 미리 저장한 bank 행을 출력한다는 사실**은 유효하다. 그러나 그 사실과 “Q8 예외가 status-conditioned coarse planner 전체를 포함한다”는 판단은 다른 명제다. 반대로 현 코드를 무조건 규정 위반으로 확정하는 것도 원문의 후보 선택 예외를 무시한다. 가장 보수적인 실험은 직접 raw planner 경로를 없애고, 허용 설명과 실제 계산 경계가 일치하는 대조를 만드는 것이다.

**현재 status 계산 그래프**

`public_model.py`는 `experiments/sparsedrivev2_20260910/public_model.py`의 약칭이다. `relative_selector.py`도 같은 디렉터리다.

| 단계 | 코드 | raw status가 실제로 하는 일 |
|---|---|---|
| 이미지 | `public_model.py:304`–307 | `_status_encoding(status)`와 R34/FPN 이미지 처리는 분리되어 있다. backbone 자체에 상태가 공통 perception 조건으로 들어가는 것은 아니다. |
| 두 coarse stage | `public_model.py:315`–320 | 같은 8→256 status 임베딩을 path와 velocity 토큰에 매 stage 더한다. path DFA, velocity image attention, self-attention, FFN과 점수 head로 전달된다. |
| 이미지 읽기 | `public_model.py:120`–126, 145–165 | 상태가 섞인 query가 learned XY sampling offset과 attention weight를 바꾼다. DFA 출력은 `feature + output_proj(values)`라 입력 토큰이 잔차로도 보존된다. 이미지 value에서만 모든 정보가 온다는 구조가 아니다. |
| 후보 축소 | `public_model.py:323`–337 | path 1024→128→20, velocity 1024→64→10, 최종 20×10 행. 상태가 제거되면 shortlist 자체가 바뀔 수 있다. |
| 기본 fine score | `public_model.py:338`–345 | 상태가 포함된 path+velocity 토큰으로 trajectory DFA/FFN/imitation score를 계산한다. 앞단 status는 coarse에만 머물지 않는다. |
| 추가 relative score | `relative_selector.py:75`–91 | 후보의 6구간 XY 속도에서 raw `(vx,vy)`를 빼고, raw `(vx,vy,ax,ay)` 정규화값도 직접 포함한다. goal/endpoint-goal, 후보 가속도, 중심화된 base logit과 함께 32D를 만든다. |
| 마지막 선택 | `relative_selector.py:168`–195 | base score + 12,545-parameter MLP 잔차를 argmax하여 같은 candidate 좌표/ID를 반환. 좌표 이동·시간 재적분·보간은 없다. |

현 primary base의 `goal_mode='none'`은 **goal만 앞단에서 쓰지 않는다는 뜻**이다. `status_mode='causal_selection'`은 여전히 앞단에 `[0,0,0,0,vx,vy,ax,ay]`를 전달한다. mode 명칭만 보고 status-free라고 해석하면 안 된다. 배포용 현재 상태는 −10…0 pose의 nominal 10Hz causal fit이며, head goal은 제공된 +50 점의 현재 ego XY다. label/row 등은 `data.py:204`–209의 명시적 입력 whitelist를 통과하지 않는다.

현 CE 모델의 다른-scene 이미지 교체 실험은 recipient status/goal/calibration/GT를 고정하고 이미지 3장만 바꿨다. D3는 0.1307486→0.2866610, shortlist oracle은 0.0904941→0.2503397로 악화했다. 따라서 이미지가 후보 포함 여부에 강하게 영향을 준다는 증거는 있다. 이것만으로 status 위치의 규정 해석이나 status 없이 재학습했을 때의 성능을 결론내릴 수는 없다. 근거: `reports/sparsedrivev2_20260910/public_init/relative_soft_ce_image_ablation/result.json:58`–89. B8 값이며 B1 재현값과 섞지 않는다.

**P7/A2에서 배워야 할 실패와 한계**

| 관찰 | 허용되는 결론 | 다음 실험의 구체적 차이 |
|---|---|---|
| P7 C 0.327603, goal-prior 중심을 옮긴 G 0.429685. G−C +0.102082, CI [+0.072843,+0.130415]. | 해당 source-prior 변경은 실패. C도 ego-centered Gaussian prior라 중립 attention control이 아니다. | goal routing을 또 바꾸지 않고 양쪽 goal 정책을 고정한다. temporal score 경로 한 가지를 비교한다. |
| P7 GT compact MLP 0.088974 vs predicted compact MLP 0.574126, full image model 0.327603. | GT compact는 privileged diagnostic. full visual 경로를 24D MLP로 대체할 근거가 없다. | 예측 상태를 raw 상태 대신 강제로 넣는 bottleneck을 만들지 않는다. 상태 보조 출력은 loss에만 쓴다. |
| P7 image vx MAE train 약 .31→tune 약 .98m/s, 1s history XY .38→.99m. | estimator의 session 일반화 정밀도 문제를 실제로 관찰했다. | train/tune session별 상태 오차와 candidate velocity recall을 모두 본다. train 상태 loss 감소만으로 채택하지 않는다. |
| Frozen P7 GT21 대체의 D3 변화 −2.81e−5, plan 이동 5.32e−5m. | 해당 학습 완료 planner는 바꾼 21개 값에 거의 반응하지 않았다. stop logit은 바꾸지 않았다. | 상태 head 오차만 개선하는 실험보다, top-k 이전 점수 손실에서 시간 특징으로 가는 gradient와 후보 포함 개선을 검증한다. |
| P8 history `[.1,.2,.5,1]`→`[.2,.5,1,2]`는 이미지·pose·dt·label을 같이 변경. | 해당 넓힌 history arm 실패를 모든 temporal 사용 실패로 일반화할 수 없다. | 첫 screen은 짧은 history 2장을 고정하고 간격 sweep/큰 모델을 동시에 하지 않는다. |
| A2는 이미 5→32→32 zero-init query delta, image value/raw planner 입력 제외, plan/occ/lane/motion loss 1/.2/.2/.2. | “공통 query + 실제 보조 perception loss”라는 이름은 이미 시험했다. | 두 번째 안은 semantic prediction 출력 경계, perception utility gate, plan gradient 차단 단계가 실제 차이여야 한다. |
| A2 matched training provided .310104 vs zero .317294, 1 seed, 개선 .007190. 후속 모델 frozen status 제거는 .2939→.5628이지만 occ .435→.434, lane .494→.494. | 학습 이득과 평가 시 의존도는 다르다. 반올림 IoU에서 perception 개선 증거는 거의 없었다. | status가 perception을 실제 개선하는지 matched training으로 먼저 확인한다. 큰 D3 ablation 배율을 perception 개선으로 바꾸어 말하지 않는다. |
| Frozen correlation radius2/4/8의 boundary hit 0%, 확대해도 flat cost volume. RAFT road-pixel forward-flow scale도 작았다. | 시험한 frozen 표현/road-pixel 방법에서 metric motion 추출이 부정확했다. | radius 확대나 RAFT 출력 재적분을 반복하지 않는다. scene 전체의 학습된 temporal 점수 특징으로 다른 가설을 시험한다. |

P7 수치 근거는 `reports/MOTIONDRIVE_P1_P8_RETROSPECTIVE_20260908.md:110`–156, A2 구조는 `reports/motiondrive_v2_shared_status_a2_protocol_20260908.md:20`–51, A2 및 frozen 진단은 `reports/MOTIONDRIVE_V2_DIAGNOSTICS_20260909.md:33`–81이다. commit 계보는 P7 결과 `4dcb240`, state sufficiency `8d5d05f`, GT21 `fbf55f5`; A2 구조 `6b9a807`, source pin `6c4bc45`, 실행 증거 `1524b63`, 9/9 진단 `528d8c1`이다.

기존 진단 문서의 “경로는 사실상 해결”, “영상 실질 기여 기준 충족”, “flow로는 접근 불가” 등의 단정은 그대로 채택하지 않는다. 종방향 오차 우세는 특정 checkpoint의 관찰이고, 영상 ablation은 공식 코드 판정의 대체물이 아니며, 일부 frozen flow 시험 실패는 학습 가능한 모든 temporal 표현의 불가능성 증명이 아니다. 반대로 temporal을 새로 학습하면 이미지 vx 오차가 .01m/s 수준이 된다는 약속도 하지 않는다.

**안 1 — raw status-free temporal 점수 학습, 상태 aux는 loss 전용**

목표는 이미지가 **어떤 velocity 후보를 top-k에 남기고 어느 완성 행을 선택할지**를 직접 학습하게 하는 것이다. 모듈의 inference 입력은 current RGB 3장, past front RGB 2장, nominal dt, 카메라 calibration뿐이다. 이 안의 첫 대조에서는 raw pose 정렬도 쓰지 않아 raw motion 정보가 들어오는 경계를 분명히 한다. 제공 goal은 완성 후보가 나온 뒤의 별도 선택 단계에서만 양 arm에 동일하게 사용한다. raw status를 마지막 relative feature에도 넣지 않는다.

구체적인 최소 계산은 다음과 같다.

1. 현재 `[front-left,front,front-right]` 512×256 R34/FPN 및 native DFA는 유지한다. 첫 screen의 과거 front는 −1/−5 frame, 즉 nominal .1/.5초로 고정한다. 같은 R34/FPN 가중치를 공유하여 모두 인코딩한다. 카메라 수가 3으로 고정된 native DFA에 과거 카메라를 억지로 붙이지 않는다.
2. 별도 작은 temporal adapter는 현재 front와 과거 front의 FPN 특징, spatial position, dt를 입력받는다. 예를 들어 1/16 scale의 256D feature token을 사용하고, path/velocity 256D 토큰이 이 temporal token에 cross-attention한다. 현재/과거 위치가 같다고 단순 차감하거나 metric displacement라고 가정하지 않는다. adapter output projection을 zero-init하여 raw status=0인 동일 공개 초기 모델과 초기 함수가 같게 한다.
3. temporal 잔차를 **각 stage의 path/velocity scoring 및 top-k 이전**에 넣는다. 입력의 출처는 이미지다. 원 `_status_encoding`은 입력을 상수 0으로 고정하여 bias만 공통 상수로 남기거나, 두 arm에서 동일하게 비활성화한다. 실제 8D 상태나 예측 8D 상태로 이를 다시 채우지 않는다.
4. final candidate feature에도 이미지 시간 문맥을 전달하되, 반환 좌표는 계속 immutable bank lookup이다. 기본 scorer에 현 D3 soft CE 및 coarse path/velocity CE를 적용한다. `losses.py:62`–80의 현재 loss는 fine + .5×(path+velocity)이며 GT는 forward 뒤 loss에서만 사용된다.
5. temporal token에서 state/history auxiliary head를 분기한다. causal state/history를 training label로만 사용하며 **aux 예측값도 planner 입력에는 연결하지 않는다**. 단위/valid mask를 고정한 Huber 또는 기존 검증 loss를 양 arm에 동일 적용하고 weight는 사전에 고정한다. 추가 state label이 forward에 새는 것을 API 및 인자 추적으로 검사한다. “좋은 state 추정치가 있어야 planner가 동작”하는 전제를 없애되, aux 정확도는 계속 진단한다.
6. 마지막 selector가 필요하면 후보 속도·가속도 같은 고정 행 특성, 이미지 base score, 제공 goal/endpoint-goal만 사용한다. 기존 32D head에서 `candidate_velocity - raw_velocity`, raw velocity/acceleration 네 값은 제거한다. 이 경우 기존 CE relative head는 입력 의미가 달라 **그대로 재사용 가능한 완성 head가 아니다**. 두 arm의 같은 신규 head를 다시 학습한다.

대조는 **current-repeat 대 real-history** 한 쌍이다. 양쪽 모듈/파라미터/이미지 forward 횟수/dt/augmentation/공개 초기화/row 순서/update 수/손실을 맞춘다. control은 과거 슬롯에 현재 front 이미지를 반복하여 현재 정보만 제공한다. temporal augmentation은 같은 front 카메라에서 시간에 걸쳐 같은 photometric 값을 사용한다. 이것은 과거 P7/P8에서도 이미 한 조치(`scripts/motiondrive_v2_data.py:388`–391)이지, 과거 실패 원인으로 새롭게 주장할 것은 아니다. real-history와 current-repeat 모두 raw status-free 상태로 학습한다. raw-status로 학습한 모델을 평가 시 갑자기 zero로 바꾼 값은 control 학습 결과가 아니다.

P7와의 delta는 **continuous token 존재**가 아니라 **state 복원 정확도에 전부 맡기지 않는 bank score 목적함수 및 후보 포함 경로**다. 따라서 첫 확인값은 최종 D3와 함께 stage별 GT-near path/velocity recall, 최종 shortlist oracle, final selection regret, temporal adapter로 향하는 coarse/fine gradient 크기다. state MAE만 좋아지고 oracle/D3가 그대로면 aux 개선이 실제 planning으로 연결되지 않은 것으로 판단한다. oracle이 크게 나쁘면 fine head만 오래 학습하지 않는다.

이 안의 위험은 metric ego speed를 단안 시간 영상에서 정밀하게 식별하기 어렵고, current-repeat 대조를 이겨도 .15에 도달하지 못할 수 있다는 점이다. original NAVSIM decoder는 native status 조건으로 학습되었으므로 status를 제거하면 공개 가중치의 입력 분포가 바뀐다. zero-init는 새 branch의 초기 변화만 제어하고 이 분포 차이를 해결하지 않는다. .1/.5초·1/16 token은 계산량을 제한한 첫 설계값이며, 그 선택이 최적이라는 실측 근거는 없다. 여러 history/해상도 sweep을 동시에 시작하지 않는다.

**안 2 — 실제 perception 작업을 통한 raw pose/status 간접 사용**

목표는 raw 정보를 planning query로 옮겨 전달하는 것이 아니라, **과거 영상 정합 및 현재 scene 인식 결과를 개선하는 데 사용**하는 것이다. Q7의 공통 특징 간접 활용 설명에 근거하지만, 모든 유사 설계를 포괄하는 사전 승인을 뜻하지 않는다.

A2와 다른 계산 경계를 명확히 하기 위해 다음과 같이 최소 2단계로 분리한다.

1. 공통 perception encoder는 현재 3장+같은 과거 front 2장, calibration, 해당 영상의 current→past SE(3)를 사용한다. pose는 과거 이미지 feature의 공간 정합에만 쓰고, 4×4 matrix나 이를 flatten한 벡터를 candidate query에 보내지 않는다. control/provided 모두 동일한 실제 pose 정합을 사용하면 대조는 추가 현재-status 조건의 가치만 측정한다. 이 비교를 “pose 자체의 이득”이라고 부르지 않는다.
2. 현재 status 조건은 작은 zero-init query module로 **perception sampling/temporal fusion 내부에서만** 사용한다. 이미지 value 및 공통 encoder의 task head 앞에 raw-status 잔차를 별도로 붙이지 않는다. 원 SDV2 `_status_encoding`에는 양쪽 모두 상수 0만 전달한다.
3. perception은 현 캐시의 `occ_target/occ_valid`, `lane_target/lane_valid`를 사용하여 실제 공간 예측을 한다. `scripts/build_scene_supervision_v2.py:23`–30에 따르면 occupancy는 현재 non-ego annotated object footprint의 제한된 support이며 complete free-space/전체 점유/drivable-area GT가 아니다. lane은 실제 `map.parquet` 선의 제한된 camera-visible support다. unknown은 mask로 제외하고 새 3-camera visibility를 반영한다. 기존 6-camera valid mask를 그대로 재사용하여 보지 못하는 곳을 완전 관측처럼 학습하지 않는다.
4. **perception-only warm-up**에서 control(status0)/provided(status실제)을 같은 초기화·row·update·loss로 학습한다. state GT를 그대로 입력하면서 state 복원을 보조 작업이라고 세는 방식은 피한다. 그것은 trivially input copy가 될 수 있고 실제 영상 인식 개선의 증거가 아니다. 최소 실제 task는 위 object occupancy와 lane이다.
5. 첫 planning 비교에서는 조건부 perception 경로와 공유 backbone/FPN을 함께 고정한다. candidate scorer에는 이 경로의 **predicted semantic probability raster만** 넘기고, raw status/pose/conditioned hidden256D는 넘기지 않는다. 별도 작은 image-semantic adapter가 candidate 위치에서 semantic 증거를 읽어 점수 잔차를 만든다. 원 공개 status-free DFA/current-image 경로도 유지한다. planning loss의 gradient가 조건부 perception으로 돌아가지 않게 하여, task 출력에 planning용 raw state를 숨겨 전달하도록 학습되는 경로를 첫 비교에서 차단한다. 출하 추론 시에는 이미지→perception→scorer를 모두 실행하며 GT map/object/cache logits를 읽지 않는다.
6. 기본 decoder와 semantic adapter를 동일 D3 coarse/fine loss로 학습한다. goal은 완성 후보의 마지막 선택에만, 양쪽 동일하게 쓴다. 마지막 relative head에도 raw status를 다시 넣지 않는다. 모든 최종 좌표는 고정 bank 행이다.

이 경계는 규정이 요구한 필수 구현이라고 주장하지 않는다. 실제 perception task의 출력만을 통한 간접 활용임을 검토하기 쉽게 만들고, 과거 A2의 단순 query hook과 구별하기 위한 **제안**이다. 조건부 perception과 backbone이 planning 중 계속 업데이트되면 위 freeze 조건이 깨진다. 그 joint 실험은 첫 대조 이후 별도 결정해야 한다.

새 task별 prediction 개선을 먼저 평가해야 하는 이유는 명확하다. A2 후속 frozen 진단에서 status 제거는 D3를 크게 바꿨지만 occ/lane IoU는 거의 바꾸지 않았다. 다만 이것은 matched perception-training 효과의 직접 추정치가 아니므로 **현재 branch가 perception에 전혀 기여하지 않는다고 단정할 수는 없다**. 새 안은 양 arm의 실제 tune task 성능/영역별 오차 및 동일 입력 visual sensitivity로 그 모호함을 줄인다. IoU에 meaningful 개선이 없으면 “task가 있기 때문에 간접 활용”이라는 설명만으로 planning 학습을 확장하지 않는다. 좁은 static occupancy/lane 작업은 raw vx/ax로 얻을 정보가 적을 수 있다. 이 안은 status가 가진 정밀한 v0를 planner에 전달하는 우회로를 보장하지 않으며, 따라서 D3 이득도 작을 수 있다.

전체 계산량과 변경 범위는 안1보다 크다. perception raster decoder/semantic adapter는 새로 학습해야 하며 public NAVSIM에서 pretrained perception head를 가져온다고 말할 수 없다. 부족한 semantic task 때문에 dynamic-object motion 같은 새 supervision을 추가하는 것은 가능성일 뿐이며, 이번 최소 안에는 포함하지 않는다. 현 object/map cache의 제한된 support, session split ancestry와 train-only 생성 계약을 먼저 재사용한다.

**안 2의 공통 hidden feature 변형 및 3-arm 최소 설계**

Q7은 semantic raster 출력만 planner에 보내라고 요구하지 않는다. **여러 task가 실제로 공유하는 hidden scene feature**에서 간접적으로 status를 활용하는 것도 원문이 허용하는 예시에 들어간다. 앞의 semantic-output/freeze 설계는 추가로 보수적인 비교안이고 의무 조건이 아니다. root가 같은 2,000-update budget으로 3개 arm을 구성한다면, 다음 공통 feature 변형이 더 작은 변경이다.

| arm | temporal 슬롯 | 원시 상태의 유일한 사용 위치 | 비교가 답하는 질문 |
|---|---|---|---|
| A | 현재 front 반복 2개 | 실제 status 없음; 공통 condition에 0 | status-free 현재 정보 control |
| B | 실제 front 과거 .1/.5s | 실제 status 없음; 공통 condition에 0 | A 대비 과거 이미지 정보의 가치 |
| C | B와 동일 | 후보와 독립인 temporal image/FPN fusion의 perception query condition | B 대비 raw status의 공통 perception 간접 활용 가치 |

세 arm 모두 같은 신규 temporal fusion, 공통 condition module, occ/lane/state auxiliary head를 포함하고 동일 공개 초기화에서 시작한다. 작은 condition module의 마지막 projection을 0으로 초기화한다. 원 SDV2 `_status_encoding` 입력은 **세 arm 모두 0**이고, late selector에도 raw state를 넣지 않는다. 공통 condition=0 arm도 동일 parameter를 가지며 학습 후 constant query shift를 배울 수 있게 한다. 초기 출력 equality 및 RNG reset은 모든 새 모듈을 생성한 뒤 확인한다.

C의 구체적인 작은 구현은 **candidate bank/path/velocity query와 독립인 front temporal FPN fusion**이다. 현재 front의 spatial query가 두 과거 front의 image key/value에 attention하고, current status는 이 **perception query**를 조건화한다. 결과 image feature map에는 image current residual과 image attention value만 들어가며, raw status embedding을 residual·concat·전용 token으로 더하지 않는다. 갱신한 front FPN과 나머지 현재 두 camera FPN을 원 3-camera DFA가 그대로 읽고, 동일 fused image feature를 별도 현재-object occupancy/lane head도 읽는다. 따라서 조건 위치는 후보 scoring query가 아니라 실제 시각 특징의 형성 단계다. 이는 공개 DFA의 3-camera/256-channel 계약을 유지할 수 있다. BEV raster supervision head는 camera calibration을 사용한 별도 작은 spatial readout이며, 실제 label valid 영역을 확인해야 한다.

state/history aux는 **status 조건을 받기 전 image-only temporal 특징**에서 분기한다. raw current state를 입력받은 공통 특징으로 같은 current state를 맞히는 aux는 복사 task가 될 수 있어, C의 perception 개선 증거로 계산하지 않는다. 조건화된 공통 특징의 실제 task는 object occupancy와 lane이며 두 작업의 출력/손실/valid mask와 planner 입력 경로가 일치해야 한다. 모델 안에 raw status를 복원하는 전용 planner bottleneck이나 수치 통로를 추가하지 않는다. Image-only value라는 형식만으로 허용을 확정할 수 없다는 Q273의 주의도 유지한다. 여기서 다른 점은 query가 **planner가 아닌, candidate와 독립적이며 실제 여러 task가 공유하는 perception 단계**라는 계산 경계다.

이 C안은 hidden common feature를 planning과 joint 학습하므로 앞의 semantic-output/freeze안보다 값의 전달 경계를 좁게 강제하지 않는다. 그만큼 source 경로와 학습된 task 성능을 함께 제시해야 한다. A2와 비교한 실질적 차이는 (i) SDV2의 pretrained coarse/fine scoring 및 dense bank와 D3 supervision, (ii) fixed spatial current/history image fusion에 앞서 조건을 적용하는 위치, (iii) 단계별 shortlist recall/oracle 및 perception utility를 공동 판정한다는 점이다. **“멀티태스크 loss가 있다”는 문구 자체는 delta가 아니다.** A2를 다시 만든 것인지 구현 후 source 그래프를 확인하고, B↔C matched 결과가 없기 전에는 개선을 예상 수치로 약속하지 않는다.

세 arm의 task 가중치/동일 backbone freeze 정책/시간 photometric jitter/은행/rows/seed/update 수를 함께 고정한다. state aux의 weight는 task 간 gradient 크기를 확인하되 tune 결과를 보고 반복 조절하지 않는다. 결과는 A↔B, B↔C 두 비교로 보고하며 A↔C만의 큰 개선으로 temporal 또는 status 한쪽 효과를 단독 주장하지 않는다. C에서 D3만 좋아지고 perception task가 같다면 유용한 score 특징인지 원시 상태 우회 전달인지 구분이 여전히 어렵다. 원문 간접 활용의 의미를 자동 충족한다고 포장하지 않고, 필요하면 별도의 semantic-output/freeze 대조로 기여 경계를 더 좁힌다.

**root가 전달한 frozen sensitivity 초기 결과에 따른 업데이트**

문서 초안 뒤 root가 다음 수치를 전달했다. 이 부분은 **root 실행 결과 전달값**이며, 이 검토자가 해당 최종 receipt를 독립 재계산한 수치가 아니다. `P7 status literal`의 실제 row/시점/오차 정의와 exact perturbation manifest는 root receipt에 따라 읽어야 한다.

| frozen 조건 | D3 | shortlist oracle |
|---|---:|---:|
| 원 입력 | .130749 | .090494 |
| P7 predicted status literal, coarse/fine 양쪽 | 1.170503 | 1.112857 |
| P7 predicted status literal, final head만 | .243276 | 원 후보 집합 고정 여부는 root receipt 확인 |
| 양쪽 status 0 | 12.7001 | 12.6905 |
| 양쪽 vx ±.1m/s | .1572 / .1603 | 전달되지 않음 |
| 양쪽 vx ±.05m/s | .1364 / .1396 | 전달되지 않음 |

현재 학습 완료 모델은 작은 vx 변화에도 목표 수치 근방으로 움직이고, P7 추정치를 coarse까지 넣으면 **oracle 자체가 1.11m**로 무너진다. 이것은 정확하지 않은 상태를 단순 대체한 뒤 fine head만 고치는 접근을 즉시 배제할 근거다. exact raw 상태를 이미지 추정으로 대체할 수 있다는 전제도 지지하지 않는다. 동시에 zero status의 12.7m는 frozen OOD 진단이며 **status-free 새 학습의 달성 가능 최저값을 뜻하지 않는다**. 다음 작업은 A/B 및 조건부 C를 같은 공개 초기화에서 다시 학습하여 coarse 후보 포함과 final 선택을 함께 재형성하는 것이다. 기존 .13에서 바로 개선이 이어지는 continuation으로 설명하면 안 된다.

**공개 NAVSIM 재사용 범위**

공식 source pin은 `swc-17/SparseDriveV2@696ef77924eb9e0a4b4047d013a50e9854bfa026`, public checkpoint는 `wenchaosun/SparseDriveV2/sparsedrive_navsimv1_92p2.ckpt`, HF revision `42c15531aa6cb18c7360dd0377cfc1c221280597`, SHA256 `330072b133981f77b0b18b669216ad50b211552a82ea45ac52d507ec9021a735`이다. 공식 repository/weight 경로는 [pinned code](https://github.com/swc-17/SparseDriveV2/tree/696ef77924eb9e0a4b4047d013a50e9854bfa026), [checkpoint revision](https://huggingface.co/wenchaosun/SparseDriveV2/tree/42c15531aa6cb18c7360dd0377cfc1c221280597)다. 이 검토에서는 설치된 pinned 원본을 읽었으며 새 인터넷 검색이나 가중치 다운로드를 수행하지 않았다.

| 항목 | 실제 재사용 가능한 내용 | 주장하면 안 되는 내용 |
|---|---|---|
| Vision | NAVSIM 계획 점수 학습을 거친 ResNet34와 4-level FPN256, BN 통계. RGB 3 current cameras 512×256. | occupancy/lane/temporal ego-motion의 task-specific pretrained encoder라고 단정. |
| Planning representation | 50×3 path→256 MLP, 8-value velocity→256 MLP, 2 decoder stage의 native DFA/camera encoder/self-attention/FFN/path·velocity heads, 마지막 trajectory DFA/imitation head. | ETRI 100m path/새 velocity bank가 원 public 50m bank와 입력분포까지 같다고 주장. |
| Status | original 8D command4+velocity2+acceleration2 Linear256 tensor shape와 값. | raw status를 없애도 native conditioned function/성능이 보존된다고 주장. 안1/2에서 weight를 로드해도 raw input 경로는 비활성이다. |
| Metrics | NAVSIMv1 six metric heads: collision, drivable-area compliance, driving-direction compliance, TTC, comfort, progress의 **candidate score** heads. | 이 head를 공간 occupancy/lane 검출기 또는 perception multitask head로 계산. |
| Temporal/new semantic heads | 공유 R34/FPN 및 기존256D attention/FFN 일부를 새로운 입력에 적용 가능. | NAVSIM temporal fusion, image-only precise velocity, ETRI semantic raster decoder가 공개 weight에 이미 들어 있다고 주장. |
| Relative head | 현 ETRI의 32→128→64→1 head는 새 ETRI 학습 모듈. | NAVSIM 공개 parameter로 계산하거나 raw-status feature를 제거한 뒤 동일 pretrained 완성 head라고 주장. |

공식 `sparsedrive_features.py:50`–56는 command/velocity/acceleration을 concatenate하고, `:84`–85는 **only last frame used**라고 명시한다. original `custom_decoder.py:238`–268은 GT path/velocity/trajectory distance 기반 soft CE, `:270`–288은 PDM simulator에서 얻은 candidate metric의 BCE 학습이다. NAVSIMv1 실행 recipe는 `scripts/training/sparsedrive_navsimv1.sh:10`–21의 navtrain, 6개 metric이며, original 코드에 실제 segmentation/object detection head는 없다. 기본 config는 NAVSIMv2 8-head이므로 이것을 다운로드한 NAVSIMv1 6-head checkpoint와 혼동하면 안 된다. 원 v1 script의 최종 velocity filter는 20이고 ETRI adapter의 최종값은 10이다. 공개 README의 PDMS 92.22는 ETRI D3=.15의 직접 근거가 아니다.

초기 원 public bank 그대로는 400 tensor, 50,370,143 elements를 로드했다. ETRI bank 교체에서는 4개 bank tensor를 의도적으로 교체하고 **396 tensor, 41,825,887 elements**를 재사용했으며, 그중 **learnable parameter는 41,808,827개**다. 근거: `reports/sparsedrivev2_20260910/public_init/public_coverage.json`, `etri_p256_v128_coverage.json:1`–20, `ADAPTATION.md:8`–31, 68–76. 이 수는 checkpoint→초기 ETRI adapter의 coverage다. 이후 학습 완료 tensor가 공개 checkpoint와 여전히 bitwise 같다는 뜻이 아니다. 안1/2의 신규 temporal/perception/score adapter는 coverage 분모와 별도로 보고하고, 로드했지만 비활성인 `_status_encoding.weight`도 사용 중 parameter와 구별해야 한다.

**root의 frozen 진단 이후 결정 기준**

현재 sensitivity 실행은 root 소유다. 아래는 초안에서 사전에 정리한 해석 기준이며, 위 업데이트의 전달 수치에도 같은 구분을 적용한다.

- **coarse status만 변경, final relative status는 실제값 고정**: 두 stage의 path/velocity IDs, final shortlist oracle, original candidate와의 overlap을 본다. oracle이 크게 악화되면 late selector를 고치는 것만으로는 없는 후보를 되살릴 수 없다. 이 경우 안1의 top-k 이전 temporal score 학습이 우선이다.
- **base forward/candidates를 고정하고 final relative status만 변경**: fixed 200행 selection reliance를 분리한다. candidate XY/IDs가 bitwise 동일해야 한다. oracle은 동일해야 하며 D3 변화는 이 shortlist 안의 선택 변화다. coarse status 의존을 설명하는 실험으로 부르지 않는다.
- **둘 다 변경**: 실제 배포 경로의 총 민감도지만 두 영향은 비선형이고 상호작용하므로 앞의 두 D3 차이를 단순 합하지 않는다.
- constant zero는 frozen OOD 입력일 수 있다. small perturbation, same-speed matched donor 등 root가 사전에 정의한 조건의 변화를 함께 보되, 이를 image-state estimator 오차분포나 재학습 결과로 동일시하지 않는다. 일부 privileged GT 치환/후보 oracle은 분석용 upper-information 값이며 출하 성능의 예측치가 아니다.
- 안1 screen을 시작하면 **양쪽 raw-status-free 재학습**을 같은 public/허용 train-only 초기화에서 비교한다. 안2는 perception 단계 자체의 matched 개선과 소스 경계가 확보된 뒤 진행한다. .13의 기존 head에 raw status 대신 부정확한 예측값만 꽂고 실패하는 실험은 이미 알려진 분포 차이를 재확인할 가능성이 높다.

두 안 모두 현재 primary bank/split/row hash, pretrained checkpoint pin, 초기 assembled weight hash, RNG와 augmentation 순서, 완성 row output identity를 유지·기록한다. 기존 confirmation population은 이미 한 번 평가된 상태이므로 새로운 안의 반복적인 선택 세트로 재활용하지 않는다. 먼저 지정된 train/tune의 matched terminal 결과, session 단위 불확실성, B1 수치 조건으로 판정한다. temporal 비용은 모든 과거/현재 인코딩을 포함하고, B200 측정값을 RTX4090 값으로 바꾸어 말하지 않는다.

.15 목표에 대한 현재의 명확한 근거는 dense bank와 실제 학습된 selector가 특정 검증 집합에서 그 근방 이하를 냈다는 것이다. **raw status-free 구조가 그 성능을 유지한다는 근거는 아직 없다.** 다음 실험은 정밀한 image-state를 가정해 목표 수치를 약속하는 대신, raw status를 제거했을 때 무너지는 단계가 후보 포함인지 마지막 선택인지 먼저 분리하고, 그 단계에 실제 영상 손실 경로를 추가해야 한다.

**읽은 핵심 소스 식별값**

| 파일 | SHA256 |
|---|---|
| `OPEN_ISSUE.md` | `ba8b50097450612b985a250fc7f75027467d1da1140555fdc155926efc079006` |
| `experiments/sparsedrivev2_20260910/public_model.py` | `2c7da9bc1fee9216513a606816c6f4e53d714812f0c4c33fe3ae7678a2b513c7` |
| `experiments/sparsedrivev2_20260910/relative_selector.py` | `7fa75415a40691f555cca27ab9fe88331725177f19887c01a0e887121e2ac59f` |
| `models/motiondrive_v2/scene_encoder.py` | `99cad3d3fe52ba76842d7dbd03754e09b1d24928e2a05522b8f69b42de1d9808` |
| `reports/motiondrive_v2_shared_status_a2_protocol_20260908.md` | `d8a8e80ee6517e3b71a2639a675b6be5a16ac3f546ac02890930e63552b28392` |
| `reports/MOTIONDRIVE_P1_P8_RETROSPECTIVE_20260908.md` | `fa586d15f1cb97792a59c5539015e7f2822c1be1abea7850b161690ccde9d8e3` |
| `reports/MOTIONDRIVE_V2_DIAGNOSTICS_20260909.md` | `d33c5386cfb318405bc33471def98a1cefd5373fd1cb432cb6ea6dd34e28ef08` |
