# 과거 MotionDrive V2 selector와 현재 C의 차이

P×V 후보와 learned selector는 새 발상이 아니다. 과거 실패를 다시 검증하는 실험이며, 현재 oracle 개선만으로 이번 selector 학습 성공을 주장할 수 없다.

| 항목 | 09-09 MotionDrive V2 P/V | 현재 temporal C + scene head |
|---|---|---|
| 후보 | stop 1 + P512×V127 = 65,025, 전수 score | bank P1024×V1024에서 이미지로 P20/V64 유지, 1,280개 score |
| 후보 포함 성능 | full-bank oracle 0.1048869705 | 실제 image-selected P20×V64 oracle 0.0714597765 |
| 시각 특징 | planner `[6,128]`을 전부 압축한 공통 32D context와 static P/V key의 cosine 내적 | 각 완성 궤적의 DFA 영상 샘플링과 candidate self-attention 이후의 256D token |
| 초기화 | 기존 parent 재사용, 추가 P/V scorer는 새 random head | NAVSIM 공개 planner 396 tensor 재사용 후 ETRI C 2,000step; 추가 37,697p scene head 자체는 새 random head |
| 손실 | expected D3 + 0.1 soft CE, 전체 trunk도 학습 | 먼저 C 고정, final score soft CE만 학습; real/zero-token 동일 head 비교 |
| 학습량 | 2,000×16 = 32,000행 노출, train54810의 약 0.584회 | 사전 선언 4,000×128 head 학습은 약 9.34회 노출; 동일 학습량 대조가 아님 |
| 실제 결과 | direct 0.2939324227 / pv 1.5094420481 / residual 1.5107735537 | 확장만 한 기존 scorer: V10 0.3133228924 / V32 0.3239029684 / V64 0.3359242555. 새 head 결과는 별도 실험 결과를 기다려야 함 |

과거 P/V 구현의 근거는 main repo `models/motiondrive_v2/pv_planner_head.py:64`의 6×128→128→32 context, `:83`의 path/profile joint key, `:94`–101의 shared-context cosine score다. 후보별로 이미지/BEV를 다시 읽는 attention은 없다. `reports/motiondrive_v2_pv_screen_protocol_20260909.md:57`–62도 이를 명시한다. 새 head random 초기화는 `scripts/run_motiondrive_v2_pv_screen.py:254`의 생성과 `:279`–284의 parent checkpoint missing-key 예외가 정확히 `factorized_pv_head.*`인 것으로 확인된다. 최초 commit은 `7d4156a`(2026-09-09 11:07:56 +0900, “add fixed P V planning head screen”)다.

현재 원 public 구현 `experiments/sparsedrivev2_20260910/public_model.py:336`–342는 완성 P/V 조합마다 trajectory DFA와 self-attention을 실행한다. 그 직후 `traj_mlp` 입력을 현재 `c_scene_selector.py`가 추출한다. 공개 tensor 재사용은 strict evaluator의 396 tensor / 41,808,827 learned parameters 검사로 확인한다. 따라서 이번 차이는 candidate-conditioned representation의 출처와 계산 방법이다. **“후보별 시각 feature를 처음 사용했다”는 설명은 틀리다.** 이전 M13에서도 후보별 경로 BEV를 사용했다.

과거 P/V 실패는 coverage 부족으로 설명되지 않는다. `work_dirs/motiondrive_v2/pv_screen_b0_{direct,pv,pv_residual}_last2000/final_eval.json`의 report와 records를 읽어 위 수치를 확인했다. PV는 1,998행에서 142개의 선택 ID를 사용하고, 가장 흔한 ID도 189행(9.46%)이었다. score entropy 평균 7.661464(균등 65,025개는 11.082527)이므로 **하나의 mode로 완전 collapse했다고 단정할 수 없다.** 같은 디렉터리 `metrics.jsonl` 마지막 10개 로그의 **학습 minibatch 평균** selected D3는 2.0108845, expected D3는 2.0785412, soft CE는 11.5513962다. 첫 10개 로그는 각각 9.9591487 / 7.7876911 / 11.1169893이었다. expected loss는 감소했지만 CE와 정밀 ranking은 해결되지 않았다. 이는 full-train 평가값이 아니며, 최적화·표현·목적함수 중 하나를 단독 원인으로 특정하지 않는다.

M13은 더 직접적인 재실패 경고다. `reports/MOTIONDRIVE_V2_SELECTOR_LATENCY_20260910.md:33` 이후에서 shared latent 정상 0.2754 vs zero 0.2712, 후보별 BEV 정상 0.2751 vs zero 0.2723을 기록했다. “시각 특징을 넣었으니 개선된다”는 가정은 이미 반증 사례가 있다. 이번에도 동일 구조·초기화·행 순서로 real-token이 zero-token보다 좋아지는지 실제 TUNE으로 검사해야 한다. P5-Z의 ZERO는 시각 token을 0으로 만든 대조가 아니라 정지 궤적 후보다. base0 세 seed의 변화 0, base1 약 1e-4 개선만 있었으므로 M13과 혼동하지 않는다(`reports/p5_zero_selector_head_results_20260908_ops.json:63` 이후).

과거와 현재의 split은 같은 train54810/tune1998이다. 현재 TUNE은 이미 여러 판단에 반복 사용되었고 학습량/표현/목표도 달라졌으므로, 이번 성공이 있더라도 새 표현 하나의 독립적 인과 효과나 미사용 최종평가 성적으로 표현하지 않는다.

원 PV final_eval SHA256: `33271f1f60b570e88482b43d6dde9ca5b2b50dbb7bfcc4818f505647b1db2046`; metrics SHA256: `5f671b05538e5b3c9c4a38d4f1e3ef3c5cd0c323b9b5ec626c133b2f1ac051b3`. 이번 stage 진단은 `VELOCITY_DIAGNOSIS.json` 및 `velocity_results/`의 source/receipt/행별 NPZ에 보존했다.
