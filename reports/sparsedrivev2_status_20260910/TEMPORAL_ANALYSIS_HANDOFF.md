# Temporal comparison CPU analysis handoff

분석기: `analyze_temporal_comparison.py`. 모델을 로드하거나 GPU를 사용하지 않는다. A/B/C는 각각 `temporal_a_repeat_s0_v1`, `temporal_b_history_s0_v1`, `temporal_c_common_s0_v1`이며, 같은 공개 초기화·은행·학습 행을 사용한 고정 2,000-step terminal만 비교한다. 이전 raw-status CE의 D3 0.130749는 학습 방식과 입력 경로가 다른 참고값이다.

실제 초기 자료에 대한 검증은 `temporal_preflight_analysis_v1/aggregate.json`에 있다. 세 arm의 step 480까지 저장된 49개 step에서 row SHA·epoch·LR·occ/lane valid pixel 수가 같았다. 초기 planning 배열은 bitwise 동일, 초기 occ/lane IoU는 동일하며, 초기 state aux는 A/B 차이를 허용하고 B/C 동일을 확인했다. 자체검증 5개도 통과했다.

```bash
python3 /home/a/adcl_status_20260910/analyze_temporal_comparison.py --self-test
```

세 학습 프로세스가 정상 종료하고 terminal 평가 저장까지 끝난 뒤 다음 원격 run 루트에서 자료를 수집한다.

```text
/NHNHOME/data/sukim/adcl/experiment_worktrees/sparsedrivev2_20260910/
  work_dirs/sparsedrivev2_status_20260910/
    temporal_a_repeat_s0_v1/
    temporal_b_history_s0_v1/
    temporal_c_common_s0_v1/
```

각 run에서 필요한 항목은 다음과 같다. run 내부 디렉터리 구조를 그대로 복사하며, `source/native_ops/`도 포함한다.

- `manifest.json`
- `source/` 전체
- `eval_000000.json`, `eval_000000.npz`
- `eval_002000.json`, `eval_002000.npz`
- `train.jsonl`
- `result.json`

`last.pth`는 이 분석에 필요하지 않다. 분석기는 checkpoint 파일을 열거나 현재 weight bytes를 직접 검증하지 않는다. 공개 초기화·은행·runtime source와 초기 GPU 계약은 manifest 및 복사된 source receipt를 검증한다. 실제 process exit 0 여부는 root의 durable launch receipt로 별도 확인한다.

예를 들어 위 세 run을 `/home/a/adcl_status_20260910/results/temporal_full_v1/` 아래에 복사했다면:

```bash
python3 /home/a/adcl_status_20260910/analyze_temporal_comparison.py \
  --runs \
  /home/a/adcl_status_20260910/results/temporal_full_v1/temporal_a_repeat_s0_v1 \
  /home/a/adcl_status_20260910/results/temporal_full_v1/temporal_b_history_s0_v1 \
  /home/a/adcl_status_20260910/results/temporal_full_v1/temporal_c_common_s0_v1 \
  --output /home/a/adcl_status_20260910/temporal_analysis_v1
```

출력 디렉터리는 없어야 한다. 검증이 실패하면 최종 보고서 디렉터리를 만들지 않는다. 성공 시 `aggregate.json`, `RESULTS_KO.md`, `analysis_receipt.json`을 생성하며 모든 입력 artifact SHA와 분석기 SHA를 기록한다. `--preflight`를 추가하면 terminal 파일 없이 초기 평가와 세 run의 공통 학습 prefix만 검사한다.

최종 검증·집계 범위:

- source/public/bank/parameter 수 및 선언한 개입 외 manifest의 완전한 일치, train 54,810행과 tune 1,998행의 고정 SHA, train/tune session 분리.
- 초기 planning의 rows/pred/ID/D3/oracle/6점 L2/XY error bitwise 동일, 초기 occ/lane **집계 IoU** 동일. Raster logits는 저장되지 않아 tensor parity를 검증하지 않는다.
- step 1과 10의 배수에 저장된 201개 학습 로그의 batch row SHA·epoch·LR·유효 label 수 일치. **저장되지 않은 나머지 batch 순서는 사후에 직접 증명할 수 없다.** 같은 RNG reset과 동일 source/row 조건은 검증되지만 이를 전체 batch 기록 검증으로 표현하지 않는다.
- terminal selected D3/6점 L2는 저장된 FP32 XY error에서 float64로 독립 재계산한다. 원 FP32 summary도 배열과 대조하며 반올림 차이를 보존한다. GT는 pred−error로 복원한 값끼리 2e−5m 이내를 검사할 뿐 별도 canonical GT 파일은 로드하지 않는다.
- shortlist oracle은 전체 후보가 저장되지 않아 기록된 FP32 값을 사용한다. finite/nonnegative/selected D3 이하를 검사하며 regret은 재계산 D3−저장 oracle이다. 같은 bank ID가 등장하면 좌표가 항상 bitwise 같은지 검사한다.
- B−A와 C−B에 공통 20,000회(seed 0) session bootstrap을 적용한다. 11개 session을 복원추출하고 각 session의 전체 행을 포함해 frame-weighted 평균을 계산한다. D3/oracle/regret/3초 L2 및 영상 state aux MAE의 paired 95% percentile CI를 산출한다. Occ/lane은 aggregate IoU 차이만 보고한다.

이는 반복 사용한 tune의 단일 seed 탐색 대조이며 독립 confirmation 또는 학습 seed 불확실성 검증이 아니다. 다중 비교 보정도 하지 않는다. C의 D3 개선만으로 perception 개선이나 규정 승인을 주장할 수 없다. 영상 state aux는 C에서도 원시 상태를 직접 읽지 않는 독립 예측이며 planner 입력으로 쓰이지 않는다.
