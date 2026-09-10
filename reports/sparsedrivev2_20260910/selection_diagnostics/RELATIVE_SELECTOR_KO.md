# 완료된 후보의 상대 운동 특징을 이용한 점수 보정 준비

2026-09-10. 학습·GPU 실행 없이 독립 CPU 모듈을 준비했다. 성능 개선 결과는 아직 없다.

`experiments/sparsedrivev2_20260910/relative_selector.py`는 이미 완성된 `candidate_xy`를 읽어 후보마다 32차원 특징을 만들고 `RelativeScoreHead`의 잔차를 기존 점수에 더한다. 좌표 생성·수정·정규화된 좌표의 출력 복원이 없으며 최종 궤적은 입력 후보 중 하나를 그대로 선택한다. 목표점과 causal 상태의 새 영향은 이 잔차 점수에만 들어간다. 원 모델이 상태에 따라 shortlist를 달리 선택하는 기존 동작은 유지되므로, 상태를 바꾼 **전체 forward의 후보 ID까지 동일하다는 주장은 하지 않는다.** 고정된 base output에 대한 `rescore`의 후보 ID·좌표는 동일 객체로 유지된다.

공통 API:

```python
features = make_candidate_features(output, status, goal_xy=None)  # FP32 [B,K,32]
head = RelativeScoreHead()                                      # 12,545 parameters
residual = head(features)                                       # FP32 [B,K]
model = CandidateRelativeSelector(base, base_goal_mode="none")
model.relative_head.load_state_dict(saved["relative_head"])
```

`output`은 `candidate_xy[B,K,6,2]`, `scores[B,K]`, 선택적으로 `candidate_valid[B,K]`만 필요하다. GT/미래 정답 인자는 없다. wrapper의 `rescore`는 추가로 `candidate_ids`를 읽는다. 상태 `[B,8]` 중 4:8의 현재 causal `vx,vy,ax,ay`만 쓴다. 특징 순서는 소스의 `FEATURE_NAMES`, 버전은 `candidate_relative_32_v1`로 고정했다.

| 차원 | 특징 |
|---|---|
| 0:12 | 0.5초 간격 후보 XY 속도 − 현재 causal vx/vy |
| 12:22 | 연속 후보 XY 속도 차이 × 2 / 3 |
| 22:24 | causal vx/vy / 20 |
| 24:26 | causal ax/ay / 3 |
| 26:28 | 제공 목표점 XY / 50 |
| 28:30 | (후보 3초 끝점 − 목표점) / 50 |
| 30 | 해당 행의 유효 후보 평균을 뺀 원 점수 |
| 31 | 목표점 있음 1, 없음 0 |

목표점이 없으면 26:30과 31 모두 0이며, invalid 후보 특징 전체도 0이다. Head는 32→128→64→1 ReLU MLP이며 마지막 Linear만 0으로 초기화한다. 첫 업데이트에서는 마지막 층에 gradient가 있고 그 이전 층의 gradient는 0인 것이 정상이다. 마지막 층 가중치가 변한 뒤에는 앞 층도 학습할 수 있다. 기존 점수+잔차의 계수는 1로 고정했다. head 연산과 최종 점수 덧셈은 autocast를 꺼 FP32를 유지한다. head 파라미터도 FP32로 유지해야 한다. RNG의 저장·복원·seed는 호출자 책임이며 생성자가 별도 재설정하지 않는다.

`base_goal_mode="none"`이면 wrapper의 목표점을 base로 전달하지 않는다. `"selection"`이면 이미 목표점 선택 기능을 가진 base에 `goal_xy`를 명시적으로 전달한다. 기본 base는 freeze/eval이며 wrapper를 train 모드로 바꾸어도 base는 eval 상태다. invalid 점수는 기존 sentinel을 보존하고 내부 선택에서 mask하므로 외부 손실/선택 코드도 `candidate_valid`를 사용해야 한다. 현재 100m 은행의 모든 후보는 유효하다.

검증:

- CPU 9개 테스트 통과: 실제 물리 단위·특징 순서, 목표점 누락, 첫 gradient와 후속 hidden gradient, 기존 FP32 필드 보존, 좌표/ID 보존, 기존 후보로 선택 변경, 목표점 전달·freeze 정책, BF16 autocast 내부 FP32, invalid 후보 처리, head state_dict 왕복.
- 저장된 old V256 / dense no-goal / dense goal step500의 동일 tune 8행 × 200후보에 실제 causal 상태와 제공 목표점을 넣어 모든 특징이 유한하고, 초기 잔차가 0이며, 원 FP32 필드가 bitwise 동일함을 확인했다. 이 검증은 저장 출력만 읽었고 모델/image forward 또는 새 학습을 하지 않았다.
- 저장 출력의 D3는 각각 0.38453773 / 0.28243458 / 0.39572152로 변하지 않았다. 이는 초기값 보존 점검이며 전체 tune 결과나 개선 수치가 아니다.

SHA256:

- `relative_selector.py`: `7fa75415a40691f555cca27ab9fe88331725177f19887c01a0e887121e2ac59f`
- `test_relative_selector.py`: `a7e0549c33f224d6b264f5d77bca405b89cd7d0978b5e7ac891194836eed15fa`
- 실제 출력 감사: `reports/sparsedrivev2_20260910/selection_diagnostics/relative_selector_audit.json`

Live `train.py`, `data.py`, `losses.py`, `public_model.py`, `goal_selector.py`는 변경하지 않았다. 성능 판단은 root가 고정 checkpoint와 행 분리를 확인한 뒤 별도 캐시 head 학습 및 전체 tune 비교로 진행한다.
