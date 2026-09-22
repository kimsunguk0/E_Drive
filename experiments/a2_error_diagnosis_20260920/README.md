# A2 잔여 오차 진단 — 2026-09-20

새 학습 없이 QREFINE/FRESH terminal과 고정 1:1 출력 평균의 잔여 오차를 조사한다.

- `frozen_probe.py`: GPU 0/1에서 각각 QREFINE/FRESH. 동일 V0 1,998행과 오류를 보지 않고 seed 20260920으로 균등 추출한 train 1,024행. 증강 없음.
- 같은 영상 forward의 scene/motion/state/history를 재사용해 ON, state 숫자 0, history 숫자 0, 둘 다 0, continuous motion 0을 비교한다. 외부 provided status/pose/goal, 모델 parameter/buffer, 원래 flag는 바꾸지 않는다.
- `analyze.py`: 저장된 예측, FRESH의 6개 평가 시점, 위 probe를 분석한다. GT는 metric/진단 그룹/기하 구성요소 교체에만 사용한다.

```bash
CUDA_VISIBLE_DEVICES=0 /home/<B200-USER>/cv2env/bin/python experiments/a2_error_diagnosis_20260920/frozen_probe.py --arm QREFINE --gpu 0
CUDA_VISIBLE_DEVICES=1 /home/<B200-USER>/cv2env/bin/python experiments/a2_error_diagnosis_20260920/frozen_probe.py --arm FRESH --gpu 1
/home/<B200-USER>/cv2env/bin/python experiments/a2_error_diagnosis_20260920/analyze.py
```

같은 산출물이 있으면 중단한다. 위 두 GPU 평가는 독립적으로 병행할 수 있고, 분석은 둘 다 완료한 뒤 실행한다. 재현 시 기존 결과를 덮어쓰지 않고 별도 출력 디렉터리를 정한다.

OFF는 학습 분포 밖 개입이다. 숫자 0 뒤에도 state MLP bias와 motion type embedding은 남는다. 제거 학습의 효과나 정보 경로의 불필요성을 증명하지 않는다. 정지 logit은 **현재 정지**를 나타내며 미래 출발 여부가 아니다. GT 길이/방향 교체는 배포 예측이나 PREFIX 최적 oracle이 아니다.

결과: `reports/a2_error_diagnosis_20260920/RESULTS_KO.md`, `analysis.json`, `probe_QREFINE.json`, `probe_FRESH.json`. 대형 배열은 기존 정책대로 Git에서 제외하고 경로와 SHA를 기록한다. 새 train/FULL/제출은 이 진단에서 실행하지 않는다.
