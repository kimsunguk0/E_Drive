# FRESH 낮은 LR continuation — 2026-09-20

사용자가 잔여 오차 진단 후 `1 고고`로 승인한 추가 학습 한 번이다.

- 부모: DEV `A2-FRESH-NUIM-s1/ckpt_step20554.pth`, SHA `f5ed023c23733fcb83ba0c85fe9b5e3c8f97a91361f8e28b98ec9da2c2e9dfce`.
- 구조·A2 query-only 입력·nominal producer·PREFIX/LEN/보조 loss·fixed BN·BF16·flip 0.5 유지.
- 가중치 전체 strict load, fresh AdamW. 기존 optimizer와 데이터 cursor를 resume하지 않는다.
- 추가 6,852 update, batch16/micro8, backbone LR1e-6/head LR1e-5, warmup100, 새 cosine schedule, seed1 epoch0.
- DEV train83,700/V0 1,998 유지. FULL 계보 사용 없음. Total joint updates 27,406.
- 평가·저장: 1,142/2,284/3,426/4,568/5,710/6,852. Step0은 동일 부모 저장 예측을 재사용한다.
- 주 판정: terminal−parent. 예정 중간점 중 best 선택은 같은 V0 재사용으로 별도 기록한다.
- 전체·일반 주행·정지/출발·첫2초·종/횡·인지/state/history와 session CI를 보존한다.

`launch.py --smoke --gpu 0`로 2-update 실행 검증 후 `smoke_summary.json`을 기록한다. `launch.py --gpu 0`은 GPU 유휴와 기존 run/receipt 부재를 확인하고 본 학습 하나만 실행한다. 실제 update와 초기 SHA를 확인한 뒤 CPU `collect.py --publish`를 별도 실행한다.

Collector는 최대8시간 동안 예정 평가를 수집하고 terminal 결과만 Git에 기록·미러 push한다. 학습을 재시작하거나 다음 arm/FULL/제출을 시작하지 않는다. Git에 동시 변경이 있으면 결과를 보존하고 publish를 중단한다. Runtime log/status와 대형 checkpoint/예측은 기존 정책대로 Git에서 제외한다.
