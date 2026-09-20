# FRESH continuation 실행 기록 — 2026-09-20

사용자 `1 고고`에 따라 제안 1번만 실행했다. GPU0에서 16:48 KST 본 학습을 시작했다.

- 이름: A2-FRESH-CONT-s1.
- 시작 가중치: FRESH DEV terminal 20,554 update, PREFIX 0.158707259. Smoke의 2-update 가중치는 본 학습에 사용하지 않았다.
- 추가 예산: 6,852 update. Fresh AdamW, backbone1e-6/head1e-5, warmup100, 새 cosine, batch16/micro8, seed1의 새 epoch0 stream.
- 기존 graph, A2 query-only 입력, nominal status, PREFIX/LEN0.25/aux0.2, uncertainty, fixed BN, BF16, flip0.5 유지.
- 데이터: 기존 DEV train83,700/V0 1,998. FULL 계보 없음. 부모 checkpoint 불변.
- 2-update smoke에서 전체 strict load, 모델 parameter 변화, optimizer step2, loss 재구성, 학습/평가 행 수, finite 수치를 확인했다.
- 본 실행의 실제 update·초기 state SHA·row SHA를 확인했다. 구체 시점과 처리량은 launch_health.json.
- 1,142 update마다 전체 PREFIX·일반 주행·정지/출발·첫2초·종/횡·인지/state/history를 보존한다. Primary는 6,852 terminal−parent다.
- 기존 부모와 모든 예정 checkpoint를 보존한다. Best-on-V0는 terminal과 별도로 표시한다.
- CPU collector가 중간 결과를 저장하고 terminal 결과를 GitHub 미러에 기록한다. 충돌/실패 시 결과를 보존하고 상태를 기록한다. 다음 학습·FULL·제출은 자동 실행하지 않는다.

진행/완료 상태는 results.json 및 RESULTS_KO.md, 프로세스 상태는 runtime/watcher_status.json에 있다. 이 실행 기록의 시작 시점 수치는 완료 성능을 뜻하지 않는다.
