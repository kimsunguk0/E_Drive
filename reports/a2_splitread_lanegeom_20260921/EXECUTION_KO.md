# CTRL 이후 SplitRead / LaneGeometry 실행

사용자의 2026-09-21 요청으로 GPU0/1/2에서 세 DEV arm을 시작했다. GPU3은 검증·후속 작업을 위해 비워두었다. 본 학습 시작은 **2026-09-21 19:57:32 KST**다.

|GPU|Arm|바꾸는 것|공통 예산|
|---|---|---|---:|
|0|P-CTRL-NEXT|동일 부모의 추가 학습 대조|6,852 update|
|1|P-SPLITREAD|TemporalRead와 비선형 길이/방향 readout 분리|동일|
|2|P-LANE-GEOM|동일 shared scene의 연속 lane offset/axis 보조 감독|동일|

부모는 DEV P-CTRL / step3,426이며 SHA256은 `5021300b0d876a6b59b2137d78fd96ab21b80bfed7141ebc1248e5c8bc05a37a`다. 새 성능은 기존 서버 FULL 0.133684828과 구분한다. FULL 가중치·통계·feature cache를 DEV에 반입하지 않는다.

- 동일 train310 / 83,700행, V0 37scene / 1,998행 / 11session.
- 동일 seed1, effective batch16 / microbatch8, BF16 이미지·FP32 planner/loss, fixed BN, 기존 flip0.5.
- Fresh AdamW, backbone1e-6 / 기존·복사 모듈1e-5, weight decay0.01, clip5, warmup100, 새 6,852-step cosine.
- LaneGeometry 새 head만5e-5. 기존 optimizer의 두 그룹 검증은 유지하고, 세 번째 그룹을 별도로 전수 검증·기록한다. 모든 trainable parameter가 정확히 한 그룹에 포함된다.
- 기존 PREFIX / LEN0.25 / occupancy / lane / motion-state-history 가중치는 그대로다.
- LaneGeometry λ0.05, 첫500 update에서 선형 증가. Offset/axis의 full-effective-batch 유효 성분 분모를 별도로 사용한다.
- LaneGeometry 로그의 기존 `total`은 inherited common+LEN 항이다. 실제 최적화 목적은 `total + lane_geometry_weighted`이며 두 값을 함께 기록한다. Geometry 항이 optimizer에 연결된 것은 실제 새 head/공유 gradient 및5-update 가중치 변경으로 확인했다.
- 평가0 /3,426 /6,852, primary는 terminal. 중간 best와 terminal을 구분하고 동일 고정 train256 probe를 보존한다.
- 계보: 부모는 이전 continuation을 끝낸 CTRL이다. 그 뒤의 새6,852 update이며, 이전 H4 terminal에서 곧바로 시작한 실험이 아니다.

## 착수 검사

초기 전체 V0:

- CTRL-NEXT: 0.1502855043
- SPLITREAD: 0.1502855507
- LANE-GEOM: 0.1502855043

세 모델 모두 같은 부모 출력을 보존했다. SplitRead의 작은 FP32 연산 순서 차이는 허용오차 안이다. 새 graph의 5-update 학습과 새 프로세스 strict reconstruction을 확인했고, 실제 샘플·증강 fingerprint가 세 arm 모두 같았다. 본 학습은 smoke 가중치/optimizer를 이어받지 않고 같은 DEV 부모에서 다시 시작했다.

LaneGeometry 학습용 target은 train310 scene /83,700행 전체를 사전 생성했다. 실제 원본 map·현재 pose·기존 lane_valid만 사용하며 미래 ego 경로로 선을 선택하지 않는다. 캐시는 float32 offset/axis, bool mask, row index와 scene별 원본/산출물 hash를 보존한다. 생성시간은 약8분이었으며 학습 step에서는 캐시를 읽는다. 좌우 flip은 기존 모든 영상·status·기하와 함께 적용된다.

Geometry head는 학습에만 실행된다. 실제5-update checkpoint에서 head를 제거한 student를 새 프로세스에 strict load하여 동일 계획 출력을 확인했다. Geometry target과 head 출력을 planner에 넣지 않는다. 기존 A2 query-only status 경계를 유지한다.

## 시간과 추가 학습 판단

20:00 KST 실측 약350 update에서 세 arm 모두 약0.53초/update다. 첫 평가20:30 전후, terminal21:00~21:10을 예상한다. 평가·다음 epoch 로딩 등에 따라 달라질 수 있다. 최신 수치는 `launch_health.json` 및 runtime 로그를 따른다.

사용자는 추가 update 판단을 위임했다. 우선 동일6,852 horizon을 완료해 비교 조건을 지킨다. 중간에 한 arm만 scheduler를 늘리지 않는다. 이후 terminal·중간 추세·train/V0 간 차이·일반 주행 손익을 보고, 추가 학습이 타당하면 유망 arm과 CONTROL에 동일한 별도 continuation을 적용한다. 현재는 추가 stage를 자동 예약하지 않았다. λ/head/해상도 스윕이나 새 구조 결합도 자동화하지 않는다.

새 DEV가 좋아질 경우 대응 FULL 부모는 완료된 CTRL FULL stage2이고, 같은 추가 노출량은8,311 update다. 현재 실행기는 새 FULL 학습이나 대회 업로드를 자동으로 시작하지 않는다. 이번 DEV의 실제 결과를 먼저 판단한다.

## 기록

- `runtime/orchestrator.json`: 실제 PID / GPU / 시작·종료 상태.
- `smoke_and_reload.json`: 본 학습 이전 검사와 source hash.
- `protocol_*.json`: 부모·graph·데이터·학습 계약.
- `geometry_cache.json`, `geometry_cache_checks.json`: train-only target 생성·flip 확인.
- `results.csv`, `result_step*.json`, `RESULTS_KO.md`: 실제 평가와 동일 control/parent 대비 통계.
- `work_dirs/a2_splitread_lanegeom_20260921/*/metrics.jsonl`: update별 학습값·row stream.
- `optimizer_groups.json`, `gradient_audit.json`, `geometry_gradient_contributions.json`: 실제 그룹과 gradient.

새 모델의 배포 비용은 실제 student graph로 별도 측정한다. 기존 FULL의 시간이나 서버 harness elapsed_ms를 새 모델의 실측 시간으로 승계하지 않는다. 본 학습 시작과5-update 검사는 성능 개선 결과가 아니다.

## RTX4090 실제 student 비용

5-update smoke의 student export를 같은 Docker/2개 raw fixture에서 측정했다. 전체B1 BF16 forward의 clip별 median은 CTRL25.38~25.47ms, SplitRead25.66~25.67ms, LaneGeometry student25.53~25.54ms다. 각각 warmup30/repeat200이며 전처리·파일읽기·전송은 제외한다. FLOPs는 CTRL/Lane730.044861G, SplitRead730.098142G다. 최종 학습 가중치의 배포 검사는 별도지만, 새 graph의 비용은 확인했다.

첫 Docker --gpus 요청은 호스트의 오래된 EGL library 참조로 시작하지 못했다. 기존 --runtime=nvidia와 compute/utility capabilities로 실행했고 호스트 driver/config를 수정하지 않았다. 상세 시도와 결과는 RTX4090_attempts.json, RTX4090_cost.json에 보존한다.
