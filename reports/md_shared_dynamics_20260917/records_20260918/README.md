# 2026-09-17/18 실행 기록 색인

실험 설정·학습 곡선·저장 평가 요약은 각 archive 디렉터리에 있다.
원본 prediction과 checkpoint는 서버에 보존하며 experiment_index.json에 경로·크기 및 주요 파일의 SHA256을 기록했다.
final_eval 파일이 있다는 것만으로 원래 계획한 학습 예산을 모두 마쳤다는 뜻은 아니다. 평가 step과 마지막 학습 로그를 따로 기록한다.

| 실행 단계 | 마지막 학습 로그 | 저장된 평가 step | PREFIX |
|---|---:|---:|---:|
| md_progress_residual_20260917/FRONT-S-s1/joint | 2650 | 2616 | 0.193752 |
| md_progress_residual_20260917/FRONT-S-s1/warmup | 1000 | 1000 | 0.190236 |
| md_progress_residual_20260917/MR-NATIVE-FULL-s1 | 24900 | 24931 | 0.097245 |
| md_progress_residual_20260917/OOF-MR-T203-s1 | 20550 | 20554 | 0.226444 |
| md_progress_residual_20260917/SIDE-S-AUX-DN-s1/joint | 2900 | 2616 | 0.194594 |
| md_progress_residual_20260917/SIDE-S-AUX-DN-s1/warmup | 1000 | 1000 | 0.190014 |
| md_progress_residual_20260917/SIDE-S-AUX-s1/joint | 2750 | 2616 | 0.193830 |
| md_progress_residual_20260917/SIDE-S-AUX-s1/warmup | 1000 | 1000 | 0.190256 |
| md_progress_residual_20260917/SIDE-S-s1/joint | 350 | 없음 | 미평가 |
| md_progress_residual_20260917/SIDE-S-s1/warmup | 1000 | 1000 | 0.190270 |
| md_progress_residual_20260917/SIDE-VA-AUX-s1/joint | 1 | 없음 | 미평가 |
| md_progress_residual_20260917/SIDE-VA-AUX-s1/warmup | 450 | 없음 | 미평가 |
| md_progress_residual_20260917/SIDE-VA-s1/joint | 350 | 없음 | 미평가 |
| md_progress_residual_20260917/SIDE-VA-s1/warmup | 1000 | 1000 | 0.190352 |
| md_progress_residual_20260917/STATUS-A2-S-SCENE-s1/warmup | 1000 | 1000 | 0.189591 |
| md_progress_residual_20260917/STATUS-A2-S-SCENE-s2/warmup | 1000 | 1000 | 0.189498 |
| md_progress_residual_20260917/STATUS-A2-S-s1/warmup | 1000 | 1000 | 0.189545 |
| md_progress_residual_20260917/STATUS-A2-S-s2/warmup | 1000 | 1000 | 0.189536 |
| md_shared_dynamics_20260917/A2-DIRECT-s1 | 20550 | 20554 | 0.164281 |
| md_shared_dynamics_20260917/A2-DIRECT-s1-smoke | 2 | 2 | 0.477651 |
| md_shared_dynamics_20260917/A3-DIRECT-s1 | 20550 | 20554 | 0.149289 |
| md_shared_dynamics_20260917/A3-DIRECT-s1-smoke | 2 | 2 | 0.477645 |
| md_shared_dynamics_20260917/A3-FP-VA-s1 | 600 | 없음 | 미평가 |
| md_shared_dynamics_20260917/A3-FP-VA-s1-smoke | 2 | 2 | 0.481559 |

A3-FP-VA의 마지막 학습 로그는 step 600이다. 이전 문서의 350은 중간 관측 시점이었다.
OOF-MR-T203-s1은 20,554 update와 V0 평가를 완료했다. V0 0.226444는 unseen new107의 결과가 아니다.
MR-NATIVE-FULL의 0.097245는 학습에 포함된 V0 행의 진단 수치다.
