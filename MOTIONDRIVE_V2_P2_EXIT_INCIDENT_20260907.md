# P2 종료 사고와 분리 진단 — 2026-09-07 20:49 KST

## 원 라운드 판정

C1/T0는 LAST3000과 마지막 tune 평가를 저장한 뒤 native signal11 문구를 남겼다.
2026-09-07 20:46:58 KST에 실제 child rc=-11/SIGSEGV, supervisor139로 자연 종료했다.
다른 세 팔은 child/supervisor0으로 종료했다. 네 팔의 own child/supervisor8개 PID는
모두 부재이며 기존 다른 작업은 생존했다. 수동 신호·OOM pressure event는 없었다.

**기존 P2 all-four clean-exit gate는 실패다.** 원 기록·checkpoint·학습 조건·통과
기준을 수정하지 않는다. 네 팔 정상 완주 또는 통과한 4조건 요인실험으로 발표하지 않는다.
저장된 C1/T0 수치는 로그상의 결과이며, 정상 종료한 run의 검증된 결과와 구분한다.
native 오류의 정확한 원인은 아직 모른다. 시점상 학습/평가 후 반환·정리 단계로
보이지만 특정 라이브러리 또는 손상 원인을 backtrace 없이 확정하지 않는다.

## 목표를 계속 진행하기 위한 별도 진단

정상 종료한 C0/T0, C0/T1, C1/T1만 **종료 사고 후의 분리 진단**으로 평가한다.
이는 원4판 결과의 성공 처리 또는 실패한 C1/T0의 우회 admission이 아니다.
엄격 `analyze_motiondrive_v2_p2.py`의 all4 gate는 그대로 둔다.

- 최초 준비 SOURCE는 `5cea6eda3fae750aa60129596903ad6275b0a10b`다.
  GPU 실행 전 사용자 GPU4·5 추가 허용에 맞춘 wrapper 변경을 별도 커밋하고,
  세 평가 모두 그 동일한 clean commit을 명시한다. 실제 SHA는 execution 기록에 남긴다.
  원 모델/학습/평가 소스와 원 학습 commit `c6845fb`는 변경하지 않는다.
- 기존 shared evaluator wrapper는 각 팔의 실제 rc0·PID부재·LAST3000·원본 SHA를
  그대로 요구한다. C1/T0는 실행하지 않는다. 정상 세 팔의 배치4/bf16/tune1998/
  normal·image_shuffle·repeat_current·reverse_history/동일-forward motion 기록은 같다.
- 모든 학습 프로세스의 종료를 먼저 확인한다. 후속 사용자 지시로 허용된
  유휴 GPU4·5에서 C1/T1과 C0/T1을 먼저 병렬 평가한 뒤, 정상 종료한 장치를
  재확인하여 C0/T0에 재사용한다. 원 arm의 학습 GPU 계보와 평가 물리 GPU를 구분한다.
  기존 cap12000MiB/reserve8192MiB/입장20192MiB 정책을 유지한다.
- 산출물은 `p2_<arm>_last3000_motion_diagnostic.json`과 별도 execution 기록이다.
  원 학습 manifest/log/last/best 및 기존 평가 파일은 덮어쓰지 않는다.
- C1/T1의 운동 상태 정확도·시간별 궤적 오류·영상 교란 반응으로 다음 설계를 정한다.
  세 팔의 사후 진단에서 원4판의 상호작용이나 정상완주 CI를 만들어 내지 않는다.

실패한 C1/T0 checkpoint의 CPU 무결성/finite 검사는 별도 수행한다. 그것이 통과해도
OS 실패를 소급해서 정상으로 바꾸거나 실제 GPU 재평가를 했다고 말하지 않는다.
최종 val/test·공식 제출·GPU6–7에는 접근하지 않는다. GPU4·5의 사용은
사용자의 "비어있는 idle인 gpu 써도 돼 4,5번 써라"라는 명시적 추가 허용에 따른다.

## CPU 저장 무결성 감사 결과

`reports/p2_c1t0_teardown_checkpoint_audit.json`의 SHA는
`90ee08e117abc454c2b61776b851e7cc17f97163f84517a8bc4f5dd6cd65be43`다.
LAST3000/BEST2250/INITIAL/P1 초기화의 ZIP CRC·CPU load·전체 tensor finite가
통과했다. 원본41개 SHA/stat과 runtime19개 소스가 검사 전후 불변이었고,
초기 모델 tensor 및 네 팔의301개 sample-order 기록도 일치했다.
이는 저장 무결성 검사이며 새로운 forward·정확도 재평가가 아니다.
actual rc-11/SIGSEGV/supervisor139와 원4팔 gate FAIL은 그대로 남는다.
