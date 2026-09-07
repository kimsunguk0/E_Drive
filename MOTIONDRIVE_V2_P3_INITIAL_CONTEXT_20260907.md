# P3 r1 초기 재현 실패와 동결 기준 확정

2026-09-07 B200. 아직 P3 학습 성능 결과가 아니다.

## 실제 종료

source `a4ca60d635fc98c04b84074fe17c4845d2aff8ae`의 두 초기 실험은
2026-09-07 21:46:25 KST 실제 rc1로 종료했다. optimizer step0, nonfinite0,
pressure 없음. completed로 다시 분류하지 않는다.

| 팔 | parent / child PID | 실패 manifest SHA | supervisor SHA |
|---|---|---|---|
| control GPU4 |2055528 /2055612|278e5dc0a0354d630c656e5b6d25eb6467c2f32c0058910b1d67250954f0ea45|2b845cc4f4572ebae34ea29532ce278b73244d0156d9ae8adda8fce653364e4c|
| state GPU5 |2055500 /2055608|68a49a866cfa6f009b005f35e4042088da310ace8ddc29218a4a84837dcc65da|faacc462ae4712fd937a477bdbc74dc061f5a33600db3815cf6578e57f92ba63|

두 팔 기대 D3는0.4454620049779748, 실제는0.4454619773599255로 같았다.
차이−2.76180493e−8 때문에 strict equality gate가 실패했다. 네 PID 부재와
GPU4/5의 compute PID 부재·memory.used0도 별도 조회했다. run/log/checkpoint는 보존한다.

## 분리 검증

1. 최초 train8 B1 검사: 기존/대조/상태 연결24 full forward,7개 출력의 dtype·shape·bytes 동일.
   PID2055030 실제 rc0/부재. [보고서](reports/p3_query_initial_train8_gpu5.json)
   SHA `b5ed19821cfefabcc43d80ce874091574c744c5f5f8e589f0de18270f5d51826`.
   이 검사는 동결 조건 차이 또는 전체 B4 조건까지 검증하지 않았다.
2. B4/B2 tune10 분리 검사: 기존 비동결 모델은 원 P2의 per-frame D3를 exact 재현했다.
   같은 기존 모델의 nonplanner requires_grad를 끄면 motion_features 최대0.001953125,
   state_hat 최대9.54e−6, plan 최대1.34e−5 차이가 관찰됐다. scene/history/occ/lane은
   이 표본에서 동일했다. 따라서 query 변경 없이도 동결 조건만으로 차이가 재현된다.
   PID2062525 실제 rc0/부재. [보고서](reports/p3_query_initial_batch_diagnostic_gpu5.json)
   SHA `f152354fb5203e4f82f1e0a141487943111df9e376f4acfde5978811879db1ab`.
   이는 정확한 내부 커널/캐시 기전을 확정한 profiler 분석은 아니다.
3. 실제 동결 legacy/control/state 전체 tune1998, B4(마지막B2), 총1500 full forward:
   모든7개 출력의 dtype/shape/bytes가 같았다. 양 r1의1998개 초기 scalar 기록도
   scene/session/frame/proxy/D3까지 exact 재현했다. 동결 초기 D3는0.4454619773599255.
   PID2063391 실제 rc0/부재. [보고서](reports/p3_query_frozen_full_tune_gpu5.json)
   SHA `f32680b3f13f5f85045e27120c63db7afa1e24d3c566dc756ce8ac4c7e3158e1`.

마지막 두 진단 스크립트는 a4ca60d 원 runtime을 변경하지 않고 새 파일로 실행했으며,
각 스크립트 SHA를 결과에 별도 기록했다. 이번 후속 commit에서 코드와 결과를 함께 보존한다.

## 정정의 범위

실험 초깃값은 동결 runtime의 공통 값으로 바꾼다. 허용오차는 여전히0이며,
실제 동결 검증 보고서 SHA를 시작·끝에 검사한다. 새 r2 경로만 쓰고 r1은 실패로 남긴다.
가중치/학습량/데이터/seed/손실/1000step 주판정/11세션 bootstrap 기준은 그대로다.
이는 초기 함수 검증이지 성능 개선, CUDA 비결정성 증명, 규정 승인 또는 수상 가능성 증거가 아니다.
