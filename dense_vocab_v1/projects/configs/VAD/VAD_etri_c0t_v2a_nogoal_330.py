# C0-T v2a **no-goal 대조군** — 5초 goal의 순수 성능 이득을 재는 유일한 비교항.
#
# 왜 no-goal full330(0.4345)과 직접 비교해서는 안 되는가 (형님 지시)
# ----------------------------------------------------------------
# 그 비교에서는 세 가지가 동시에 달라진다.
#     goal 유무 / planner 구조 / content-only 제약
# 그래서 차이가 나와도 goal 덕인지 구조 덕인지 알 수 없다.
# 여기서는 **goal이 나르는 정보량만** 0으로 만든다.
#
#     v2a goal    Q = time query + command + goal   K = geometry/position   V = image content
#     v2a no-goal Q = time query + command          K = geometry/position   V = image content
#
# 구현: `disable_goal=True`가 `goal_features(goal_xy)`를 `zeros_like`로 바꾼다.
# 입력 차원을 줄이지 않는다 -- 줄이면 `cond_mlp.0`의 shape가 달라져 같은 seed로도
# 초기값이 갈리고 "goal만 다르다"가 성립하지 않는다. 이 판은 파라미터 shape와
# 초기값이 goal 판과 **완전히 동일**하다.
#
# 그 외 전부 동일:
#     같은 epoch3 visual trunk        /tmp/pm97/ckpt/etri_tvad_visual_trunk_v1_ep3.pth
#     같은 새-head 초기화 seed         --seed 0 --deterministic
#     같은 frame>=30 dataset           89,100 앵커
#     같은 queue/cadence               queue 7 / 고정 0.5초 / filter_empty_gt=False
#     같은 optimizer/LR 그룹/epoch     AdamW 2e-4, cosine, 4 epoch
#
# 발사 시점: v2a goal이 0.5 epoch 영상 게이트를 통과한 **뒤**. 통과 못 하면 이 판도
# 의미가 없다 (planner가 영상 content를 못 읽는데 goal 기여를 재봐야 무의미).
_base_ = ['./VAD_etri_c0t_v2a_full330.py']

model = dict(pts_bbox_head=dict(goal_decoder=dict(disable_goal=True)))
