# C0-T의 **no-goal 대조군**. 형님 §11-4용.
#
# VAD_etri_c0t_overfit.py 와 **loader/파이프라인/pkl/seed/epoch/lr이 전부 같고**
# goal_decoder만 없다. 그래야 "goal head를 채택할 최소 기준"을 같은 조건에서 비교할 수
# 있다. 기존 tvad_metric_overfit epoch 8 (0.8537)은 대조군으로 쓸 수 없다 --
# temporal_shuffle=True, filter_empty_gt=True 로 loader가 다르다.
#
# goal_decoder=None 이면 head는 원본 VAD 경로(ego_query -> ego_agent/map attn ->
# ego_fut_decoder MLP)를 쓴다. ego_fut_goal 은 collect 되지만 head에서 무시된다.
_base_ = ['./VAD_etri_c0t_overfit.py']

model = dict(pts_bbox_head=dict(goal_decoder=None))
