# C0-T v3의 no-goal 대조군. loader/파이프라인/pkl/seed/epoch/lr/matcher/queue 전부 동일,
# goal_decoder만 없다 (형님 §11-4의 "동일 조건" 요구).
_base_ = ['./VAD_etri_c0t_overfit_v3.py']

model = dict(pts_bbox_head=dict(goal_decoder=None))
