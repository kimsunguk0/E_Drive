# C0-T v2(부채 청산판)의 **no-goal 대조군**. 형님 §11-4.
#
# VAD_etri_c0t_overfit_v2.py 와 loader/파이프라인/pkl/seed/epoch/lr/matcher가 전부 같고
# goal_decoder만 없다. 그래야 goal head 채택 기준을 같은 조건에서 비교할 수 있다.
_base_ = ['./VAD_etri_c0t_overfit_v2.py']

model = dict(pts_bbox_head=dict(goal_decoder=None))
