# v2a 크래시 원인 분리 대조군 C1 — **content/goal 경로만** 제거한다.
#
# v2a_full330에서 정확히 하나만 뺀다: `expose_image_evidence` + `goal_decoder`.
# LR 그룹, frame>=30 앵커(89,100), queue/cadence, seed, epoch, trunk 초기값은 동일.
# (LR custom_keys의 `goal_decoder` / `content_value_proj` 항목은 매칭되는 파라미터가
#  없으므로 무시된다 -- mmcv는 없는 키를 조용히 건너뛴다.)
#
# 죽으면    원인은 LR 그룹(2) 또는 frame>=30 샘플 순서(3)
# 살아남으면 원인은 content value 경로 / 새 head의 gradient(1)(4)
#
# 주의: 이 판은 planning head가 없다 (`goal_decoder=None`이면 기존 `ego_fut_decoder`가
# 쓰이는데 trunk ckpt에서 그 6개 키를 제거했으므로 random init이다). 성능을 보는
# 판이 아니라 **크래시 재현 여부만** 보는 진단용이다.
_base_ = ['./VAD_etri_c0t_v2a_full330.py']

model = dict(
    expose_image_evidence=False,
    pts_bbox_head=dict(
        goal_decoder=None,
        transformer=dict(expose_image_evidence=False)))
