# v2a 크래시 원인 분리 대조군 C2 — **LR 그룹만** 원래대로 되돌린다.
#
# v2a 330 pilot이 iter ~600에서 죽었다.
#     hungarian_assigner_3d.py:123  ValueError: matrix contains invalid numeric entries
# GT 쪽 log(0)은 배제됐다 (치수<=0 박스 0개, w/l/h 최소 0.488 m).
# no-goal full330은 같은 pkl의 상위집합(99,000 앵커)으로 4 epoch를 완주했으므로
# 데이터 단독 원인도 아니다.
#
# v2a와 no-goal의 차이는 넷이다.
#     (1) content value 경로 + goal decoder
#     (2) LR 그룹 (head 2e-5 / backbone-neck 0)   <- 이 판이 되돌린다
#     (3) frame>=30 앵커 (89,100) -> 샘플 순서
#     (4) 새 head의 큰 gradient (grad_norm 67~80, clip 35에 계속 걸림)
#
# 이 config는 (2)만 no-goal과 같게 만든다: head 2e-4 / backbone 2e-5, freeze 없음.
# 여전히 죽으면 원인은 (1)/(3)/(4)이고, 살아남으면 LR 그룹 설계가 원인이다.
_base_ = ['./VAD_etri_c0t_v2a_full330.py']

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    # `_delete_=True` 없으면 부모의 custom_keys와 **머지**된다 (pts_bbox_head 0.1,
    # img_neck 0.0 이 남아서 이 대조군의 목적이 사라진다). 실측으로 확인했다.
    paramwise_cfg=dict(custom_keys={'_delete_': True,
                                    'img_backbone': dict(lr_mult=0.1)}),
    weight_decay=0.01)
