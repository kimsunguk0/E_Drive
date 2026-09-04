# v2a 안정화판 — 구조 게이트를 하나도 약화시키지 않는 두 가지 변경.
#
# 첫 발사가 iter ~600에서 죽었다 (det assigner cost에 non-finite).
# grad_norm이 no-goal 27~31 대비 67~81로 계속 clip(35)에 걸려 있었다.
# 재현이 비트 단위로 안 되므로(deformable attention backward의 atomicAdd)
# 특정 샘플이 아니라 한계적 불안정으로 취급한다.
#
# 1) query_norm=True
#    현재 `query = ts_embed + cond_mlp(...)`에 정규화가 전혀 없다. lr 2e-4로
#    자라면 attention logit이 포화한다. Q에만 걸리므로 `V=0 => Z=0` 불변식과
#    무관하다 -- 출력은 여전히 value의 결합뿐이고 Q는 attention weight만 만든다.
#
# 2) detach_content_addressing=True
#    content 분기가 `sampling_offsets`/`attention_weights`를 main value와
#    **공유**한다. 그래서 planning loss의 gradient가 perception trunk의 sampling
#    주소를 흔든다. v2a에만 있는 경로다. detach하면 의미가 오히려 명확해진다:
#    evidence는 "trunk가 고른 위치에서 읽은 영상 content"이고, goal의 주소 지정력은
#    goal decoder 자신의 cross-attention(Q=goal, K=bev_embed)에 온전히 남는다.
#
# 두 변경 모두 T6/T6'/T6''/shift/T6''' 를 재측정해서 exact 0을 확인한 뒤 학습한다.
_base_ = ['./VAD_etri_c0t_v2a_full330.py']

model = dict(
    pts_bbox_head=dict(
        goal_decoder=dict(query_norm=True),
        transformer=dict(detach_content_addressing=True)))
