"""영상 전용 궤적 사전 스코어러 (규정 준수 계보).

왜 이 파일이 따로 있는가
------------------------
8/31 운영국 답변이 판정 기준을 좁혔다.

    "과거 정보를 직접적으로 planner 입력으로 또는 단순 임베딩 형태로 사용하는 것을
     금지하며, 여러 task의 공통 특징을 향상하는 등의 간접적 활용은 허용합니다."
    (target point 후속) "미래도 동일합니다. 다만 ... 모델의 여러 출력 중
     선택에만 활용되는 경우는 허용됩니다."

즉 goal이 planner에 입력되면 형태와 무관하게 금지이고, 허용되는 유일한 자리는
**모델이 낸 여러 출력 중 하나를 고르는 것**이다.

`goal_decoder.py`의 `GoalWaypointDecoder`(v2a)와 `DenseVocabularyDecoder`는 둘 다
goal·cmd를 attention query에 넣는다 -- 후자는 후보 좌표가 고정이라 v2a보다 안전하지만
여전히 goal이 학습된 스코어링에 참여한다. 이 파일의 디코더는 **goal/cmd/ego status를
아예 받지 않는다.** 선택은 모델 밖의 고정 규칙(`scripts/etri_vocab_select.py`)이 한다.

베이스라인과 같은 패턴
----------------------
배포 VAD도 cmd를 planner 입력으로 쓰지 않는다. 3개 mode 출력을 낸 뒤
`ego_fut_preds[ego_fut_cmd==1]`로 **고르기만** 한다 (VAD_head L1264/1274).
여기서는 mode 3개 대신 사전 top-M을 내고 goal이 고른다. 구조가 동형이다.

영상 0 → 정지 궤적 (구조적 보증)
--------------------------------
value 경로에 bias/affine이 없다:
    LayerNorm(elementwise_affine=False) -> MultiheadAttention(bias=False)
    -> Linear(bias=False) -> anchor logit 내적
따라서 image_evidence == 0 이면 logits가 전부 정확히 0으로 동률이 되고,
`argmax`가 index 0을 반환한다. 사전 index 0은 **정확한 정지 궤적**으로 강제된다
(`scripts/etri_dense_vocabulary.py`가 kmeans 앞에 zeros를 붙인다).
goal이 무엇이든 이 결과를 바꿀 수 없다 -- 선택이 모델 밖에 있고 후보가 하나뿐이다.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHALLENGE_W = (11.0, 11.0, 5.0, 5.0, 2.0, 2.0)


class ImageOnlyVocabularyDecoder(nn.Module):
    """고정 train-only 사전에 **영상 증거만으로** 점수를 매긴다.

    forward 는 goal·command·ego status 를 인자로 받지 않는다. 받지 않는다는 사실
    자체가 규정 준수의 증거이므로 시그니처를 바꾸지 말 것.

    Args:
        anchors_path: `[K, fut_ts, 2]` **누적** 좌표. index 0 = 정지.
        top_m: 추론 시 노출할 후보 수. 선택은 밖에서 한다.
        target_temperature: metric 거리 기반 soft target의 온도. 작을수록
            argmin 후보에 질량이 몰린다. K가 크면 인접 후보가 사실상 동등하므로
            hard label(교차엔트로피)은 학습을 망친다.
    """

    requires_image_evidence = True
    uses_condition = False          # ★ goal/cmd 를 받지 않는다. head 가 이걸 본다.

    def __init__(self, anchors_path, embed_dims=256, fut_ts=6, n_head=8,
                 loss_weight=1.0, target_temperature=0.1, top_m=6):
        super().__init__()
        if not os.path.isfile(anchors_path):
            raise FileNotFoundError(anchors_path)
        anchors_abs = np.load(anchors_path).astype(np.float32)
        if anchors_abs.ndim != 3 or anchors_abs.shape[1:] != (fut_ts, 2):
            raise ValueError(
                f'anchors must be [K,{fut_ts},2], got {anchors_abs.shape}')
        if not np.array_equal(anchors_abs[0], np.zeros((fut_ts, 2), np.float32)):
            raise ValueError('anchor index 0 must be the exact stopped trajectory')

        anchors_inc = np.diff(
            np.concatenate([np.zeros_like(anchors_abs[:, :1]), anchors_abs],
                           axis=1), axis=1)
        self.register_buffer('anchors_abs', torch.from_numpy(anchors_abs))
        self.register_buffer('anchors_inc', torch.from_numpy(anchors_inc))
        self.register_buffer(
            'challenge_w',
            torch.tensor(CHALLENGE_W, dtype=torch.float32) / sum(CHALLENGE_W))

        self.fut_ts = fut_ts
        self.num_anchors = anchors_abs.shape[0]
        self.top_m = int(top_m)
        self.loss_weight = float(loss_weight)
        if target_temperature <= 0:
            raise ValueError('target_temperature must be positive')
        self.target_temperature = float(target_temperature)

        # 조건 없는 시간 쿼리. goal/cmd 가 들어올 자리가 없다.
        self.ts_embed = nn.Embedding(fut_ts, embed_dims)
        self.value_norm = nn.LayerNorm(embed_dims, elementwise_affine=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dims, n_head, dropout=0.0, bias=False)
        self.scene_proj = nn.Linear(fut_ts * embed_dims, embed_dims, bias=False)
        self.anchor_encoder = nn.Sequential(
            nn.Linear(fut_ts * 2, embed_dims, bias=False),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims, bias=False))
        self.logit_scale = embed_dims ** -0.5
        self._init()

    def _init(self):
        nn.init.normal_(self.ts_embed.weight, std=0.02)
        nn.init.xavier_uniform_(self.scene_proj.weight)
        for m in self.anchor_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def _sequence_first(t, batch_size, name):
        if t.dim() != 3:
            raise ValueError(f'{name} must be 3D, got {tuple(t.shape)}')
        return t.permute(1, 0, 2) if t.shape[0] == batch_size else t

    def forward(self, bev_key, image_value, batch_size, bev_pos=None):
        """Returns (selected_inc, logits, selected_index).

        `selected_inc` 는 top-1 증분 궤적이다. 최종 제출용 선택은 밖에서
        `logits` 의 top-M 을 받아 고정 규칙으로 한다.

        `batch_size` 를 **head 가 명시적으로 넘긴다.** goal 을 없애면서
        `goal_xy.shape[0]` 이라는 batch 출처가 사라졌고, `bev_embed` 는
        sequence-first `[num_bev_query, B, D]` (num_bev_query=10000) 이라
        shape 로 batch 축을 추측하면 조용히 어긋난다 (실측: B=10000 으로 잡혀
        plan_metric_loss 에서 reshape 실패).
        """
        key = self._sequence_first(bev_key, batch_size, 'bev_key')
        value = self._sequence_first(image_value, batch_size, 'image_value')
        if key.shape != value.shape:
            raise ValueError(f'key/value mismatch {tuple(key.shape)} '
                             f'vs {tuple(value.shape)}')
        if bev_pos is not None:
            key = key + bev_pos
        if key.shape[1] != batch_size:
            raise ValueError(
                f'batch 축이 어긋났다: key {tuple(key.shape)}, B={batch_size}')

        query = self.ts_embed.weight.unsqueeze(1).expand(-1, batch_size, -1)
        attended, _ = self.cross_attn(
            query=query, key=key, value=self.value_norm(value),
            need_weights=False)
        scene = self.scene_proj(
            attended.permute(1, 0, 2).reshape(batch_size, -1))
        candidate = self.anchor_encoder(
            self.anchors_abs.reshape(self.num_anchors, -1))
        logits = torch.matmul(scene, candidate.t()) * self.logit_scale
        index = logits.argmax(dim=-1)
        return self.anchors_inc.index_select(0, index), logits, index

    # ------------------------------------------------------------------ loss
    def _metric_distance(self, ego_fut_gt, ego_fut_masks):
        """[B, K] 챌린지 가중 누적 L2. 채점 함수와 같은 정의."""
        gt_abs = ego_fut_gt.cumsum(dim=-2)
        d = torch.linalg.vector_norm(
            self.anchors_abs.unsqueeze(0) - gt_abs.unsqueeze(1), dim=-1)
        valid = ego_fut_masks.to(d.dtype)
        return (d * self.challenge_w.view(1, 1, -1)
                * valid.unsqueeze(1)).sum(-1)

    def vocabulary_loss(self, logits, ego_fut_gt, ego_fut_masks):
        weighted = self._metric_distance(ego_fut_gt, ego_fut_masks)
        target = weighted.argmin(dim=-1)
        target_prob = F.softmax(-weighted / self.target_temperature, dim=-1)
        loss = -(target_prob * F.log_softmax(logits, dim=-1)).sum(-1).mean()
        selected = logits.argmax(dim=-1)
        m = min(self.top_m, weighted.shape[-1])
        topm = logits.topk(m, dim=-1).indices
        best_in_topm = weighted.gather(1, topm).min(dim=-1).values
        return {
            'loss_plan_vocab': loss * self.loss_weight,
            'plan_vocab_oracle_l2': weighted.gather(
                1, target[:, None]).mean().detach(),
            'plan_vocab_selected_l2': weighted.gather(
                1, selected[:, None]).mean().detach(),
            f'plan_vocab_top{m}_l2': best_in_topm.mean().detach(),
            'plan_vocab_top1': selected.eq(target).float().mean().detach(),
        }
