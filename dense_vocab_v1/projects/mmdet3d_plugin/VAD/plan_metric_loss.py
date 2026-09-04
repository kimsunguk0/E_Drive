"""챌린지 metric과 **동일한** 미분가능 planning loss.

왜 필요한가 (전부 8-scene overfit 실측)
--------------------------------------
베이스라인 `loss_plan_reg`는 세 곳에서 채점 함수와 어긋난다.

1. **증분에 걸린다.** 채점은 누적 waypoint다. 종방향 증분 오차의 부호가 6스텝 모두
   같은 앵커가 **75.6%**이고 `3초 누적종오차 / 증분종오차합`의 p50이 정확히
   **1.000**이었다 -- 증분 오차가 완전히 같은 방향으로 누적된다. 그래서 증분 L1이
   0.62일 때 누적 챌린지 L2는 2.77까지 벌어진다(증분에 같은 식 적용 시 1.13).
   증분 loss는 이 상관을 보지 않으므로 "매 스텝 조금씩 같은 쪽으로 틀리는" 속도
   편향을 고칠 유인이 없다.

2. **3개 mode로 희석된다.** `loss_plan_l1_weight`가 GT command mode만 1인데,
   mmdet `L1Loss`의 `reduction='mean'`은 분모로 `B*3*T*2` **전체**를 쓴다.
   실측: 같은 입력에서 `L1Loss(weight=cmd마스크)` 0.306678 vs 선택 mode만의 평균
   0.920033 -> 비율 **정확히 3.0000**.

3. **성분별 L1이다.** 채점은 waypoint별 **Euclidean** 거리다.
   `|dx| + |dy|` != `sqrt(dx^2 + dy^2)`. 지금은 오차의 95%가 종방향이라 둘이 비슷하게
   움직이지만, 회전·횡오차가 다시 중요해지면 최적점이 갈린다.

그래서 이 loss는
    GT command mode를 **먼저 slice** -> cumsum -> waypoint별 **Euclidean** ->
    챌린지 waypoint 가중 [11,11,5,5,2,2]/36 -> 배치 평균
을 한다. 로그값 자체가 미터 단위 챌린지 L2와 같은 의미를 갖는다.

가중치 유도: 챌린지 L2 = mean(ADE@1s, ADE@2s, ADE@3s)를 시점별로 전개하면
    (1/2+1/4+1/6)/3 = 11/36,  (1/4+1/6)/3 = 5/36,  (1/6)/3 = 2/36
이다. `src/challenge_metrics.py`가 이 정의를 회귀 테스트로 지킨다.
"""
import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES

# 챌린지 waypoint 가중. src/challenge_metrics.py 의 waypoint_weights() 와 동일.
CHALLENGE_W = (11.0, 11.0, 5.0, 5.0, 2.0, 2.0)


@LOSSES.register_module()
class PlanChallengeL2Loss(nn.Module):
    """누적 Euclidean + 챌린지 waypoint 가중.

    Args:
        loss_weight (float): 최종 스칼라 배수.
        weights (tuple | None): waypoint 가중. None이면 균등(=ADE@3s 단독).
        normalize (bool): True면 가중을 합 1로 정규화(기본). False면 원시 [11,11,...].
        eps (float): sqrt의 gradient가 0에서 발산하는 것을 막는다. 예측이 GT와
            정확히 일치하는 waypoint에서 NaN이 나오는 사고를 방지한다.
    """

    def __init__(self, loss_weight=1.0, weights=CHALLENGE_W, normalize=True,
                 eps=1e-6):
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize
        self.eps = eps
        w = None if weights is None else torch.tensor(weights, dtype=torch.float32)
        if w is not None and normalize:
            w = w / w.sum()
        self.register_buffer('w', w if w is not None else torch.empty(0))

    def forward(self, pred_delta, gt_delta, cmd_onehot, masks=None):
        """
        Args:
            pred_delta (Tensor): [B, M, T, 2] **증분** 예측 (모델 원출력).
            gt_delta (Tensor):   [B, T, 2] 또는 [B, M, T, 2] **증분** GT.
                pkl의 `gt_ego_fut_trajs`가 증분이다 (`np.diff`). 검증:
                `ego_cache.npz['fut'] == cumsum(gt_ego_fut_trajs)` 최대차 7.6e-06.
            cmd_onehot (Tensor): [B, M] GT command one-hot.
            masks (Tensor | None): [B, T] 유효 waypoint 마스크.
        """
        B, M = pred_delta.shape[0], pred_delta.shape[1]
        b = torch.arange(B, device=pred_delta.device)
        ci = cmd_onehot.reshape(B, M).argmax(dim=-1)
        # ★ 먼저 slice한다. 마스크로 감싸고 전체를 평균하면 M배 희석된다(실측 3.0000).
        p = pred_delta[b, ci]                                   # [B, T, 2]
        g = gt_delta[b, ci] if gt_delta.dim() == 4 else gt_delta

        p_xy = torch.cumsum(p, dim=1)
        g_xy = torch.cumsum(g, dim=1)
        dist = torch.sqrt(((p_xy - g_xy) ** 2).sum(dim=-1) + self.eps)   # [B, T]

        T = dist.shape[1]
        if self.w.numel() == 0:
            w = dist.new_full((T,), 1.0 / T)
        else:
            assert self.w.numel() == T, (self.w.numel(), T)
            w = self.w.to(dist.dtype).to(dist.device)
        per = (dist * w).sum(dim=-1)                            # [B]

        if masks is not None:
            # 유효 waypoint가 하나도 없는 샘플은 통째로 제외한다. 부분 마스크는
            # 가중 재정규화가 필요해 지금은 다루지 않는다(ETRI는 masks가 전부 1이다 --
            # overfit8 pkl 실측 mean 1.0, 전부1인 샘플 비율 1.0).
            keep = masks.reshape(B, -1).min(dim=-1).values > 0.5
            if keep.sum() == 0:
                return per.sum() * 0.0
            per = per[keep]
        return self.loss_weight * per.mean()
