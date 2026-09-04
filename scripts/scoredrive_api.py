"""ScoreDrive in-model complete-candidate API + external selector (canonical logic).

설계서 §3.2/§3.3/§7 + user spec (③):
  - 모델 반환: candidate_xy_abs_5s [B,12,10,2], candidate_xy_inc_5s [B,12,10,2],
    visual_logits [B,12], candidate_ids [B,12].  full-K 미노출. goal/cmd 미입력.
  - shortlist = 확정 score3 + nms9 (stable visual sort, train-only NMS threshold).
  - 선택기: J = norm_goal + λ·norm_visual, zero-evidence 시 goal fallback 금지(visual top-1).

numpy(오프라인 gate)와 torch(모델) 두 경로가 bitwise 동일한 shortlist를 내도록 구현.
"""
import numpy as np

N_OUT = 12
SCORE_TOP = 3
NMS_POOL = 64


# --------------------------- shortlist (numpy) ---------------------------
def shortlist_np(logit_row, anchor_dist, nms_tau):
    """확정 score3+nms9 shortlist. logit_row [K], anchor_dist [K,K] (diag=+inf)."""
    order = np.argsort(-logit_row, kind="stable")
    sel = list(order[:SCORE_TOP])
    sset = set(sel)
    for c in order[:NMS_POOL]:
        if len(sel) >= N_OUT:
            break
        if c in sset:
            continue
        if anchor_dist[c, sel].min() > nms_tau:
            sel.append(int(c)); sset.add(int(c))
    for c in order:
        if len(sel) >= N_OUT:
            break
        if int(c) not in sset:
            sel.append(int(c)); sset.add(int(c))
    return np.array(sel[:N_OUT], dtype=np.int64)


def build_candidates_np(logits, abs5, inc5, ids, anchor_dist, nms_tau):
    """logits [B,K] -> dict. abs5/inc5 [K,10,2], ids [K]."""
    B = logits.shape[0]
    sel = np.stack([shortlist_np(logits[b], anchor_dist, nms_tau) for b in range(B)])  # [B,12]
    return {
        "candidate_ids": ids[sel],                       # [B,12]
        "candidate_xy_abs_5s": abs5[sel],                # [B,12,10,2]
        "candidate_xy_inc_5s": inc5[sel],                # [B,12,10,2]
        "visual_logits": np.take_along_axis(logits, sel, axis=1),  # [B,12]
        "_shortlist_idx": sel,                           # 내부용 (parent index)
    }


# --------------------------- selector (numpy) ----------------------------
def select_index_np(cand, goal_xy, lambda_visual, eps=1e-9):
    """cand dict(단일 batch row 슬라이스: abs [12,10,2], vlog [12]) -> shortlist-local index.
    zero-evidence(exact tie)면 visual top-1(=0) 반환. lambda=inf면 visual top-1."""
    vlog = cand["visual_logits"]                          # [12]
    abs5 = cand["candidate_xy_abs_5s"]                    # [12,10,2]
    # 3.3 compliance: 영상 증거 없음 -> goal fallback 금지
    if float(vlog.max()) == float(vlog.min()):
        return 0                                          # stable visual top-1 (= exact stop when zero-feature)
    if not np.isfinite(lambda_visual):                    # lambda=inf -> pure visual
        return int(np.argmin(vlog.max() - vlog))
    gc = np.linalg.norm(abs5[:, 9] - goal_xy, axis=1)     # endpoint dist (abs buffer)
    vc = vlog.max() - vlog
    gcn = (gc - gc.min()) / (gc.max() - gc.min() + eps)
    vcn = (vc - vc.min()) / (vc.max() - vc.min() + eps)
    return int(np.argmin(gcn + lambda_visual * vcn))


def select_path_np(cand_b, goal_xy, lambda_visual):
    """반환: 제출 첫6점(inc), 선택 shortlist-local index, 전역 candidate id."""
    j = select_index_np(cand_b, goal_xy, lambda_visual)
    inc6 = cand_b["candidate_xy_inc_5s"][j, :6]
    return inc6, j, int(cand_b["candidate_ids"][j])


def slice_batch(cand, b):
    return {k: v[b] for k, v in cand.items() if not k.startswith("_")}


# --------------------------- shortlist (torch mirror) --------------------
def shortlist_torch(logits, anchor_dist, nms_tau):
    """torch [B,K] -> [B,12] long. numpy 경로와 bitwise 동일해야 한다.
    torch.argsort(stable=True, descending=True) 는 동률 시 최저 index 유지 = np.argsort(-x,'stable')."""
    import torch
    B, K = logits.shape
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)  # [B,K]
    out = torch.empty(B, N_OUT, dtype=torch.long, device=logits.device)
    ad = anchor_dist
    for b in range(B):
        ob = order[b]
        sel = ob[:SCORE_TOP].tolist()
        sset = set(sel)
        for c in ob[:NMS_POOL].tolist():
            if len(sel) >= N_OUT:
                break
            if c in sset:
                continue
            if float(ad[c, sel].min()) > nms_tau:
                sel.append(c); sset.add(c)
        for c in ob.tolist():
            if len(sel) >= N_OUT:
                break
            if c not in sset:
                sel.append(c); sset.add(c)
        out[b] = torch.tensor(sel[:N_OUT], device=logits.device)
    return out
