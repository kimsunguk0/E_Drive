#!/usr/bin/env python
"""v2a 330 pilot이 iter ~620에서 죽은 **최초 사건**을 잡는다.

1차 프로브가 알아낸 것 (iter 621)
    img_feats[0] (4,6,6,256,14,24)  NaN 12,386,304 / 12,386,304  = 100%
    bbox_pred / cls_pred  전부 NaN
    gt_bboxes finite, w/l/h 최소 3.01/8.23/3.86 -> GT 무죄

전파 기전
    `img_backbone`/`img_neck`은 lr_mult=0.0으로 동결했는데도 출력이 100% NaN이었다.
    PyTorch AdamW는 lr=0에서도 `p.addcdiv_(exp_avg, denom, value=-step_size)`를
    실행하고 IEEE754에서 **0 x NaN = NaN** 이다. 즉 gradient가 한 번 NaN이 되면
    (`clip_grad_norm_`의 total_norm이 NaN이면 **모든** grad에 NaN이 곱해진다)
    lr=0 파라미터까지 전부 오염된다. **lr=0은 NaN 방어가 아니다.**

그래서 필요한 건 25 iter 간격 스냅샷이 아니라 **매 iteration** 검사다.
    (1) loss dict  -- 어느 항이 먼저 NaN인가            (backward 전)
    (2) gradient   -- 어느 파라미터가 먼저 NaN인가        (clip 전)
    (3) grad_norm  -- clip이 NaN을 전파했는가            (clip 후)

    python scripts/etri_v2a_nan_probe.py --config <cfg> --max-iters 750
"""
import argparse
import os
import runpy
import sys

DUMP = "/tmp/pm97/v2a_nan_probe.txt"


def log(msg):
    print(msg, flush=True)
    with open(DUMP, "a") as f:
        f.write(msg + "\n")


def stat(name, t):
    import torch
    if t is None:
        return f"{name}: None"
    if not torch.is_tensor(t):
        return f"{name}: {type(t).__name__}"
    f = torch.isfinite(t)
    fin = t[f]
    rng = (f"[{float(fin.min()):.4g}, {float(fin.max()):.4g}]"
           if fin.numel() else "[empty]")
    return (f"{name}{tuple(t.shape)} nan {int(torch.isnan(t).sum())} "
            f"inf {int(torch.isinf(t).sum())} / {t.numel()} {rng}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--work-dir", default="/tmp/pm97/work_dirs/c0t_v2a_probe")
    ap.add_argument("--max-iters", type=int, default=750)
    ap.add_argument("--anomaly-from", type=int, default=590,
                    help="이 iteration부터 autograd anomaly detection을 켠다")
    args = ap.parse_args()
    open(DUMP, "w").close()

    import torch
    import projects.mmdet3d_plugin  # noqa: F401
    from projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d import (
        HungarianAssigner3D)
    from mmdet.models.detectors.base import BaseDetector
    from mmcv.runner.hooks.optimizer import OptimizerHook

    st = {"it": 0, "done": False}

    def finish(reason):
        st["done"] = True
        log(f"\n[종료] {reason}")
        raise SystemExit(0)

    # ---- (1) loss dict, 매 iteration ----
    orig_step = BaseDetector.train_step

    def train_step(self, data, optimizer):
        if not st.get("name_of"):
            st["name_of"] = {id(p): n for n, p in self.named_parameters()}
        # 크래시 직전부터 anomaly detection을 켠다. 전 구간에 켜면 2~5배 느려지고,
        # 실측 발화 지점이 iter 621~622이므로 590부터면 충분하다. 이게 NaN을 만든
        # **backward 연산 이름**을 직접 알려준다.
        if st["it"] == args.anomaly_from:
            log(f"\n[anomaly detection ON] iter {st['it']+1} 부터")
            torch.autograd.set_detect_anomaly(True)
        out = orig_step(self, data, optimizer)
        st["it"] += 1
        it = st["it"]
        bad = []
        for k, v in out.get("log_vars", {}).items():
            if v != v or v in (float("inf"), float("-inf")):
                bad.append((k, v))
        loss_t = out.get("loss")
        loss_bad = torch.is_tensor(loss_t) and not torch.isfinite(loss_t).all()
        if bad or loss_bad:
            log(f"\n{'='*72}\n[최초 non-finite LOSS] iter {it}")
            log(f"  총 loss  {float(loss_t) if torch.is_tensor(loss_t) else loss_t}")
            log(f"  NaN/inf 항 {len(bad)}:")
            for k, v in bad:
                log(f"    {k} = {v}")
            log(f"  finite 항 (참고):")
            for k, v in list(out.get("log_vars", {}).items()):
                if (k, v) not in bad:
                    log(f"    {k} = {v:.6g}")
            # 이 시점의 파라미터 상태 -- loss가 먼저 터졌나 param이 먼저 터졌나
            bp = [n for n, p in self.named_parameters()
                  if not torch.isfinite(p).all()]
            log(f"  이 시점 non-finite 파라미터 {len(bp)}"
                + (f"  예: {bp[:4]}" if bp else "  -> loss가 먼저다 (param은 아직 깨끗)"))
            log(f"{'='*72}\n")
            finish(f"iter {it}에서 loss가 먼저 non-finite")
        if it % 100 == 0:
            log(f"[iter {it}] loss {float(loss_t):.4f}  전부 finite")
        if it >= args.max_iters:
            finish(f"{args.max_iters} iter까지 크래시 없음 -- 비결정적 사건")
        return out
    BaseDetector.train_step = train_step

    # ---- (2)(3) gradient, clip 전/후 ----
    orig_clip = OptimizerHook.clip_grads

    def clip_grads(self, params):
        params = list(params)
        # 이름을 알아야 어느 서브그래프인지 특정된다. id -> name 역인덱스를 쓴다.
        pre_bad = [p for p in params
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
        if pre_bad and st.get("name_of"):
            names = [st["name_of"].get(id(p), "?") for p in pre_bad]
            log(f"\n[clip 전 NaN grad 파라미터 {len(names)}개]")
            import collections
            grp = collections.Counter()
            for n in names:
                if n.startswith("pts_bbox_head.transformer"):
                    grp["pts_bbox_head.transformer"] += 1
                elif n.startswith("pts_bbox_head"):
                    grp["pts_bbox_head." + n.split(".")[1]] += 1
                else:
                    grp[n.split(".")[0]] += 1
            for k, v in sorted(grp.items(), key=lambda x: -x[1]):
                log(f"    {v:>4}  {k}")
            log(f"  전체 목록(앞 20):")
            for n in sorted(names)[:20]:
                log(f"    {n}")
        gn = orig_clip(self, params)
        post_bad_n = sum(1 for p in params
                         if p.grad is not None and not torch.isfinite(p.grad).all())
        if pre_bad or (gn is not None and not torch.isfinite(
                gn if torch.is_tensor(gn) else torch.tensor(float(gn)))):
            log(f"\n{'='*72}\n[최초 non-finite GRAD] iter {st['it']+1}")
            log(f"  clip 전 non-finite grad 파라미터 {len(pre_bad)} / "
                f"{sum(1 for p in params if p.grad is not None)}")
            log(f"  clip 후 non-finite grad 파라미터 {post_bad_n}")
            log(f"  grad_norm {gn}")
            log(f"  -> clip 전 0개인데 후에 전부면 `clip_grad_norm_`의 total_norm이")
            log(f"     NaN/inf였다는 뜻이다 (전 grad에 NaN이 곱해진다).")
            log(f"{'='*72}\n")
            finish(f"iter {st['it']+1}에서 grad가 먼저 non-finite")
        return gn
    OptimizerHook.clip_grads = clip_grads

    # ---- assigner (최후 방어선) ----
    orig_assign = HungarianAssigner3D.assign

    def assign(self, bbox_pred, cls_pred, gt_bboxes, gt_labels, *a, **kw):
        try:
            return orig_assign(self, bbox_pred, cls_pred, gt_bboxes,
                               gt_labels, *a, **kw)
        except ValueError as e:
            log(f"\n[assigner ValueError] iter {st['it']+1}  {e}")
            log(f"  {stat('bbox_pred', bbox_pred)}")
            log(f"  {stat('gt_bboxes', gt_bboxes)}")
            raise
    HungarianAssigner3D.assign = assign

    log(f"probe2 시작  {args.config}  max_iters {args.max_iters}")
    log(f"매 iteration loss/grad 검사. 첫 non-finite에서 멈춘다.")
    sys.argv = ["tools/train.py", os.path.abspath(args.config),
                "--work-dir", args.work_dir,
                "--no-validate", "--deterministic", "--seed", "0"]
    runpy.run_path("tools/train.py", run_name="__main__")


if __name__ == "__main__":
    main()
