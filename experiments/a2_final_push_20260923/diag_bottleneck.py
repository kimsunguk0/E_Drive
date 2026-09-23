#!/usr/bin/env python3
"""Bottleneck diagnostic on held-out V0 (analysis only, never a submission path).

The planner reads the motion head's image-predicted state/history. Measure how
accurate those are, and what PREFIX the SAME planner reaches if they were exact
(the ceiling for better image-based ego-motion, which Q10 allows).
"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path('/NHNHOME/data/sukim/adcl/experiments/a2_final_push_20260923')
sys.path.insert(0, str(HERE))
import eval_views as ev

W = np.array([11, 11, 5, 5, 2, 2]) / 36.0


class Swap(torch.nn.Module):
    def __init__(self, planner):
        super().__init__(); self.inner = planner; self.gt = None; self.extra = {}
    def forward(self, scene, motion, state, history, pairs):
        out = self.inner(scene, motion, state, history, pairs)
        gs, gh = self.gt
        self.extra = dict(
            gt_state=self.inner(scene, motion, gs.to(state.dtype), history, pairs),
            gt_state_hist=self.inner(scene, motion, gs.to(state.dtype), gh.to(history.dtype), pairs),
            state=state.float(), history=history.float())
        return out


def main():
    ck = ev.ROOT / 'work_dirs/a2_long_motion_20260922/L-TRAIN3106-s1/ckpt_step34326.pth'
    model, _ = ev.build_model([ck]); dev = torch.device('cuda:0'); model.to(dev).eval()
    swap = Swap(model.planner); model.planner = swap
    raw = ev.MotionDriveDataset(data_root='/tmp/pm97', split_manifest=str(ev.SPLIT), split='tune',
                                supervision_root=str(ev.SUP), min_frame=30, frame_stride=5, augment=False,
                                seed=0, history_contract='control')
    ds = ev.H4StatusDataset(ev.mr.MotionCanvasDataset(raw, 'native'))
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
    keep = {k: [] for k in ('plan', 'gt_state', 'gt_state_hist', 'state', 'history', 'state_t', 'hist_t', 'status5', 'gt', 'row')}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for rawb in loader:
            b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in rawb.items()}
            x = ev.mr.model_inputs_with_canvas(ev.trainer.model_inputs, b, time_input='nominal')
            x['provided_status5'] = b['provided_status5']
            if len(keep['row']) == 0:
                print('state_target', tuple(b['state_target'].shape), 'history_target', tuple(b['history_target'].shape))
            swap.gt = (b['state_target'].float(), b['history_target'].float())
            out = model(**x)
            keep['plan'].append(out['plan_abs'].float().cpu()); keep['gt'].append(rawb['gt_plan'].float())
            for k in ('gt_state', 'gt_state_hist', 'state', 'history'): keep[k].append(swap.extra[k].float().cpu())
            keep['state_t'].append(rawb['state_target'].float()); keep['hist_t'].append(rawb['history_target'].float())
            keep['status5'].append(rawb['provided_status5'].float()); keep['row'].append(rawb['row'])
    a = {k: torch.cat(v).numpy() for k, v in keep.items()}
    np.savez_compressed(ev.OUT / 'DIAG_L34_state.npz', **a)
    sc = lambda p: float((np.linalg.norm(p - a['gt'], axis=-1) @ W).mean())
    st, sh = a['state'], a['state_t']
    print(json.dumps(dict(
        PREFIX_predicted_state=sc(a['plan']), PREFIX_gt_state=sc(a['gt_state']),
        PREFIX_gt_state_and_history=sc(a['gt_state_hist']),
        state_MAE_per_dim=np.abs(st[:, :5] - sh[:, :5]).mean(0).round(4).tolist(),
        state_target_std=sh[:, :5].std(0).round(3).tolist(),
        status5_vs_state_target_MAE=np.abs(a['status5'] - sh[:, :5]).mean(0).round(4).tolist(),
        history_MAE_per_dim=np.abs(a['history'] - a['hist_t']).mean((0, 1)).round(4).tolist(),
        plan_first_speed_err=float(np.abs(a['plan'][:, 0, 0] - a['gt'][:, 0, 0]).mean() / .5),
    ), indent=1))


if __name__ == '__main__':
    main()
