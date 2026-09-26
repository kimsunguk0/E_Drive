"""Analysis only: ceiling for a better motion head under goal-endpoint selection.
Run the held-out twin ExtModel on V0 with the planner reading (a) its own
image-predicted state/history and (b) the exact targets."""
import sys, json
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_ext as te
from build_ext_submission import load
W = np.array([11, 11, 5, 5, 2, 2]) / 36.0

class Swap(torch.nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner; self.gt = None; self.use_gt = False
    def forward(self, scene, motion, state, history, pairs):
        if self.use_gt: state, history = self.gt[0].to(state.dtype), self.gt[1].to(history.dtype)
        return self.inner(scene, motion, state, history, pairs)

ck = sys.argv[1]
model, _ = load(ck, torch.device('cuda:0')); sw = Swap(model.planner); model.planner = sw
raw_train, raw_tune = te.base.nominal.raw_datasets(False, 1)
ds = te.base.wrapped_eval(raw_tune, False)
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=6, pin_memory=True)
out = {False: [], True: []}; gts = []
with torch.no_grad():
    for b in loader:
        x = te.inputs(b, torch.device('cuda:0'))
        sw.gt = (b['state_target'].float().cuda(), b['history_target'].float().cuda())
        for g in (False, True):
            sw.use_gt = g
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out[g].append(model(**x)['plan_abs'].float().cpu())
        gts.append(b['gt_plan'].float())
gt = torch.cat(gts).numpy()
res = {('exact state/history' if g else 'image-predicted state/history'): float((np.linalg.norm(torch.cat(v).numpy() - gt, axis=-1) @ W).mean()) for g, v in out.items()}
print(json.dumps(res, indent=1))
