"""Future footprint quality for DYN0/DYN1 against two persistence baselines."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path("/NHNHOME/data/sukim/adcl")
D = ROOT / "experiments/motiondrive_round_20260912"
for p in (str(D), str(ROOT), str(ROOT / "scripts")):
    sys.path.insert(0, p)
import run_round_dyn as rs
from motiondrive_v2_data import MotionDriveDataset
from motiondrive_v2_training import model_inputs
from models.motiondrive_v2 import MotionDriveV2Config


def iou(pred, target, valid):
    p, t, v = pred & valid, target.astype(bool) & valid, valid
    union = (p | t) & v
    return float((p & t).sum()) / max(int(union.sum()), 1)


future = np.load(ROOT / "cache/motiondrive_round_20260912/future_targets/tune.npz",
                 allow_pickle=False)
lookup = {int(r): i for i, r in enumerate(future["row"])}
dataset = MotionDriveDataset(
    data_root="/tmp/pm97",
    split_manifest=str(ROOT / "data/etri/motiondrive_v2/grouped_split_rawtime.json"),
    split="tune", supervision_root=str(ROOT / "data/etri/motiondrive_v2/train_tune_geometry_v2"),
    min_frame=30, frame_stride=5, max_samples=0, augment=False, seed=0,
    history_contract="control")
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False,
                                     num_workers=4, pin_memory=True)
result = {}
for arm in ("dyn0", "dyn1"):
    run = ROOT / "work_dirs/motiondrive_round_20260912" / arm
    payload = torch.load(run / "last.pth", map_location="cpu", weights_only=False)
    manifest = json.loads((run / "manifest.json").read_text())
    config = MotionDriveV2Config(**manifest["model_config"])
    rs._ARM["name"] = arm
    model = rs.install_dynamic(rs.make_model(config), config)
    model.load_state_dict(payload["model"], strict=True)
    model = model.cuda().eval()
    store = {k: [] for k in ("future_logits", "occ_logits", "target", "valid")}
    with torch.inference_mode():
        for batch in loader:
            inputs = model_inputs(batch, time_input="nominal",
                                  nominal_history_seconds=config.nominal_history_seconds)
            inputs = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                      for k, v in inputs.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**inputs)
            rows = batch["row"].numpy().astype(int)
            index = [lookup[int(r)] for r in rows]
            store["future_logits"].append(out["future_logits"].float().cpu().numpy())
            store["occ_logits"].append(out["occ_logits"].float().cpu().numpy())
            store["target"].append(future["future_target"][index])
            store["valid"].append(future["future_valid"][index])
    arrays = {k: np.concatenate(v, 0) for k, v in store.items()}
    predicted = arrays["future_logits"] > 0
    occupancy = arrays["occ_logits"][:, 0] > 0
    entry = {}
    for h in range(2):
        v, t = arrays["valid"][:, h], arrays["target"][:, h]
        entry[f"h{h}_future_head_iou"] = iou(predicted[:, h], t, v)
        entry[f"h{h}_model_persistence_iou"] = iou(occupancy, t, v)
        entry[f"h{h}_positive_rate"] = float(t[v].mean())
    result[arm] = entry
    del model
    torch.cuda.empty_cache()

# GT-current persistence: the current GT footprint held in place, a privileged check.
current = []
with torch.inference_mode():
    for batch in loader:
        current.append(batch["occ_target"].numpy())
current = np.concatenate(current, 0)[:, 0] > .5
gt_persist = {}
for h in range(2):
    v, t = future["future_valid"][[lookup[int(r)] for r in
                                   np.concatenate([b["row"].numpy() for b in
                                                   torch.utils.data.DataLoader(
                                                       dataset, batch_size=256, num_workers=2,
                                                       collate_fn=lambda x: {"row": torch.tensor(
                                                           [i["row"] for i in x])})])]][:, h], None
    pass
rows_all = np.array([dataset[i]["row"] for i in range(len(dataset))], dtype=int)
index_all = [lookup[int(r)] for r in rows_all]
for h in range(2):
    v = future["future_valid"][index_all][:, h]
    t = future["future_target"][index_all][:, h]
    gt_persist[f"h{h}_gt_current_persistence_iou"] = iou(current, t, v)
result["gt_current_persistence_privileged"] = gt_persist
print(json.dumps(result, indent=1))
Path(sys.argv[1]).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
