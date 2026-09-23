#!/usr/bin/env python3
"""Turn tta_flops.json into the report package_submission.py reads: the
submitted __flops__ is the two-forward TTA count, since both forwards are scored.
Usage: flops_report.py TTA_FLOPS_JSON CKPT[,CKPT...] OUT_JSON"""
import json, sys
from pathlib import Path
import torch
ROOT = Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0, str(ROOT / 'scripts'))
from motiondrive_v2_training import tensor_state_sha256

tta = json.loads(Path(sys.argv[1]).read_text())
ckpts = sys.argv[2].split(',')
states = [torch.load(p, map_location='cpu', weights_only=False)['model'] for p in ckpts]
if len(states) == 1:
    state = states[0]
else:
    state = {k: (torch.stack([s[k].double() for s in states]).mean(0).to(v.dtype)
                 if v.is_floating_point() else v) for k, v in states[0].items()}
flops = int(tta['tta_two_forward_flops'])
out = dict(flops=flops, gflops=flops / 1e9, passes_cutoff=bool(flops / 1e9 <= 7053.0),
           cutoff_gflops=7053.0, counter=tta['counter'],
           scope='two full forwards per clip (upright + mirrored Flip TTA), FP32 counting',
           single_forward_flops=int(tta['single_forward_flops']),
           checkpoint=','.join(str(Path(p).resolve()) for p in ckpts),
           weight_soup=len(ckpts) > 1, model_state_sha256=tensor_state_sha256(state),
           source=str(Path(sys.argv[1]).resolve()))
Path(sys.argv[3]).write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
