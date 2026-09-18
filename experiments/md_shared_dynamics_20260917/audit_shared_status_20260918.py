"""Read-only image/status interventions on terminal A2/A3 checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path('/NHNHOME/data/sukim/adcl')
for directory in (ROOT, ROOT / 'scripts',
                  ROOT / 'experiments/md_r0_reset_20260914',
                  ROOT / 'experiments/md_shared_dynamics_20260917'):
    sys.path.insert(0, str(directory))

from factorized_model import SharedDynamicsMotionDriveV2
from models.motiondrive_v2 import MotionDriveV2Config
from train_shared_dynamics import CausalStatusDataset, raw_datasets
import matching_resolution as mr
from motiondrive_v2_training import model_inputs, to_device

IMAGE_KEYS = ('images', 'history_images', mr.MOTION_CURRENT_KEY, mr.MOTION_HISTORY_KEY)
CONDITIONS = ('normal', 'images_other_session', 'images_zero_normalized', 'status_other_session')
W = np.asarray([11, 11, 5, 5, 2, 2], np.float64) / 36


class PairedImages(Dataset):
    def __init__(self, base, donor_indices):
        self.base, self.donor_indices = base, donor_indices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        donor = self.base[int(self.donor_indices[index])]
        for key in IMAGE_KEYS:
            item['donor_' + key] = donor[key]
        item['donor_status'] = donor['provided_status5']
        return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('A2-DIRECT', 'A3-DIRECT'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260918)
    run = ROOT / 'work_dirs/md_shared_dynamics_20260917' / (args.arm + '-s1')
    payload = torch.load(run / 'ckpt_step20554.pth', map_location='cpu', weights_only=False)
    model = SharedDynamicsMotionDriveV2(MotionDriveV2Config(**payload['manifest']['model_config']), arm=args.arm)
    model.load_state_dict(payload['model'], strict=True)
    model.eval().cuda()
    del payload
    _, raw = raw_datasets(1)
    records = json.loads((run / 'final_eval.json').read_text())['records']
    assert [int(x['row']) for x in records] == raw.rows.tolist()
    session = np.asarray([x['session'] for x in records])
    frame = np.asarray([int(x['frame']) for x in records])
    rng = np.random.default_rng(20260918)
    donor = []
    for index in range(len(records)):
        candidates = np.flatnonzero((session != session[index]) & (frame == frame[index]))
        assert len(candidates)
        donor.append(int(rng.choice(candidates)))
    dataset = PairedImages(CausalStatusDataset(mr.MotionCanvasDataset(raw, 'native')), donor)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4,
                        pin_memory=True, prefetch_factor=2)
    predicted = {condition: [] for condition in CONDITIONS}
    states = {condition: [] for condition in CONDITIONS}
    target, state_target = [], []
    started = time.monotonic()
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            batch = to_device(batch, torch.device('cuda:0'))
            inputs = mr.model_inputs_with_canvas(model_inputs, batch, time_input='nominal')
            inputs['provided_status5'] = batch['provided_status5']
            names = dict(zip(IMAGE_KEYS, ('images', 'history_images', 'motion_current', 'motion_history')))
            for condition in CONDITIONS:
                current = dict(inputs)
                if condition == 'images_other_session':
                    for key, input_key in names.items():
                        current[input_key] = batch['donor_' + key]
                elif condition == 'images_zero_normalized':
                    for input_key in names.values():
                        current[input_key] = torch.zeros_like(inputs[input_key])
                elif condition == 'status_other_session':
                    current['provided_status5'] = batch['donor_status']
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(**current)
                predicted[condition].append(output['plan_abs'].float().cpu().numpy())
                states[condition].append(output['state_hat'].float().cpu().numpy())
            target.append(batch['gt_plan'].float().cpu().numpy())
            state_target.append(batch['state_target'].float().cpu().numpy())
            if index % 50 == 0:
                print(json.dumps({'arm': args.arm, 'batches': index + 1,
                                  'elapsed_s': time.monotonic() - started}), flush=True)
    target, state_target = np.concatenate(target), np.concatenate(state_target)
    np.testing.assert_allclose(target, np.asarray([r['gt_abs_xy'] for r in records]), atol=0, rtol=0)
    predicted = {k: np.concatenate(v).astype(np.float64) for k, v in predicted.items()}
    states = {k: np.concatenate(v).astype(np.float64) for k, v in states.items()}
    normal = predicted['normal']
    result = {
        'arm': args.arm, 'checkpoint': str(run / 'ckpt_step20554.pth'),
        'n': len(records), 'training_performed': False,
        'donor_policy': 'same frame, different session, seed 20260918; no label-based selection',
        'images_replaced': list(IMAGE_KEYS),
        'receiver_metadata_preserved_in_image_interventions': ['pose alignment', 'goal', 'status', 'calibration', 'timestamps', 'GT'],
        'normal_stored_plan_max_abs_difference': float(np.max(np.abs(normal - np.asarray([r['pred_abs_xy'] for r in records])))),
        'conditions': {},
        'limitation': 'Counterfactual sensitivity is evidence of image use, not organizer approval or proof against learned status shortcuts.',
    }
    for condition in CONDITIONS:
        e = np.linalg.norm(predicted[condition] - target, axis=-1)
        delta = np.linalg.norm(predicted[condition] - normal, axis=-1)
        result['conditions'][condition] = {
            'PREFIX': float((e @ W).mean()),
            'L2_1s': float(e[:, :2].mean()), 'L2_2s': float(e[:, :4].mean()), 'L2_3s': float(e.mean()),
            'plan_change_from_normal_PREFIX': float((delta @ W).mean()),
            'state_mae_vx_vy_ax_ay_yaw': np.abs(states[condition][:, :5] - state_target[:, :5]).mean(0).tolist(),
        }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    np.savez_compressed(path.with_suffix('.npz'),
                        rows=raw.rows, donor_indices=np.asarray(donor), gt=target,
                        **{'plan_' + k: v for k, v in predicted.items()},
                        **{'state_' + k: v for k, v in states.items()})
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
