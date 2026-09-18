"""Pinned A2 joint-learning/sampling experiment resources."""
from pathlib import Path
import sys
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/a2_next_20260919'
RUNS = ROOT / 'work_dirs/a2_next_20260919'
for p in (ROOT, ROOT/'scripts', ROOT/'experiments/md_a2_nominal_mh4_20260918',
          ROOT/'experiments/md_a2_scene_extensions_20260918'):
    sys.path.insert(0, str(p))
import train_nominal as nominal
import train_motiondrive_v2 as trainer
from scene_extensions import SceneExtensionModel, QREFINE
from a2_model import A2NominalModel
from nominal_data import NominalStatusDataset, sha, CACHE
from motiondrive_v2_training import (LossWeights, compute_loss, build_loss_normalizers,
    model_inputs, to_device, tensor_state_sha256)
from models.motiondrive_v2 import MotionDriveV2Config
legacy = nominal.legacy
mr = legacy.mr
from length_auxiliary import length_loss, wrap_compute_loss

PARENT = ROOT/'work_dirs/md_a2_scene_extensions_20260918/A2-QREFINE-NOM-s1/ckpt_step20554.pth'
BASE_CONTROL = ROOT/'work_dirs/md_a2_nominal_mh4_20260918/A2-BASE-NOM-s1'
G_ARMS = ('A2-G0', 'A2-G1')
S_ARM = 'A2-LEARNED-SAMPLE'

def inputs(batch):
    value = mr.model_inputs_with_canvas(model_inputs, batch, time_input='nominal')
    value['provided_status5'] = batch['provided_status5']
    return value

def load_parent():
    common = torch.load(PARENT, map_location='cpu', weights_only=False)
    model = SceneExtensionModel(MotionDriveV2Config(**common['manifest']['model_config']), arm=QREFINE)
    mr.rebuild_correlation_fuse(model, 4)
    model.load_state_dict(common['model'], strict=True)
    return model, common

def training_data():
    import motiondrive_v2_data as api
    from motiondrive_v2_flip_augment import FlipAugmented
    raw = api.MotionDriveDataset(data_root='/tmp/pm97', split_manifest=str(legacy.SPLIT),
        supervision_root=str(legacy.SUPERVISION), min_frame=30, max_samples=0, seed=1,
        history_contract='control', split='train', frame_stride=1, augment=True)
    import motiondrive_v2_flip_augment as flip
    # The same native-motion/status flip wrapper used by the actual trainer.
    flip.flip_item = mr.wrap_flip_item(flip.flip_item)
    data = FlipAugmented(NominalStatusDataset(mr.MotionCanvasDataset(raw, 'native')),768,384,p=.5,seed=1)
    data.set_epoch(0)
    return data

def separated_loss(output, batch, weights, normalizers=None):
    total, parts = compute_loss(output,batch,weights,normalizers=normalizers)
    prefix = weights.plan * parts['plan_d3']
    length = .25 * length_loss(output,batch,normalizers)
    auxiliary = weights.occupancy*parts['occ_bce'] + weights.lane*parts['lane_bce'] + weights.motion*parts['motion']
    reconstructed = prefix + auxiliary
    assert torch.allclose(total, reconstructed, atol=1e-6, rtol=1e-6)
    return {'P':prefix, 'M':prefix+length, 'A':auxiliary}, parts, (total-reconstructed).abs()
