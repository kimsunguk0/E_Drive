"""Resources for one matched, semantic-command A2 experiment."""
from pathlib import Path
import sys
import torch

ROOT = Path('/NHNHOME/data/sukim/adcl')
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'reports/a2_command_20260919'
RUNS = ROOT / 'work_dirs/a2_command_20260919'
CACHE = ROOT / 'data/etri/motiondrive_v2/a2_command_20260919'
ARM = 'A2-COMMAND-NOM'
CONTROL = ROOT / 'work_dirs/md_a2_nominal_mh4_20260918/A2-BASE-NOM-s1'
for p in (ROOT, ROOT/'scripts', ROOT/'experiments/md_a2_nominal_mh4_20260918'):
    sys.path.insert(0, str(p))
import train_nominal as nominal
import train_motiondrive_v2 as trainer
from a2_model import A2NominalModel
from nominal_data import NominalStatusDataset, sha
from motiondrive_v2_training import model_inputs, to_device, tensor_state_sha256
from models.motiondrive_v2 import MotionDriveV2Config
legacy = nominal.legacy
mr = legacy.mr
sys.path.insert(0, str(HERE))

def inputs(batch, command=True):
    value = mr.model_inputs_with_canvas(model_inputs, batch, time_input='nominal')
    value['provided_status5'] = batch['provided_status5']
    if command:
        value['provided_command'] = batch['provided_command']
    return value
