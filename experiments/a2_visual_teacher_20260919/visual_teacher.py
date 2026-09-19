"""Training-only RGB spatial distillation, with unchanged A2 inference graph."""
import json
import os
from pathlib import Path
import sys

ROOT = Path('/NHNHOME/data/sukim/adcl')
sys.path.insert(0, str(ROOT/'experiments/a2_next_20260919'))
from common import (torch, SceneExtensionModel, QREFINE, MotionDriveV2Config,
                    mr, tensor_state_sha256, sha)
from torch import nn
from torch.nn import functional as F

REPORT = ROOT/'reports/a2_visual_teacher_20260919'
RUNS = ROOT/'work_dirs/a2_visual_teacher_20260919'
DINO_REPO = ROOT/'third_party/dinov2_rgb_teacher'
DINO_REVISION = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINO_WEIGHTS = ROOT/'checkpoints/dinov2/dinov2_vitb14_reg4_pretrain.pth'
DINO_SHA = '73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71'
TEACHER_HW = (378, 672)
TEACHER_GRID = (27, 48)
PROJECTOR_PREFIX = 'visual_projector.'
VISUAL_SEED = 2026091901


def fork_devices():
    return list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []


class VisualStudent(SceneExtensionModel):
    """Capture the first/current-six-camera upper FPN level, once per forward."""
    def __init__(self, config):
        super().__init__(config, arm=QREFINE)
        with torch.random.fork_rng(devices=fork_devices()):
            torch.manual_seed(VISUAL_SEED)
            self.visual_projector = nn.Conv2d(128, 768, 1, bias=False)
        self.capture_in_eval = False
        self._visual_active = False
        self._visual_capture = []
        self._visual_calls = 0
        def capture(_module, args, output):
            if not self._visual_active:
                return
            self._visual_calls += 1
            if self._visual_calls != 1:
                return
            if tuple(args[0].shape[-2:]) != (432, 768) or len(args[0]) != self._visual_batch*6:
                raise ValueError('First backbone call must be the current six-camera scene pass')
            feature = output[0][1]
            assert tuple(feature.shape[1:]) == (128, 27, 48)
            self._visual_capture.append(feature)
        self.backbone_fpn.register_forward_hook(capture)

    def forward(self, *args, **kwargs):
        images = kwargs.get('images', args[0] if args else None)
        assert not self._visual_active and not self._visual_capture
        self._visual_active = self.training or self.capture_in_eval
        self._visual_batch = len(images)
        self._visual_calls = 0
        try:
            output = super().forward(*args, **kwargs)
            if self._visual_active:
                assert len(self._visual_capture) == 1 and self._visual_calls == 3
                output['_visual_feature'] = self._visual_capture[0]
            return output
        finally:
            self._visual_active = False
            self._visual_capture.clear()


def shared_state(model):
    return {k:v for k,v in model.state_dict().items() if not k.startswith(PROJECTOR_PREFIX)}


def load_student_parent(model, state):
    merged = dict(state)
    added = {k:v for k,v in model.state_dict().items() if k.startswith(PROJECTOR_PREFIX)}
    assert set(added) == {'visual_projector.weight'}
    merged.update(added)
    model.load_state_dict(merged, strict=True)
    assert tensor_state_sha256(shared_state(model)) == tensor_state_sha256(state)


def load_teacher(device='cuda:0'):
    import subprocess
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=DINO_REPO,text=True).strip() == DINO_REVISION
    assert sha(DINO_WEIGHTS) == DINO_SHA
    os.environ['XFORMERS_DISABLED'] = '1'
    sys.path.insert(0, str(DINO_REPO))
    from dinov2.hub.backbones import dinov2_vitb14_reg
    with torch.random.fork_rng(devices=fork_devices()):
        teacher = dinov2_vitb14_reg(pretrained=False)
    teacher.load_state_dict(torch.load(DINO_WEIGHTS,map_location='cpu',weights_only=True),strict=True)
    return teacher.to(device).eval().requires_grad_(False)


def spatial_grid(device, dtype=torch.float32):
    """Teacher patch centres -> original RGB pixel centres -> stride16 FPN.

    Resize is bilinear/align_corners=False, full FOV, no crop or padding.
    ResNet stride16 centres are (16*j,16*i) in zero-based input pixel centres.
    Exclude sites whose bilinear support leaves the student feature image.
    """
    ht,wt = TEACHER_GRID
    y = (torch.arange(ht,device=device,dtype=dtype)+.5)*14*432/TEACHER_HW[0]-.5
    x = (torch.arange(wt,device=device,dtype=dtype)+.5)*14*768/TEACHER_HW[1]-.5
    yy,xx = torch.meshgrid(y/16,x/16,indexing='ij')
    mask = (xx>=0)&(xx<=47)&(yy>=0)&(yy<=26)
    grid = torch.stack([2*(xx+.5)/48-1,2*(yy+.5)/27-1],-1)
    return grid,mask


@torch.no_grad()
def teacher_targets(teacher, images, chunk=12):
    if images.ndim != 5 or tuple(images.shape[1:]) != (6,3,432,768):
        raise ValueError('Expected already augmented current six-camera RGB only')
    assert not teacher.training and all(not p.requires_grad for p in teacher.parameters())
    device = next(teacher.parameters()).device
    result = []
    # A2 and official DINO RGB use ImageNet mean/std. Explicitly recover RGB
    # so interpolation geometry and normalization remain independently specified.
    mean = torch.tensor([.485,.456,.406],device=device)[None,:,None,None]
    std = torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
    flat = images.flatten(0,1)
    for start in range(0,len(flat),chunk):
        x = flat[start:start+chunk].to(device,non_blocking=True).float()
        rgb = x*std+mean
        rgb = F.interpolate(rgb,size=TEACHER_HW,mode='bilinear',align_corners=False,antialias=True)
        normalized = (rgb-mean)/std
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            tokens = teacher.forward_features(normalized)['x_norm_patchtokens']
        assert tokens.shape[1:] == (27*48,768)  # excludes CLS and all four registers
        result.append(tokens.detach().transpose(1,2).reshape(-1,768,27,48))
    targets = torch.cat(result).reshape(len(images),6,768,27,48)
    _,geometric_valid = spatial_grid(device)
    valid = torch.isfinite(targets).all(2)&geometric_valid[None,None]
    return targets,valid


def visual_loss(feature, projector, target, valid, full_count):
    """Unreduced sum divided by the one full-effective-batch valid count."""
    b,views,c,ht,wt = target.shape
    assert feature.shape == (b*views,128,27,48) and valid.shape == (b,views,ht,wt)
    with torch.autocast(device_type=feature.device.type,enabled=False):
        grid,_ = spatial_grid(feature.device)
        aligned = F.grid_sample(feature.float(),grid[None].expand(b*views,-1,-1,-1),
            mode='bilinear',padding_mode='zeros',align_corners=False)
        projection = projector(aligned).reshape(b,views,c,ht,wt)
        clean_target = torch.where(torch.isfinite(target),target,torch.zeros_like(target)).float()
        distance = 1-F.cosine_similarity(projection,clean_target,dim=2,eps=1e-6)
        selected = torch.where(valid,distance,torch.zeros_like(distance))
        loss = selected.sum()/torch.as_tensor(full_count,device=feature.device).clamp_min(1)
    return loss


def teacher_manifest():
    _,mask = spatial_grid('cpu')
    return dict(model='dinov2_vitb14_reg', architecture='ViT-B/14 with four registers',
        official_repository='https://github.com/facebookresearch/dinov2',
        revision=DINO_REVISION, checkpoint=str(DINO_WEIGHTS),checkpoint_sha256=DINO_SHA,
        official_weights_url='https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth',
        license='Apache-2.0 (DINOv2 code/model weights; not Cell-DINO or XRay-DINO)',
        license_sha256=sha(DINO_REPO/'LICENSE'), rgb_only=True,teacher_updated=False,
        output='forward_features.x_norm_patchtokens after final norm; CLS/registers excluded',
        feature_channels=768,teacher_input_hw=list(TEACHER_HW),teacher_grid_hw=list(TEACHER_GRID),
        student_feature='first/current-six-camera backbone_fpn forward output[0][1] = out3(P3)',
        student_shape_per_image=[128,27,48],student_stride=16,
        normalized_input='RGB ImageNet mean [.485,.456,.406], std [.229,.224,.225]',
        augmentation='same augmented student current-camera image; no extra teacher augmentation',
        resize='full FOV to 378x672 bilinear align_corners=False antialias=True, no crop/pad',
        alignment='teacher patch centre mapped to original pixel centre, then stride16 feature coordinate',
        valid_sites_per_image=int(mask.sum()), spatial_mask='bilinear support inside student grid, plus finite target',
        normalization='sum masked cosine distances / all valid sites in logical B16',
        projector='train-only bias-free Conv1x1 128->768; same 1e-5 head LR in selected recipe',
        teacher_batching='online, chunk12 current images; no cached targets or repeated student pass',
        adaptation_data='DEV train310 RGB only; no FULL/V0/H/test adaptation',
        pretraining_overlap='Public provenance recorded; full pretraining corpus overlap is not certified',
        inference='student only; teacher/projector excluded from deployment model')
