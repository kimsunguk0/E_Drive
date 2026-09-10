"""Raw ETRI clip -> current three-camera SparseDriveV2 fixed-bank trajectory.

No training dataset, labels, global ego cache, scene ID, timestamp, or command
file is read by this module. Checkpoint/bank paths are explicit initialization
artifacts. Runtime files are calibration.parquet, ego_pose.parquet, and exactly
three current JPEGs. Candidate coordinates are never adjusted at inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence
from threading import Lock

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

from models.motiondrive_v2_inputs import (
    CALIBRATION_COLUMNS, POSE_COLUMNS, build_camera_geometry, cache_compatible_image,
)
try:
    from .public_model import PUBLIC_SHA256, PublicSparseDriveV2, load_etri_bank
    from .goal_selector import GoalConditionedSelector
except ImportError:
    from public_model import PUBLIC_SHA256, PublicSparseDriveV2, load_etri_bank
    from goal_selector import GoalConditionedSelector


CAMERAS = ('camera_front_left', 'camera_front', 'camera_front_right')
IMAGE_WH = (512, 256)
MEAN = np.asarray([.485, .456, .406], np.float32)
STD = np.asarray([.229, .224, .225], np.float32)
POSE_FIELDS = set(POSE_COLUMNS)
BANK_NAMES = ('path_vocab', 'vel_vocab', 'traj_vocab', 'traj_mask')


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _pose_rows(records):
    rows = {}
    for record in records:
        _require(set(record) == POSE_FIELDS, 'Pose records must contain only official frame/XYZ/RPY columns')
        frame = record['frame']
        _require(not isinstance(frame, bool) and int(frame) == frame, 'Invalid pose frame')
        frame = int(frame)
        _require(frame not in rows, 'Duplicate pose frame')
        rows[frame] = record
    _require(set(rows) == set(range(-30, 1)) | {50}, 'Expected official frames -30..0 and +50 only')
    return rows


def causal_status_from_records(records):
    """Exact overlay fit: nominal 10Hz, 11 poses, free-intercept quadratic.

    Only poses -10..0 contribute. Current full-SE3 frame, m/s and m/s².
    The yaw-rate result is retained in metadata; model status uses its first4.
    """
    rows = _pose_rows(records)
    selected = [rows[i] for i in range(-10, 1)]
    xyz = np.asarray([[r[k] for k in ('x','y','z')] for r in selected], np.float64)
    rpy = np.asarray([[r[k] for k in ('roll','pitch','yaw')] for r in selected], np.float64)
    _require(np.isfinite(xyz).all() and np.isfinite(rpy).all(), 'Nonfinite causal pose')
    poses = np.broadcast_to(np.eye(4), (11,4,4)).copy()
    poses[:,:3,:3] = Rotation.from_euler('xyz', rpy).as_matrix()
    poses[:,:3,3] = xyz
    relative = np.linalg.inv(poses[-1]) @ poses
    t = np.arange(-10,1,dtype=np.float64)/10.
    design = np.column_stack((np.ones_like(t), t, .5*t*t))
    position = np.linalg.lstsq(design, relative[:,:2,3], rcond=None)[0]
    angle = np.unwrap(np.arctan2(relative[:,1,0], relative[:,0,0]))
    angular = np.linalg.lstsq(design, angle, rcond=None)[0]
    status5 = np.asarray([*position[1], *position[2], angular[1]], np.float32)
    residual = relative[:,:2,3] - design @ position
    _require(np.linalg.cond(design) < 100 and np.isfinite(status5).all()
             and np.sqrt(np.mean(residual**2)) < .25, 'Invalid causal nominal status fit')
    return status5


def provided_goal_from_records(records):
    """Use current pose and provided +50 XYZ only; future orientation is unused."""
    rows = _pose_rows(records)
    current, future = rows[0], rows[50]
    xyz = np.asarray([current[k] for k in ('x','y','z')], np.float64)
    rpy = np.asarray([current[k] for k in ('roll','pitch','yaw')], np.float64)
    goal = np.asarray([future[k] for k in ('x','y','z')], np.float64)
    _require(np.isfinite(xyz).all() and np.isfinite(rpy).all() and np.isfinite(goal).all(), 'Nonfinite provided goal/current pose')
    return (Rotation.from_euler('xyz', rpy).as_matrix().T @ (goal-xyz))[:2].astype(np.float32)


@dataclass
class PreparedInput:
    inputs: dict[str, torch.Tensor]
    metadata: dict


class RawInputAdapter:
    """Cache geometry only, keyed by every calibration value; no image features.

    Exactly three current images are decoded. Recreate the training Q95 cache
    JPEG round-trip in memory before PIL resize to 512x256. Removing that JPEG
    step changes pixels and is deliberately not a production option here.
    """
    def __init__(self, status_mode='causal_selection', goal_mode='none', camera_workers=3):
        _require(status_mode in ('zero','causal_selection'), 'Unknown status mode')
        _require(goal_mode in ('none','selection'), 'Unknown goal mode')
        _require(camera_workers in (1,3), 'camera_workers must be 1 or 3')
        self.status_mode, self.goal_mode = status_mode, goal_mode
        self._geometry_key, self._geometry = None, None
        self._geometry_lock = Lock()
        self.camera_workers = camera_workers
        self._pool = ThreadPoolExecutor(max_workers=3,thread_name_prefix='etri_camera') if camera_workers==3 else None
        self._closed = False

    def close(self):
        """Release camera workers after callers have finished their requests."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def _prepare_camera(g, raw_bytes):
        # This worker owns all image buffers. Shared remap grids are read-only.
        rgb, provenance = cache_compatible_image(raw_bytes, g)
        rgb = rgb.resize(IMAGE_WH, Image.Resampling.BILINEAR)
        value = (np.asarray(rgb,np.float32)/255. - MEAN)/STD
        return value.transpose(2,0,1).copy(), provenance

    def prepare_records(self, calibration_rows: Sequence[Mapping], pose_rows: Sequence[Mapping],
                        image_loader: Callable[[str,int],bytes]) -> PreparedInput:
        calibration_rows, pose_rows = list(calibration_rows), list(pose_rows)
        _require(not self._closed, 'RawInputAdapter is closed')
        _pose_rows(pose_rows)
        raw_key = json.dumps(calibration_rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        key = hashlib.sha256(raw_key).hexdigest()
        with self._geometry_lock:
            if key != self._geometry_key:
                all_geometry = {g.name:g for g in build_camera_geometry(calibration_rows)}
                self._geometry = tuple(all_geometry[name] for name in CAMERAS)
                self._geometry_key = key
            geometry = self._geometry
        # Read in official camera order on the caller thread: arbitrary image
        # loaders are not required to be thread-safe. Only pixel work is parallel.
        raw_images = [image_loader(g.name,0) for g in geometry]
        pending = [self._pool.submit(self._prepare_camera,g,raw) for g,raw in zip(geometry,raw_images)] if self._pool is not None else None
        status = np.zeros(8,np.float32)
        status5 = None
        if self.status_mode == 'causal_selection':
            status5 = causal_status_from_records(pose_rows)
            status[4:] = status5[:4]
        images, matrices, camera_metadata = [], [], {}
        for i,g in enumerate(geometry):
            value,provenance = pending[i].result() if pending is not None else self._prepare_camera(g,raw_images[i])
            images.append(torch.from_numpy(value))
            matrix = g.lidar2img.copy()
            matrix[0] *= IMAGE_WH[0]/768.
            matrix[1] *= IMAGE_WH[1]/432.
            matrices.append(matrix)
            camera_metadata[g.name] = {**provenance, 'crop_xy':list(g.crop_xy),
                'K_cache':g.cached_intrinsic.tolist(), 'raw_wh':list(g.raw_wh)}
        inputs = {'images':torch.stack(images)[None], 'lidar2img':torch.from_numpy(np.stack(matrices))[None],
                  'image_hw':torch.tensor([[256.,512.]],dtype=torch.float32),
                  'status':torch.from_numpy(status)[None]}
        if self.goal_mode == 'selection':
            inputs['goal_xy'] = torch.from_numpy(provided_goal_from_records(pose_rows))[None]
        _require(all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in inputs.values()), 'Nonfinite prepared inputs')
        metadata = {'input_contract':'sparsedrivev2-etri-current3-cache-equivalent-v1',
            'camera_order':list(CAMERAS), 'image_wh':list(IMAGE_WH), 'raw_image_reads':3,
            'calibration_value_sha256':key, 'camera_geometry':camera_metadata,
            'status_mode':self.status_mode, 'goal_mode':self.goal_mode,
            'status5':None if status5 is None else status5.tolist(),
            'pose_times':'nominal -1.0..0.0s, dt=.1; no timestamp column',
            'geometry_cache':'calibration-derived remap grids only; keyed by full calibration values',
            'camera_workers':self.camera_workers,
            'image_feature_cache':False, 'labels_read':False}
        return PreparedInput(inputs,metadata)

    def prepare_clip(self, clip_dir) -> PreparedInput:
        import pyarrow.parquet as pq
        directory = Path(clip_dir)
        calibration = directory/'calibration.parquet'
        poses = directory/'ego_pose.parquet'
        cal_rows = pq.read_table(calibration,columns=list(CALIBRATION_COLUMNS)).to_pylist()
        pose_rows = pq.read_table(poses,columns=list(POSE_COLUMNS)).to_pylist()
        result = self.prepare_records(cal_rows,pose_rows,
            lambda camera,frame: (directory/camera/f'frame_{frame}.jpg').read_bytes())
        result.metadata['source_sha256'] = {'calibration.parquet':file_sha256(calibration),
                                          'ego_pose.parquet':file_sha256(poses)}
        return result


class FrozenBankDriver:
    """Explicitly selected, strictly loaded trained checkpoint; raw in -> 6x2 XY.

    Initialization reads only the checkpoint, fixed bank and optional public
    initializer for provenance. Inference uses only RawInputAdapter inputs.
    This entry point does not submit predictions or write a leaderboard JSON.
    """
    def __init__(self,model,provenance,status_mode,goal_mode,device='cpu',precision='fp32'):
        _require(precision in ('fp32','bf16'), 'Unsupported precision')
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.provenance = provenance
        self.precision = precision
        self.adapter = RawInputAdapter(status_mode,goal_mode)

    @classmethod
    def from_training_checkpoint(cls, checkpoint, *, expected_sha256, bank_path,
                                 public_checkpoint=None,device='cpu',precision=None,backend=None,
                                 relative_head_path=None,relative_head_sha256=None,
                                 relative_cache_manifest_path=None,export_receipt_path=None,
                                 relative_export_receipt_path=None):
        checkpoint = Path(checkpoint)
        # Hash and load the same open inode even if a training process atomically
        # replaces a path named last.pth between those operations.
        with checkpoint.open('rb') as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                digest.update(chunk)
            observed = digest.hexdigest()
            _require(isinstance(expected_sha256,str) and len(expected_sha256)==64 and observed==expected_sha256,
                     'Selected checkpoint SHA256 mismatch')
            stream.seek(0)
            payload = torch.load(stream,map_location='cpu',weights_only=True)
        export=None
        if export_receipt_path is not None:
            try:from .inference_export import validate_export_receipt
            except ImportError:from inference_export import validate_export_receipt
            export=validate_export_receipt(payload,observed,export_receipt_path,'model')
        lineage_sha=observed if export is None else export['source_checkpoint_sha256']
        manifest = payload['manifest']; settings = manifest['arguments']
        _require(int(payload['step'])>0, 'A trained checkpoint is required')
        _require(manifest['public_checkpoint_sha256']==PUBLIC_SHA256, 'Unrecognized public initializer provenance')
        _require(manifest['score_mode']=='imitation', 'Unsupported checkpoint scoring contract')
        goal_mode = settings.get('goal_mode','none'); status_mode = settings['status_mode']
        _require(goal_mode in ('none','selection') and status_mode in ('zero','causal_selection'), 'Unknown checkpoint input modes')
        bank_sha = file_sha256(bank_path)
        _require(bank_sha==manifest['bank_sha256'], 'Fixed bank SHA256 mismatch')
        bank = load_etri_bank(bank_path)
        sources = {}
        required = ['public_model.py'] + (['goal_selector.py'] if goal_mode=='selection' else [])
        for name in required:
            relative = 'experiments/sparsedrivev2_20260910/'+name
            expected = manifest['source_sha256'].get(relative)
            actual = file_sha256(Path(__file__).with_name(name))
            _require(expected==actual, f'Trained runtime source mismatch: {name}')
            sources[relative] = actual
        if public_checkpoint is not None:
            _require(file_sha256(public_checkpoint)==PUBLIC_SHA256, 'Public initializer file hash mismatch')
        backend = backend or ('native' if torch.device(device).type=='cuda' else 'grid')
        base = PublicSparseDriveV2(bank,backend=backend,score_mode='imitation',mask_invalid_candidates=True)
        model = GoalConditionedSelector(base) if goal_mode=='selection' else base
        prefix = 'base.' if goal_mode=='selection' else ''
        state = payload['model']
        for name in BANK_NAMES:
            _require(torch.equal(bank[name],state[prefix+'_trajectory_head.'+name]),f'Embedded bank differs: {name}')
        model.load_state_dict(state,strict=True)
        provenance = {'checkpoint_sha256':observed,'checkpoint_step':int(payload['step']),
            'bank_sha256':bank_sha,'public_checkpoint_sha256':PUBLIC_SHA256,
            'public_initializer_file_rechecked':public_checkpoint is not None,
            'source_sha256':sources,'deployment_sha256':file_sha256(__file__),
            'strict_state_dict_load':True,'checkpoint_weights_only':True,'bank_rows_immutable':True,
            'status_mode':status_mode,'goal_mode':goal_mode,
            'path_filter':[128,20],'velocity_filter':[64,10],
            'output':'six cumulative current rear-axle ego XY metres at .5..3s; no cumsum'}
        if export is not None:
            provenance['export']=export
            provenance['source_checkpoint_sha256']=lineage_sha
        if relative_head_path is not None:
            _require(relative_head_sha256 is not None,'Explicit selected relative-head SHA256 is required')
            try:
                from .relative_artifact import load_relative_artifact
                from .relative_selector import CandidateRelativeSelector
            except ImportError:
                from relative_artifact import load_relative_artifact
                from relative_selector import CandidateRelativeSelector
            head,cache,receipt=load_relative_artifact(relative_head_path,
                expected_sha256=relative_head_sha256,cache_manifest_path=relative_cache_manifest_path,
                base_checkpoint_sha256=lineage_sha,bank_sha256=bank_sha,base_goal_mode=goal_mode,
                export_receipt_path=relative_export_receipt_path)
            model=CandidateRelativeSelector(model,base_goal_mode=goal_mode,freeze_base=True)
            model.relative_head.load_state_dict(head.state_dict(),strict=True)
            provenance['relative_selector']=receipt
            # Head features always use provided goal, including a goal-free base.
            # The wrapper forwards it into the base only if base_goal_mode permits.
            provenance['base_goal_mode']=goal_mode
            goal_mode='selection'
            provenance['goal_mode']=goal_mode
        else:
            _require(relative_head_sha256 is None and relative_cache_manifest_path is None and relative_export_receipt_path is None,
                     'Relative provenance options require a relative head')
        return cls(model,provenance,status_mode,goal_mode,device=device,
                   precision=precision or settings.get('precision','fp32'))

    @torch.inference_mode()
    def predict_prepared(self,prepared:PreparedInput,return_details=False):
        expected = {'images','lidar2img','image_hw','status'}
        if self.adapter.goal_mode=='selection':expected.add('goal_xy')
        _require(set(prepared.inputs)==expected,'Unexpected model inputs')
        _require(prepared.metadata['status_mode']==self.adapter.status_mode and
                 prepared.metadata['goal_mode']==self.adapter.goal_mode,'Prepared input mode mismatch')
        values = {k:v.to(self.device) for k,v in prepared.inputs.items()}
        with torch.autocast(device_type=self.device.type,dtype=torch.bfloat16,enabled=self.precision=='bf16'):
            out = self.model(**values)
        bank = self.model._trajectory_head.traj_vocab.flatten(0,1)
        candidate = bank[out['candidate_ids'],:6,:2]
        selected = bank[out['selected_candidate_id'],:6,:2]
        _require(torch.equal(candidate,out['candidate_xy']) and torch.equal(selected,out['trajectory']),
                 'Model output is not a fixed bank row')
        trajectory = out['trajectory'].detach().float().cpu().numpy()
        _require(trajectory.shape==(1,6,2) and np.isfinite(trajectory).all(), 'Invalid official trajectory shape/value')
        result = trajectory[0].copy()
        if return_details:
            return {'trajectory':result,'candidate_id':int(out['selected_candidate_id'][0]),
                    'input_metadata':prepared.metadata,'provenance':self.provenance}
        return result

    def predict_clip(self,clip_dir,return_details=False):
        return self.predict_prepared(self.adapter.prepare_clip(clip_dir),return_details)

    def predict_records(self,calibration_rows,pose_rows,image_loader,return_details=False):
        prepared = self.adapter.prepare_records(calibration_rows,pose_rows,image_loader)
        return self.predict_prepared(prepared,return_details)

    __call__ = predict_clip
