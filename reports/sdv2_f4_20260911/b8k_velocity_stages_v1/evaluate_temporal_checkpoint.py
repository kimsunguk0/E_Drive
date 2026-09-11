"""Strict frozen-source B1 evaluation of temporal SDV2 on original tune1998 only.

CPU: --audit-only verifies ancestry/source/strict tensor loading, no forward.
CPU: --self-test exercises guards with synthetic fixtures only.
GPU: CUDA_VISIBLE_DEVICES=<0|1|4> ... --gpu <same> --checkpoint last.pth
     --output NEW_DIRECTORY [--limit N] [--profile]
All learned model forwards, including history/perception/goal-final selection,
are included in model-only profiling. Decode, I/O, H2D and metric calculation
are excluded. B200 measurements are not RTX4090 measurements.
"""
from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import types
import uuid

import numpy as np
import torch

PUBLIC_SHA = '330072b133981f77b0b18b669216ad50b211552a82ea45ac52d507ec9021a735'
SPLIT_SHA = 'f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936'
EGO_SHA = 'd35a69bb229b2d4736aff7dddefd3522ddbd346e112283e2ce58ce3986a023bd'
TUNE_SHA = '1a65ade632db6a835be84ea6e30e9cc028917b6e7db344897d529a9e2d251d88'
TRAIN_SHA = '75ebfad637566f0731d8989e8e8a18aa9ed7462384522d8e4880ad831f33f854'
PINNED = {
    'train_temporal_comparison.py': 'a31296105ebecca63a40204e6cb7d95b086599697ef586726cf29f2c604a70e6',
    'temporal_model.py': '95684e777e18981f1beb2cc9485e08da9301e3e1a73fb2990eaaa9cd7330382a',
    'temporal_data.py': '4fc75a92b503d472a0fcb18189cbcedce6dd4c2c722219e04c0c0fa4d09f672b',
    'public_model.py': '2c7da9bc1fee9216513a606816c6f4e53d714812f0c4c33fe3ae7678a2b513c7',
    'relative_selector.py': '7fa75415a40691f555cca27ab9fe88331725177f19887c01a0e887121e2ac59f',
    'data.py': '54bcd88b1c2567d8aab3337fd82f5e6be50a4950e51957ba1df2908185f1eddb',
    'losses.py': 'f21c779742e0018b56d0b55123d5fbbfe1474127d8f56279221f28fd003fff17',
    'train.py': 'be221f0574a562e8449bffd90a6dd7a9f817b25159b006270de3f61b4cc9e08d',
    'goal_selector.py': 'b302b6db5e497ce26fb8fa7ecc817a1f2e1984a1c7dc21a1ed65fd94e16d581d',
}
INPUT_KEYS = {'images', 'lidar2img', 'image_hw', 'history_images', 'time_offsets'}
WEIGHTS = np.asarray([11,11,5,5,2,2],np.float64)/36.


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def row_sha(rows):
    return hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes()).hexdigest()


def tensor_sha(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def json_write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temporary.replace(path)


def validate_sources(root, recorded, pinned=PINNED):
    root = Path(root).resolve()
    require(set(pinned).issubset(recorded), 'Incomplete frozen Python source closure')
    for name, expected in pinned.items():
        require(recorded[name] == expected, f'Unsupported runtime revision: {name}')
    native = [name for name in recorded if name.startswith('native_ops/')]
    require(len(native) >= 2, 'Missing native source closure')
    for name, digest in recorded.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe source path')
        path = (root/relative).resolve()
        require(path.is_relative_to(root) and path.is_file(), f'Frozen source missing: {name}')
        require(sha(path) == digest, f'Frozen source hash mismatch: {name}')
    return root


@contextmanager
def isolated_runtime(root, module_names=None):
    """Private package and per-module import resolver; no global data/train swap.

    Qualified modules remain registered while DataLoader workers exist, permitting
    inspect/pickle to find the correct class source. Existing top-level modules
    are untouched. Imported bytecode is compiled directly from verified .py files.
    """
    root = Path(root).resolve()
    names = set(module_names or (Path(name).stem for name in PINNED))
    package_name = '_temporal_frozen_'+uuid.uuid4().hex
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    loaded = {}
    original_import = builtins.__import__

    def local_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level in (0,1) and name in names:
            return load(name)
        return original_import(name,globals,locals,fromlist,level)

    def load(name):
        if name in loaded:
            return loaded[name]
        require(name in names,'Import outside recorded Python closure')
        qualified = package_name+'.'+name
        path = root/(name+'.py')
        spec = importlib.util.spec_from_file_location(qualified,path)
        module = importlib.util.module_from_spec(spec)
        module.__dict__['__builtins__'] = {**vars(builtins),'__import__':local_import}
        loaded[name] = module
        sys.modules[qualified] = module
        setattr(package,name,module)
        exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
        return module

    try:
        if module_names is None:
            for name in ('data','public_model','relative_selector','goal_selector','losses','train',
                         'temporal_model','temporal_data','train_temporal_comparison'):
                load(name)
        else:
            for name in module_names:
                load(name)
        yield types.SimpleNamespace(**loaded)
    finally:
        for key in list(sys.modules):
            if key == package_name or key.startswith(package_name+'.'):
                sys.modules.pop(key,None)


def find_worktree(checkpoint, explicit=None):
    candidates = [Path(explicit).resolve()] if explicit else Path(checkpoint).resolve().parents
    for root in candidates:
        if (root/'third_party/SparseDriveV2').is_dir() and (root/'experiments/sparsedrivev2_20260910/public_model.py').is_file():
            return root
    raise RuntimeError('Cannot resolve original worktree; pass --worktree')


def locate(value, worktree, override=None):
    path = Path(override or value)
    return path.resolve() if path.is_absolute() else (worktree/path).resolve()


def load_internal_payload(path):
    # Internal training results contain NumPy numeric/string scalar dictionary
    # values. Permit these constructors only, without enabling general pickle.
    dtype_names=('float16','float32','float64','int8','int16','int32','int64',
                 'uint8','uint16','uint32','uint64','bool','U','S')
    safe=[np.core.multiarray.scalar,np.dtype]+list({type(np.dtype(x)) for x in dtype_names})
    with torch.serialization.safe_globals(safe):
        return torch.load(path,map_location='cpu',weights_only=True,mmap=True)


def inspect_checkpoint(path, *, worktree=None, bank=None, public_checkpoint=None, base=None,
                       source_root=None, allow_canary=False):
    checkpoint = Path(path).resolve()
    first_hash = sha(checkpoint)
    payload = load_internal_payload(checkpoint)
    require(first_hash == sha(checkpoint),'Checkpoint changed while loading')
    require(isinstance(payload,dict) and {'model','manifest','step'}.issubset(payload),'Malformed temporal checkpoint')
    payload.pop('optimizer',None)
    manifest = payload['manifest']
    require(manifest['schema'] == 'sdv2_temporal_comparison_v1','Wrong checkpoint schema')
    a = manifest['arguments']
    require(payload['step'] == a['steps'],'Only the predeclared terminal checkpoint is evaluable')
    # Longer-trained bases are the subject here; the pin to the 2,000-step
    # primaries is relaxed to any full (non-limited) run.
    require(allow_canary or a['eval_limit']==0,'Non-primary run needs --limit canary mode')
    require(type(a['common_status']) is bool and a['history_mode'] in ('repeat','real'),'Invalid arm contract')
    require(not a['common_status'] or a['history_mode']=='real','Unrecognized common-status/repeat arm')
    require(manifest['planner_status']=='constant zero, no raw or predicted status','Planner status contract changed')
    require(manifest['relative_head_status']=='constant zero, no raw or predicted status','Relative status contract changed')
    require(manifest['goal_route']=='only relative scoring of completed original bank candidates','Goal route changed')
    disk_manifest = checkpoint.parent/'manifest.json'
    require(canonical(json.loads(disk_manifest.read_text()))==canonical(manifest),'Checkpoint/disk manifest mismatch')
    root = find_worktree(checkpoint,worktree)
    captured = validate_sources(source_root or checkpoint.parent/'source',manifest['source_sha256'])
    public_path = locate(a['checkpoint'],root,public_checkpoint)
    bank_path = locate(a['bank'],root,bank)
    require(manifest['public_checkpoint_sha256']==PUBLIC_SHA and sha(public_path)==PUBLIC_SHA,'Public checkpoint pin mismatch')
    require(sha(bank_path)==manifest['bank_sha256'],'Bank hash mismatch')
    sidecar = bank_path.with_suffix(bank_path.suffix+'.json')
    require(sha(sidecar)==manifest['initialization_audit']['bank_sidecar_sha256'],'Bank sidecar changed')
    split = Path(manifest['validation']['split_manifest']).resolve()
    ego = Path(manifest['validation']['ego_cache']).resolve()
    require(sha(split)==SPLIT_SHA and sha(ego)==EGO_SHA,'Primary split/ego identity pin mismatch')
    m = json.loads(split.read_text())
    # Deserialize identity arrays only; future targets are first materialized by the evaluator dataset.
    with np.load(ego,allow_pickle=False) as z:
        scenes = z['scenarios'].astype(str)[z['scen_idx'].astype(np.int64)]
        frames = z['frame'].astype(np.int64)
    train_rows = np.flatnonzero(np.isin(scenes,m['splits']['train']) & (frames>=30))
    tune_rows = np.flatnonzero(np.isin(scenes,m['splits']['tune']) & (frames>=30) & (frames%5==0))
    require(len(train_rows)==54810 and row_sha(train_rows)==TRAIN_SHA,'Original training rows changed')
    require(len(tune_rows)==1998 and row_sha(tune_rows)==TUNE_SHA,'Original tune1998 rows changed')
    train_sessions = {m['scene_to_session'][s] for s in np.unique(scenes[train_rows])}
    tune_sessions = {m['scene_to_session'][s] for s in np.unique(scenes[tune_rows])}
    require(not train_sessions & tune_sessions,'Train/tune session overlap')
    require(manifest['train']['allowed_rows_sha256']==TRAIN_SHA,'Manifest training population mismatch')
    require(manifest['validation']['allowed_rows_sha256']==TUNE_SHA,'Manifest tune population mismatch')
    with np.load(bank_path,allow_pickle=False) as z:
        require(np.array_equal(z['train_rows'],train_rows),'Bank used a different fitting population')
        metadata = json.loads(str(z['metadata_json']))
        require(metadata['partition']=='train' and metadata['split_sha256']==SPLIT_SHA,'Bank fitting split mismatch')
        require(metadata['train_rows_sha256']==TRAIN_SHA and str(z['train_rows_sha256'])==TRAIN_SHA,'Bank row digest mismatch')
        require(not set(metadata['train_scenes']) & set(np.unique(scenes[tune_rows])),'Bank includes tune scenes')
    receipt = dict(checkpoint=str(checkpoint),checkpoint_sha256=first_hash,step=int(payload['step']),
        manifest_sha256=sha(disk_manifest),source_root=str(captured),source_sha256=manifest['source_sha256'],
        worktree=str(root),public_checkpoint=str(public_path),public_checkpoint_sha256=PUBLIC_SHA,
        bank=str(bank_path),bank_sha256=manifest['bank_sha256'],split_sha256=SPLIT_SHA,ego_sha256=EGO_SHA,
        population='original_tune1998',tune_rows_sha256=TUNE_SHA,train_rows_sha256=TRAIN_SHA,
        history_mode=a['history_mode'],common_status=a['common_status'],future_targets_deserialized_in_inspection=False)
    return types.SimpleNamespace(checkpoint=checkpoint,payload=payload,manifest=manifest,worktree=root,
        source=captured,public=public_path,bank=bank_path,split=split,ego=ego,rows=tune_rows,
        base=Path(base or a['base']).resolve(),receipt=receipt)


def native_runtime_location(plan, public_module):
    """Reuse frozen native loader code, with its path root explicitly verified.

    Flat training snapshots cannot satisfy public loader's parents[2] convention.
    Only its __file__ global is rebased; Python code and CUDA source bytes stay
    identical. Existing task-local extension cache can therefore be reused.
    """
    root = plan.worktree
    virtual = root/'experiments/sparsedrivev2_20260910/public_model.py'
    require(sha(virtual)==plan.manifest['source_sha256']['public_model.py'],'Native loader reference source changed')
    checked = {}
    for name,digest in plan.manifest['source_sha256'].items():
        if name.startswith('native_ops/'):
            path = root/'third_party/SparseDriveV2/navsim/agents/sparsedrive/ops/src'/Path(name).name
            require(sha(path)==digest,f'Native source changed: {name}')
            checked[str(path)] = digest
    original = public_module.load_native_extension
    namespace = dict(original.__globals__)
    namespace['__file__'] = str(virtual)
    namespace['_NATIVE'] = None
    rebound = types.FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    rebound.__kwdefaults__ = original.__kwdefaults__
    public_module.load_native_extension = rebound
    return {'strategy':'same frozen loader code; verified __file__ root only',
            'path_root':str(root),'native_sources':checked}


def strict_load(plan, runtime):
    public,coverage = runtime.public_model.PublicSparseDriveV2.from_public_checkpoint(
        plan.public,bank_path=plan.bank,backend='native',score_mode='imitation')
    require(coverage['reused_tensor_count']==396 and coverage['reused_learned_parameter_numel']==41808827,
            'Public tensor coverage differs from recorded model')
    temporal = runtime.temporal_model.TemporalPerceptionModel(public)
    model = runtime.train_temporal_comparison.FinalGoalTemporalSelector(temporal)
    bank_keys = ['base.base._trajectory_head.'+k for k in ('path_vocab','vel_vocab','traj_vocab','traj_mask')]
    constant_keys = bank_keys+['base.nominal_time_offsets','base.temporal.status_scale','base.perception.points']
    initial = model.state_dict()
    expected = {name:tensor_sha(initial[name]) for name in constant_keys}
    del initial
    model.load_state_dict(plan.payload['model'],strict=True)
    for name,tensor in model.state_dict().items():
        require(not tensor.is_floating_point() or bool(torch.isfinite(tensor).all()),f'Nonfinite checkpoint tensor: {name}')
        if name in expected:
            require(tensor_sha(tensor)==expected[name],f'Checkpoint mutated fixed bank/geometry buffer: {name}')
    require(not any(isinstance(mod,runtime.goal_selector.GoalConditionedSelector) for mod in model.modules()),
            'Planner goal-conditioning wrapper is forbidden in this experiment')
    require(model.base_goal_mode=='none' and model.freeze_base is False,'Unexpected final goal wrapper configuration')
    require('goal_xy' not in inspect.signature(temporal.forward).parameters,'Goal reached temporal planner signature')
    model.eval()
    plan.receipt.update(strict_state_load=True,fixed_buffer_sha256=expected,
        public_coverage=coverage,native_loader=native_runtime_location(plan,runtime.public_model))
    return model


def build_dataset(plan,runtime,limit=0):
    recorded = plan.manifest['validation']
    base = runtime.data.PlanDataset(base=plan.base,split_manifest=plan.split,ego_cache=plan.ego,
        split='tune',stride=5,augment=False,status_mode='zero',goal_mode='selection')
    require(np.array_equal(base.rows,plan.rows),'Frozen dataset changed tune population/order')
    base_provenance=base.provenance()
    for key in ('split_sha256','ego_cache_sha256','calibration_sha256','camera_order','image_wh','normalization'):
        require(base_provenance[key]==recorded[key],f'Dataset input contract changed: {key}')
    source = recorded['temporal_data']
    dataset = runtime.temporal_data.TemporalPlanDataset(base,history_mode=plan.manifest['arguments']['history_mode'],
        auxiliary=False,causal_status_root=Path(source['causal_status']['path']).parent)
    actual = dataset.provenance()['temporal_data']
    for key in ('source_sha256','base_source_sha256','history_mode','frame_lags','time_offsets','pose_alignment_used'):
        require(actual[key]==source[key],f'Temporal input contract changed: {key}')
    actual_causal=dict(actual['causal_status']); recorded_causal=dict(source['causal_status'])
    require(recorded_causal.pop('rows_sha256')==recorded['rows_sha256'],'Recorded causal row identity mismatch')
    require(actual_causal.pop('rows_sha256')==TUNE_SHA and actual_causal==recorded_causal,'Causal input contract changed')
    # Slicing only the outer Subset preserves complete dataset provenance and masks.
    if limit:
        require(0<limit<=1998,'Canary --limit must be in 1..1998')
        subset = torch.utils.data.Subset(dataset,list(range(limit)))
        subset.rows = plan.rows[:limit]
        return subset,dataset.provenance()
    return dataset,dataset.provenance()


def input_tensors(batch,runtime,common_status,device):
    selected = runtime.temporal_data.model_inputs(batch,common_status=common_status)
    expected = INPUT_KEYS | ({'perception_status'} if common_status else set())
    require(set(selected)==expected,'Unexpected temporal inference input')
    require('goal_xy' in batch,'Provided final-selection goal absent')
    selected = {**selected,'goal_xy':batch['goal_xy']}
    require(all(isinstance(value,torch.Tensor) for value in selected.values()),'Model input is not a tensor')
    # Labels, metadata and row IDs are never transferred with model input tensors.
    return {key:value.to(device,non_blocking=True) for key,value in selected.items()}


@contextmanager
def route_monitor(model,common_status):
    counts = {'public_calls':0,'temporal_calls':0,'relative_calls':0}
    temporal,public = model.base,model.base.base
    def public_pre(_module,args,kwargs):
        require(not args and set(kwargs)=={'images','lidar2img','image_hw','status'},'Public planner input route changed')
        status = kwargs['status']
        require(status.shape==(len(kwargs['images']),8) and bool((status==0).all()),'Nonzero status reached public planner')
        counts['public_calls']+=1
    def temporal_pre(_module,args,kwargs):
        expected=INPUT_KEYS|({'perception_status'} if common_status else set())
        require(not args and set(kwargs)==expected,'Goal/status bypass reached temporal planner')
        counts['temporal_calls']+=1
    def relative_pre(_module,args):
        features=args[0]
        require(features.shape[-1]==32 and bool((features[...,22:26]==0).all()),'Raw status reached relative-head feature slots')
        counts['relative_calls']+=1
    handles=[public.register_forward_pre_hook(public_pre,with_kwargs=True),
        temporal.register_forward_pre_hook(temporal_pre,with_kwargs=True),model.relative_head.register_forward_pre_hook(relative_pre)]
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def verify_rows(model,out):
    valid=out['candidate_valid'].bool()
    require(valid.ndim==2 and bool(valid.any(-1).all()),'Invalid candidate mask')
    require(bool(torch.isfinite(out['scores'][valid]).all()),'Nonfinite valid scores')
    ids=out['candidate_ids'].long()
    bank=model._trajectory_head.traj_vocab.flatten(0,1)
    require(bool(((ids>=0)&(ids<len(bank))).all()),'Candidate ID outside bank')
    require(torch.equal(out['candidate_xy'],bank[ids,:6,:2]),'Candidate coordinates changed')
    index=out['scores'].masked_fill(~valid,-torch.inf).argmax(-1)
    selected=ids.gather(1,index[:,None]).squeeze(1)
    require(torch.equal(selected,out['selected_candidate_id']),'Selected ID differs from valid argmax')
    require(torch.equal(out['trajectory'],bank[selected,:6,:2]),'Selected coordinates changed')


@torch.inference_mode()
def sample_route_audit(model,inputs,common_status):
    with route_monitor(model,common_status) as counts:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            first=model(**inputs)
            other_goal={**inputs,'goal_xy':inputs['goal_xy']+inputs['goal_xy'].new_tensor([10.,-3.])}
            second=model(**other_goal)
        for out in (first,second):
            verify_rows(model,out)
        for key in ('candidate_xy','candidate_ids','base_scores','aux_state','aux_occ','aux_lane'):
            require(torch.equal(first[key],second[key]),f'Final goal changed an upstream output: {key}')
        state_test='not applicable: common status input absent'
        if common_status:
            altered={**inputs,'perception_status':inputs['perception_status']+inputs['perception_status'].new_tensor([1.,.1,.1,.1])}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                third=model(**altered)
            verify_rows(model,third)
            require(torch.equal(first['aux_state'],third['aux_state']),'Aux state depends on raw common status')
            state_test='bitwise aux_state invariant under [1,.1,.1,.1] status perturbation'
    return {'goal_changes_only_final_selection':True,'aux_state_independence':state_test,
            'observed_zero_planner_and_relative_status':True,'calls':counts}


def check_gpu(requested):
    require(requested in (0,1,4),'GPU must be explicitly assigned 0, 1, or 4')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(requested),'CUDA_VISIBLE_DEVICES must equal explicit --gpu')
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    entries=[line.split(',',2) for line in query.strip().splitlines()]
    info=next((parts for parts in entries if int(parts[0])==requested),None)
    require(info is not None,'Requested physical GPU not found')
    uuid_value=info[1].strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
    others=[line for line in apps.strip().splitlines() if line.startswith(uuid_value+',') and int(line.split(',')[1])!=os.getpid()]
    require(not others,'Assigned GPU is occupied; do not overlap running training')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'Expose exactly one CUDA device')
    return {'physical_index':requested,'uuid':uuid_value,'name':info[2].strip(),'other_compute_processes_at_launch':others}


@torch.inference_mode()
def evaluate(model,dataset,runtime,common_status,workers=4):
    loader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False,num_workers=workers,pin_memory=True)
    chunks={}; first_inputs=None
    def add(key,value):
        chunks.setdefault(key,[]).append(value)
    with route_monitor(model,common_status) as counts:
        for i,batch in enumerate(loader):
            inputs=input_tensors(batch,runtime,common_status,'cuda:0')
            if first_inputs is None:
                first_inputs={key:value.clone() for key,value in inputs.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                out=model(**inputs)
            verify_rows(model,out)
            pred=out['trajectory'].float().cpu().numpy()
            gt=batch['gt_plan'].float().numpy()
            delta=pred.astype(np.float64)-gt.astype(np.float64)
            point=np.linalg.norm(delta,axis=-1)
            candidate=out['candidate_xy'].float().cpu().numpy()
            cost=(np.linalg.norm(candidate.astype(np.float64)-gt[:,None].astype(np.float64),axis=-1)*WEIGHTS).sum(-1)
            valid=out['candidate_valid'].cpu().numpy()
            score=out['base_scores'].float().masked_fill(~out['candidate_valid'],-torch.inf)
            base_idx=score.argmax(-1)
            base_pred=out['candidate_xy'][torch.arange(len(pred),device=score.device),base_idx].float().cpu().numpy()
            for key,value in {'rows':batch['row'].numpy(),'pred':pred,'gt':gt,'error_xy':delta,'point_l2':point,
                'd3':(point*WEIGHTS).sum(-1),'shortlist_oracle':np.where(valid,cost,np.inf).min(-1),
                'candidate_id':out['selected_candidate_id'].cpu().numpy(),
                'base_pred':base_pred,'base_d3':(np.linalg.norm(base_pred.astype(np.float64)-gt,axis=-1)*WEIGHTS).sum(-1),
                'state_pred':out['aux_state'].cpu().numpy(),'state_target':batch['state_target'].numpy(),
                'session':np.asarray(batch['session']),'scenario':np.asarray(batch['scenario'])}.items():
                add(key,value)
            if (i+1)%200==0:
                print(json.dumps({'evaluation_rows':i+1}),flush=True)
    arrays={key:np.concatenate(value) for key,value in chunks.items()}
    require(np.array_equal(arrays['rows'],dataset.rows),'Evaluation row order changed')
    error=arrays['d3']; oracle=arrays['shortlist_oracle']
    result={'n':len(error),'official_d3':float(error.mean()),'shortlist_oracle_d3':float(oracle.mean()),
        'selection_regret':float((error-oracle).mean()),'base_same_forward_d3':float(arrays['base_d3'].mean()),
        'point_l2':arrays['point_l2'].mean(0).tolist(),'three_second_l2':float(arrays['point_l2'][:,-1].mean()),
        'state_mae_vx_vy_ax_ay':np.abs(arrays['state_pred']-arrays['state_target']).mean(0).tolist(),
        'session_d3':{s:float(error[arrays['session']==s].mean()) for s in np.unique(arrays['session'])},
        'batch_size':1,'precision':'bf16_base_fp32_relative_head','metric_arithmetic':'FP64 Euclidean distances and D3 weights',
        'all_candidate_and_selected_bank_rows_exact':True,'route_checks':counts}
    return arrays,result,first_inputs


@torch.inference_mode()
def profile_model(model,inputs,warmup=10,iterations=50):
    require(warmup>=1 and iterations>=30,'Profiling requires warmup>=1 and at least 30 iterations')
    for _ in range(warmup):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            model(**inputs)
    torch.cuda.synchronize()
    wall=[]; gpu=[]
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iterations):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); before=time.perf_counter(); start.record()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model(**inputs)
        end.record(); torch.cuda.synchronize()
        wall.append((time.perf_counter()-before)*1000.)
        gpu.append(start.elapsed_time(end))
    verify_rows(model,out)
    summary=lambda xs:{'mean':float(np.mean(xs)),'p50':float(np.percentile(xs,50)),'p95':float(np.percentile(xs,95))}
    return {'warmup':warmup,'iterations':iterations,'batch_size':1,'wall_ms':summary(wall),'cuda_event_ms':summary(gpu),
        'wall_samples_ms':wall,'cuda_event_samples_ms':gpu,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'scope':'full temporal model forward: all current/history encodings, perception auxiliaries, base DFA and final goal selector',
        'excluded':'raw/cache decode, disk/tar I/O, input assembly, H2D, GT/metrics, evaluator route hooks and bank checks',
        'inputs_contain_gt':False,'rtx4090_measured':False,'cached_input_one_row_repeated':True}


def compare_reference(arrays,path,training_batch_size=8):
    with np.load(path,allow_pickle=False) as ref:
        n=len(arrays['rows'])
        require(np.array_equal(arrays['rows'],ref['rows'][:n]),'Training reference row order differs')
        ids=ref['candidate_id'][:n]; pred=ref['pred'][:n]; old=ref['d3'][:n]
        return {'path':str(Path(path).resolve()),'sha256':sha(path),'training_batch_size':int(training_batch_size),
            'xy_bitwise_equal':bool(np.array_equal(arrays['pred'],pred)),
            'ids_bitwise_equal':bool(np.array_equal(arrays['candidate_id'],ids)),
            'changed_selected_id_count':int(np.count_nonzero(arrays['candidate_id']!=ids)),
            'max_absolute_xy_delta':float(np.max(np.abs(arrays['pred'].astype(np.float64)-pred))),
            'd3_delta_mean':float(arrays['d3'].mean()-old.astype(np.float64).mean()),
            'metric_note':'B1 model arithmetic and FP64 metric may differ from B8/BF16 and training FP32 metric; actual predictions are preserved'}


def self_test():
    import tempfile
    import unittest
    class Guards(unittest.TestCase):
        def test_private_import_collision_and_cleanup(self):
            marker=types.ModuleType('data'); marker.VALUE='outside'
            previous=sys.modules.get('data'); sys.modules['data']=marker
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    p=Path(tmp); (p/'data.py').write_text('VALUE="frozen"\n')
                    (p/'use.py').write_text('from data import VALUE\n')
                    before=set(sys.modules)
                    with isolated_runtime(p,['use','data']) as runtime:
                        self.assertEqual(runtime.use.VALUE,'frozen')
                        self.assertIs(sys.modules['data'],marker)
                    self.assertEqual(set(sys.modules),before)
            finally:
                if previous is None: sys.modules.pop('data',None)
                else: sys.modules['data']=previous
        def test_import_failure_cleanup(self):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp); (p/'bad.py').write_text('raise RuntimeError("fixture")\n')
                before={x for x in sys.modules if x.startswith('_temporal_frozen_')}
                with self.assertRaisesRegex(RuntimeError,'fixture'):
                    with isolated_runtime(p,['bad']): pass
                self.assertEqual(before,{x for x in sys.modules if x.startswith('_temporal_frozen_')})
        def test_source_hash_and_traversal(self):
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp); (p/'one.py').write_text('X=1\n'); (p/'native_ops').mkdir()
                for name in ('a','b'): (p/'native_ops'/name).write_text(name)
                record={name:sha(p/name) for name in ('one.py','native_ops/a','native_ops/b')}
                validate_sources(p,record,{'one.py':record['one.py']})
                with self.assertRaisesRegex(RuntimeError,'Unsafe'):
                    validate_sources(p,{**record,'../escape':'bad'},{'one.py':record['one.py']})
                (p/'one.py').write_text('X=2\n')
                with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                    validate_sources(p,record,{'one.py':record['one.py']})
        def test_internal_numpy_scalar_safe_load(self):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.pth'
                torch.save({'model':{'x':torch.tensor([1.])},'metric':np.float64(.2),
                            'session':np.str_('fixture'),'step':np.int64(2)},path)
                data=load_internal_payload(path)
                self.assertEqual(data['step'],2)
                self.assertEqual(data['session'],'fixture')
                self.assertTrue(torch.equal(data['model']['x'],torch.tensor([1.])))

        def test_tensor_and_rows_digests(self):
            self.assertEqual(row_sha([1,2]),row_sha(np.asarray([1,2],np.int32)))
            self.assertNotEqual(tensor_sha(torch.tensor([0.,1.])),tensor_sha(torch.tensor([0.,2.])))
        def test_input_whitelist(self):
            runtime=types.SimpleNamespace(temporal_data=types.SimpleNamespace(model_inputs=lambda b,common_status: {k:b[k] for k in INPUT_KEYS}))
            batch={k:torch.zeros(1) for k in INPUT_KEYS|{'goal_xy','gt_plan','state_target'}}
            actual=input_tensors(batch,runtime,False,'cpu')
            self.assertEqual(set(actual),INPUT_KEYS|{'goal_xy'})
            runtime.temporal_data.model_inputs=lambda b,common_status: b
            with self.assertRaisesRegex(RuntimeError,'Unexpected'):
                input_tensors(batch,runtime,False,'cpu')
        def test_bank_identity_and_selection_guard(self):
            bank=torch.arange(2*8*3).reshape(1,2,8,3).float()
            model=types.SimpleNamespace(_trajectory_head=types.SimpleNamespace(traj_vocab=bank))
            out={'candidate_ids':torch.tensor([[0,1]]),'candidate_xy':bank[0,:,:6,:2][None],
                 'candidate_valid':torch.ones(1,2,dtype=torch.bool),'scores':torch.tensor([[0.,1.]]),
                 'selected_candidate_id':torch.tensor([1]),'trajectory':bank[0,1,:6,:2][None]}
            verify_rows(model,out)
            broken={**out,'trajectory':out['trajectory']+.01}
            with self.assertRaisesRegex(RuntimeError,'coordinates changed'): verify_rows(model,broken)
            broken={**out,'candidate_valid':torch.tensor([[True,False]])}
            with self.assertRaisesRegex(RuntimeError,'argmax'): verify_rows(model,broken)
        def test_gpu_guard_no_cuda_probe(self):
            with self.assertRaisesRegex(RuntimeError,'explicitly assigned'): check_gpu(2)
        def test_b1_reference_differences_reported(self):
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'eval.npz'
                np.savez(path,rows=np.asarray([1]),pred=np.zeros((1,6,2)),candidate_id=np.asarray([2]),d3=np.asarray([.2]))
                arrays={'rows':np.asarray([1]),'pred':np.ones((1,6,2)),'candidate_id':np.asarray([3]),'d3':np.asarray([.3])}
                report=compare_reference(arrays,path)
                self.assertEqual(report['changed_selected_id_count'],1)
                self.assertFalse(report['xy_bitwise_equal'])
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Guards))
    require(result.wasSuccessful(),'CPU self-tests failed')


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--checkpoint'); p.add_argument('--output'); p.add_argument('--worktree')
    p.add_argument('--source-root'); p.add_argument('--bank'); p.add_argument('--public-checkpoint'); p.add_argument('--base')
    p.add_argument('--gpu',type=int,choices=(0,1,4)); p.add_argument('--limit',type=int,default=0)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--audit-only',action='store_true')
    p.add_argument('--profile',action='store_true'); p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--profile-iterations',type=int,default=50); p.add_argument('--reference-predictions')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test(); return
    require(a.checkpoint and a.output,'--checkpoint and --output are required')
    require(a.workers>=0 and 0<=a.limit<=1998,'Invalid worker/limit count')
    require(not a.audit_only or not a.profile,'CPU audit cannot profile CUDA')
    gpu=None if a.audit_only else check_gpu(a.gpu)
    output=Path(a.output).resolve(); require(not output.exists(),'Output directory already exists')
    torch.set_num_threads(4)
    plan=inspect_checkpoint(a.checkpoint,worktree=a.worktree,bank=a.bank,public_checkpoint=a.public_checkpoint,
        base=a.base,source_root=a.source_root,allow_canary=bool(a.limit))
    output.mkdir(parents=True)
    shutil.copy2(__file__,output/Path(__file__).name)
    shutil.copytree(plan.source,output/'frozen_source')
    receipt={**plan.receipt,'evaluator_sha256':sha(__file__),'audit_only':a.audit_only,
        'gpu':gpu,'canary_limit':a.limit,'batch_size':1,'precision':'bf16_base_fp32_relative_head',
        'runtime_torch':str(torch.__version__),'training_torch':plan.manifest['torch']}
    started=time.time()
    try:
        with isolated_runtime(plan.source) as runtime:
            model=strict_load(plan,runtime)
            receipt.update(plan.receipt)
            if a.audit_only:
                result={'status':'PASS_CPU_STRICT_LOAD','forward_performed':False,'future_targets_deserialized':False}
            else:
                model.cuda()
                dataset,provenance=build_dataset(plan,runtime,a.limit)
                arrays,result,sample=evaluate(model,dataset,runtime,plan.manifest['arguments']['common_status'],a.workers)
                # Preserve all actual B1 results before reference comparison/audits can fail.
                np.savez_compressed(output/'predictions.npz',**arrays)
                result.update(status='completed',step=plan.payload['step'],canary=bool(a.limit))
                json_write(output/'result.json',result)
                receipt.update(dataset_provenance=provenance,predictions_sha256=sha(output/'predictions.npz'),
                    route_sample_audit=sample_route_audit(model,sample,plan.manifest['arguments']['common_status']))
                reference=Path(a.reference_predictions) if a.reference_predictions else plan.checkpoint.parent/f"eval_{plan.payload['step']:06d}.npz"
                if reference.is_file():
                    receipt['training_reference_comparison']=compare_reference(arrays,reference,plan.manifest['arguments']['eval_batch'])
                if a.profile:
                    receipt['profile']=profile_model(model,sample,a.warmup,a.profile_iterations)
            receipt['elapsed_seconds']=time.time()-started
            json_write(output/'receipt.json',receipt); json_write(output/'result.json',result)
            print(json.dumps(result,allow_nan=False),flush=True)
    except BaseException:
        receipt.update(status='failed',traceback=traceback.format_exc(),elapsed_seconds=time.time()-started)
        json_write(output/'failure.json',receipt)
        raise


if __name__=='__main__':
    main()
