"""Create an optimizer-stripped, self-contained primary CE inference bundle."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
import torch
try:
    from .inference_export import state_fingerprint,validate_export_receipt
except ImportError:
    from inference_export import state_fingerprint,validate_export_receipt

BASE_SHA='bfe29df9f510c3ff76f36f41688a56faaadcc5c07071063504f4e244fcb29aa1'
HEAD_SHA='5ec5bc6286610d35f7239c0f0cb41caee3c5fd1cc0b033cb384c04bbb3aed374'
BANK_SHA='4aff9b40696f91383bfbec377f388e6209d619fa509e51a2019b19af977b5e04'
CACHE_SHA='e9455b8e6577ff2a9d488dfc6780129df5b336d247adeeaabded246180b9e298'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def export(source,destination,expected,state_key):
    with source.open('rb') as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
        if h.hexdigest()!=expected:raise ValueError('Selected source checkpoint changed')
        f.seek(0);payload=torch.load(f,map_location='cpu',weights_only=True)
    if 'optimizer' not in payload:raise ValueError('Expected original training checkpoint with optimizer')
    original_keys=sorted(payload)
    fingerprint=state_fingerprint(payload[state_key])
    payload.pop('optimizer')
    torch.save(payload,destination)
    reloaded=torch.load(destination,map_location='cpu',weights_only=True)
    exported=state_fingerprint(reloaded[state_key])
    if fingerprint!=exported:raise ValueError('Export changed state tensors')
    for key in payload:
        if key!=state_key and payload[key]!=reloaded[key]:raise ValueError('Export changed metadata: '+key)
    receipt={'schema':'sparsedrivev2_optimizer_stripped_export_v1','source_checkpoint_path':str(source.resolve()),
        'source_checkpoint_sha256':expected,'artifact_sha256':sha(destination),'state_key':state_key,
        'removed_keys':['optimizer'],'source_payload_keys':original_keys,'exported_payload_keys':sorted(payload),
        'source_state_fingerprint':fingerprint,'exported_state_fingerprint':exported,
        'original_bytes':source.stat().st_size,'exported_bytes':destination.stat().st_size,
        'non_state_metadata_preserved_exactly':True}
    receipt_path=destination.with_suffix('.export.json')
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    validate_export_receipt(reloaded,receipt['artifact_sha256'],receipt_path,state_key)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    task=Path(__file__).resolve().parent;wt=task.parents[1]
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
    artifact=out/'artifacts';artifact.mkdir()
    base=wt/'work_dirs/sparsedrivev2_20260910/dense2000_none_s0_v1/last.pth'
    head=wt/'work_dirs/sparsedrivev2_20260910/relative2000_soft_ce_s0_v1/last.pth'
    bank=wt/'cache/sparsedrivev2_20260910/bank/p1024_v1024_native100m_progress6.npz'
    cache=wt/'cache/sparsedrivev2_20260910/features_primary_none2000_v1/manifest.json'
    if sha(bank)!=BANK_SHA or sha(cache)!=CACHE_SHA:raise ValueError('Selected bank/cache provenance changed')
    receipts={'base':export(base,artifact/'base.pth',BASE_SHA,'model'),
              'head':export(head,artifact/'head.pth',HEAD_SHA,'relative_head')}
    shutil.copy2(bank,artifact/'bank.npz');shutil.copy2(cache,artifact/'cache_manifest.json')
    source_paths=[]
    for name in ('public_model.py','goal_selector.py','relative_selector.py','relative_artifact.py','inference_export.py','deployment.py'):
        source_paths.append(Path('experiments/sparsedrivev2_20260910')/name)
    for name in ('__init__.py','motiondrive_v2_inputs.py','motiondrive_v2_input_contract.py','motiondrive_v2_temporal_contract.py'):
        source_paths.append(Path('models')/name)
    for name in ('deformable_aggregation.cpp','deformable_aggregation_cuda.cu'):
        source_paths.append(Path('third_party/SparseDriveV2/navsim/agents/sparsedrive/ops/src')/name)
    source_paths.append(Path('third_party/SparseDriveV2/LICENSE'))
    for relative in source_paths:
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(wt/relative,target)
    shutil.copy2(task/'bundle_predict.py',out/'predict_one.py')
    shutil.copy2(task/'build_deployment_bundle.py',out/'build_deployment_bundle.py')
    documentation=wt/'reports/sparsedrivev2_20260910/public_init'
    shutil.copy2(documentation/'deployment_environment_packages.json',out/'environment_packages.json')
    readme=task/'BUNDLE_README.md';shutil.copy2(readme,out/'README.md')
    (out/'requirements.txt').write_text('\n'.join(('torch==2.7.1+cu128','torchvision==0.22.1+cu128','timm==0.6.13',
        'numpy==1.26.4','scipy==1.15.3','opencv-python==4.8.1.78','pillow==12.2.0','pyarrow==24.0.0','ninja==1.13.0'))+'\n')
    files={str(path.relative_to(out)):{'sha256':sha(path),'bytes':path.stat().st_size} for path in sorted(out.rglob('*')) if path.is_file()}
    selected={'base_path':'artifacts/base.pth','base_sha256':receipts['base']['artifact_sha256'],
        'base_export_receipt':'artifacts/base.export.json','base_source_sha256':BASE_SHA,
        'head_path':'artifacts/head.pth','head_sha256':receipts['head']['artifact_sha256'],
        'head_export_receipt':'artifacts/head.export.json','head_source_sha256':HEAD_SHA,
        'bank_path':'artifacts/bank.npz','bank_sha256':BANK_SHA,'cache_manifest_path':'artifacts/cache_manifest.json',
        'cache_manifest_sha256':CACHE_SHA,'base_goal_mode':'none','head_goal_mode':'selection'}
    manifest={'schema':'sparsedrivev2_primary_ce_inference_bundle_v1','selected':selected,'files':files,
        'source_worktree':str(wt),'builder_sha256':sha(Path(__file__)),
        'runtime_training_data_required':False,'runtime_original_checkpoint_required':False,
        'native_extension':'source only; compile current-stream-patched copy for deployed GPU',
        'validated_development_arch':'sm100 B200','future_target_arch':'sm89 RTX4090, not yet measured',
        'primary_tune_b8_d3':.13074861450346642,'primary_tune_b1_d3':.1305221511370592,
        'hidden_test_performance_claimed':False,'external_submission_created':False}
    (out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'bundle':str(out),'manifest_sha256':sha(out/'artifact_manifest.json'),
        'file_count':len(files),'total_bytes':sum(x['bytes'] for x in files.values()),'selected':selected},indent=2))


if __name__=='__main__':main()
