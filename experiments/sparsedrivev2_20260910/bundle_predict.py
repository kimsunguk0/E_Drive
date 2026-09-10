"""One official-shaped raw clip -> six cumulative XY points, with strict hashes."""
import argparse,hashlib,json,os
from pathlib import Path

# Per-process startup configuration before importing OpenCV, not a global setter.
os.environ.setdefault('OPENCV_FOR_THREADS_NUM','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip');p.add_argument('--output')
    p.add_argument('--device',default='cuda:0');p.add_argument('--precision',choices=('bf16','fp32'))
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args()
    if not a.verify_only and (a.clip is None or a.output is None):p.error('--clip and --output are required for prediction')
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'artifact_manifest.json').read_text())
    for relative,record in manifest['files'].items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or sha(path)!=record['sha256']:raise ValueError('Bundle artifact changed: '+relative)
    import torch
    torch.set_num_threads(4)
    from experiments.sparsedrivev2_20260910.deployment import FrozenBankDriver
    artifact=manifest['selected']
    driver=FrozenBankDriver.from_training_checkpoint(root/artifact['base_path'],
        expected_sha256=artifact['base_sha256'],bank_path=root/artifact['bank_path'],
        export_receipt_path=root/artifact['base_export_receipt'],
        relative_head_path=root/artifact['head_path'],relative_head_sha256=artifact['head_sha256'],
        relative_export_receipt_path=root/artifact['head_export_receipt'],
        relative_cache_manifest_path=root/artifact['cache_manifest_path'],
        device=a.device,precision=a.precision or ('bf16' if a.device.startswith('cuda') else 'fp32'))
    result={'verified':True,'provenance':driver.provenance,'submission_written':False}
    if not a.verify_only:
        prediction=driver.predict_clip(a.clip,return_details=True)
        result.update(trajectory=prediction['trajectory'].tolist(),candidate_id=prediction['candidate_id'],
                      output_contract='six cumulative current rear-axle ego XY metres at .5..3s, no cumsum')
        Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    driver.adapter.close()


if __name__=='__main__':main()
