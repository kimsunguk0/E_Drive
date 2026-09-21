"""Export only inference sources, weights and two train fixtures to a portable folder."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,tarfile

ROOT=Path('/NHNHOME/data/sukim/adcl')
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()

def build(destination,checkpoint):
    destination=Path(destination);assert not destination.exists(),destination
    code=destination/'code';artifacts=destination/'artifacts'
    code.mkdir(parents=True);artifacts.mkdir()
    old=json.loads((ROOT/'reports/a2_full_submission_20260919/source_bundle.json').read_text())
    files=set(old['files'])
    files={p for p in files if not p.startswith('experiments/a2_visual_teacher_20260919/')}
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'models/motiondrive_v2').glob('*.py'))
    files.update([
        'experiments/md_a2_scene_extensions_20260918/scene_extensions.py',
        'experiments/a2_motion_fresh_20260919/motion_model.py',
        'experiments/a2_temporal_read_20260920/temporal_model.py',
        'experiments/a2_progress_h4_20260920/progress_model.py',
        'experiments/a2_progress_h4_20260920/h4_status.py',
        'experiments/a2_progress_full_20260921/infer_full.py',
        'experiments/a2_progress_full_20260921/Dockerfile'])
    manifest={}
    for name in sorted(files):
        original=ROOT/name;target=code/name;target.parent.mkdir(parents=True,exist_ok=True)
        content=original.read_text();rewritten=False
        if name.startswith('experiments/'):
            for literal in ("Path('/NHNHOME/data/sukim/adcl')",'Path("/NHNHOME/data/sukim/adcl")'):
                if literal in content:
                    content=content.replace(literal,'Path(__file__).resolve().parents[2]');rewritten=True
        target.write_text(content)
        manifest[name]={'repository_sha256':sha(original),'portable_sha256':sha(target),
                        'root_path_rewritten_without_graph_changes':rewritten}
    shutil.copy2(checkpoint,artifacts/'model.pth')
    clips=artifacts/'raw_clips';clips.mkdir()
    for name in ('fixture_000','fixture_001'):
        shutil.copytree(ROOT/'data/etri/motiondrive_v2/deploy_fixture_train8'/name,clips/name)
    readme='''# A2-H4-PROGRESS-FULL inference reproduction

`submission.zip` is the official prediction artifact in the parent package. This folder is reproduction material, not another submission.

Build the container from this directory:

```sh
docker build -f code/experiments/a2_progress_full_20260921/Dockerfile -t adcl-h4-progress:20260921 code
mkdir -p output
docker run --rm --gpus all -v "$PWD/code:/workspace/code:ro" -v "$PWD/artifacts:/workspace/artifacts:ro" -v "$PWD/output:/workspace/output" adcl-h4-progress:20260921 --checkpoint /workspace/artifacts/model.pth --clips-root /workspace/artifacts/raw_clips --output /workspace/output/predictions.json --require-full
```

The two included clips are training fixtures for correctness only. For evaluation replace `raw_clips` with the actual test clips. Output is absolute XY (6x2); never apply another cumsum. The teacher is absent. Provided status is fitted from five RGB-consumed pose times and conditions only the shared scene query. This is an input-path statement, not organizer approval.

CPU correctness: append `--cpu --precision fp32` and omit `--gpus all`. CPU results are not GPU latency. Whole GPU forward timing: append `--timing --expected-device "RTX 4090"`; this includes all image encoders but excludes raw preprocessing. B200 timing cannot establish RTX4090 latency.
'''
    (destination/'README.md').write_text(readme)
    record={'checkpoint_source':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'files':manifest,'root_rewrite':'experiment import roots only; no tensor/forward changes',
        'fixtures':'two training clips, not test accuracy','official_upload_performed':False}
    (destination/'source_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    return record

def archive(folder,output):
    assert not Path(output).exists()
    with tarfile.open(output,'w:gz') as tar:tar.add(folder,arcname=Path(folder).name)
    return {'path':str(output),'bytes':Path(output).stat().st_size,'sha256':sha(output)}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--destination',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--archive')
    a=ap.parse_args();result=build(a.destination,Path(a.checkpoint))
    if a.archive:result['archive']=archive(a.destination,a.archive)
    print(json.dumps({k:result[k] for k in ('checkpoint_sha256','source_commit','archive') if k in result},indent=2))
