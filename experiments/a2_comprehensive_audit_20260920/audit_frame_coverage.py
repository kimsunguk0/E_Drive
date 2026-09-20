"""Trace images actually consumed by the A2 adapter vs causal status fit times."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pyarrow.parquet as pq
ROOT=Path('/NHNHOME/data/sukim/adcl');REPORT=ROOT/'reports/a2_comprehensive_audit_20260920'
for p in (ROOT,ROOT/'scripts',ROOT/'experiments/md_r0_reset_20260914',ROOT/'experiments/md_a2_nominal_mh4_20260918',ROOT/'experiments/md_shared_dynamics_20260917',ROOT/'experiments/md_a2_deploy_status_20260918'):
    sys.path.insert(0,str(p))
from nominal_status import status5_from_clip_records
from models import motiondrive_v2_inputs as adapter
import mr_deploy

def main():
    fixture=json.loads((ROOT/'reports/motiondrive_v2_deploy_fixture_train8_manifest.json').read_text())['clips'][0]
    clip=ROOT/'data/etri/motiondrive_v2/deploy_fixture_train8'/fixture['clip_id']
    calibration=pq.read_table(clip/'calibration.parquet',columns=list(adapter.CALIBRATION_COLUMNS)).to_pylist()
    poses=pq.read_table(clip/'ego_pose.parquet',columns=list(adapter.POSE_COLUMNS)).to_pylist()
    image_calls=[]
    def image(camera,frame):
        image_calls.append((camera,int(frame)));return (clip/camera/f'frame_{frame}.jpg').read_bytes()
    prepared=mr_deploy.prepare_mr_clip_from_records(calibration,poses,image,detail='native')
    status,diag=status5_from_clip_records(poses)
    pose_frames=[int(round(t*10)) for t in diag['fit_relative_times']]
    image_frames=sorted(set(f for _,f in image_calls));missing=sorted(set(pose_frames)-set(image_frames))
    out={'fixture':str(clip),'image_getter_calls':image_calls,'unique_image_frame_indices':image_frames,'status_fit_pose_frame_indices':pose_frames,'pose_fit_frames_without_consumed_image':missing,'status_fit_count':diag['fit_count'],'input_shapes':{k:list(v.shape) for k,v in prepared.inputs.items()},'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'experiments/md_a2_deploy_status_20260918/nominal_status.py',ROOT/'experiments/a2_visual_teacher_20260919/infer_a2.py',Path(mr_deploy.__file__),ROOT/'OPEN_ISSUE.md']},'finding':'A2 query-only destination and frame-coverage obligation are separate checks. Preserved notice Q2 requires images for every used past-information frame; Q4 reiterates this. Q7/Q8 do not explicitly waive it. No conclusion of organizer approval or final disqualification is made.','optimizer_updates':0,'model_or_package_changed':False}
    (REPORT/'input_frame_coverage.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
