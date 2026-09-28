#!/usr/bin/env python3
"""Assemble the evaluation release for the final ETRI E2E model (EXT-FULL-v7).

Sources are copied byte-identical (no path rewriting): inside the container the
code tree is mounted at the original repository path, so every source-hash pin
in the training scripts still holds. The only non-HEAD source is
temporal_model.py, restored to the exact version the stage-1/2 checkpoints were
trained with (commit ac62e26, sha256 pinned in their protocols).
"""
import hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT = Path('/NHNHOME/data/sukim/adcl')
OUT = ROOT / 'release/etri_e2e_final'
CODE = OUT / 'code'
TRACE = OUT / '_trace'

ENTRY = [
    'experiments/a2_final_push_20260923/ext_v6/build_ext_submission.py',   # inference (stage 3 model)
    'experiments/a2_final_push_20260923/ext_v6/train_ext.py',              # stage 3 training
    'experiments/a2_final_push_20260923/ext_v6/ext_model.py',
    'experiments/a2_final_push_20260923/pack_ext.sh',
    'experiments/a2_long_motion_20260922/train_long.py',                   # stage 2 training
    'experiments/a2_long_motion_20260922/verify_long.py',
    'experiments/a2_progress_full_20260921/train_full.py',                 # stage 1 training
    'experiments/a2_progress_full_20260921/verify_smoke.py',
    'experiments/a2_progress_full_20260921/build_full_cache.py',
    'experiments/md_r0_reset_20260914/package_submission.py',
    # data preparation (derived caches are shipped; these rebuild them)
    'scripts/etri_ego_cache.py', 'scripts/etri_label_5s.py', 'scripts/build_grouped_split_v2.py',
    'scripts/build_scene_supervision_v2.py', 'scripts/derive_motiondrive_v2_geometry.py',
    'scripts/trace_imports.py',
]
DATA = [  # ROOT-relative derived data read by training / validation
    'data/etri/motiondrive_v2/grouped_split_r0reset_tplus.json',
    'data/etri/motiondrive_v2/r0reset_tplus_geometry_v2',
    'data/etri/motiondrive_v2/a2_h4_status_full_20260921',
    'data/etri/motiondrive_v2/a2_h4_status_20260920',
    'data/etri/motiondrive_v2/a2_nominal_status_20260918',
    'data/etri/motiondrive_v2/deploy_fixture_train8',
    'reports/a2_final_push_20260923/views/L34.npz',
    'reports/a2_progress_h4_20260920/protocol_A2-H4-PROGRESS_s1.json',
    'reports/a2_long_motion_20260922/smoke/checks_L-FULL6.json',
    'work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/final_eval.json',
    'work_dirs/a2_progress_h4_20260920/A2-H4-PROGRESS-s1/experiment.json',
    'work_dirs/md_a2_scene_extensions_20260918/A2-QREFINE-NOM-s1/manifest.json',
]
CKPT = [  # ROOT-relative weights: final model + the training chain back to public init
    'work_dirs/a2_ext_select_20260923/EXT-FULL-v7/ckpt_step6344.pth',            # submitted (0.117067)
    'work_dirs/a2_long_motion_20260922/L-FULL6-s1/ckpt_step38070.pth',            # stage 2 output = stage 3 parent
    'work_dirs/a2_long_motion_20260922/L-FULL6-s1/experiment.json',
    'work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1/ckpt_step24931.pth',  # stage 1 output
    'work_dirs/a2_progress_full_20260921/A2-H4-PROGRESS-FULL-s1/experiment.json',
    'work_dirs/a2_motion_fresh_20260919/initializers/public_nuimages_qrefine_s1.pth',  # stage 1 initializer
    'ckpt/backbones/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth',  # public
]
PM97 = ['/tmp/pm97/data/etri/ego_cache.npz', '/tmp/pm97/data/etri/ego_cache_5s.npz']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(8 << 20), b''): h.update(c)
    return h.hexdigest()


def copy(rel, dst_root=CODE, src_root=ROOT):
    src, dst = src_root / rel, dst_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    (shutil.copytree(src, dst, dirs_exist_ok=True) if src.is_dir() else shutil.copy2(src, dst))


def main():
    assert not CODE.exists(), 'refuse to overwrite ' + str(CODE)
    mods = set(json.loads((TRACE / 'modules_union.json').read_text()))
    files = sorted(mods | set(ENTRY))
    for rel in files:
        copy(rel)
    # restore the exact temporal_model.py the checkpoints were trained with
    tm = 'experiments/a2_temporal_read_20260920/temporal_model.py'
    original = subprocess.check_output(['git', 'show', f'ac62e26:{tm}'], cwd=ROOT)
    (CODE / tm).write_bytes(original)
    assert hashlib.sha256(original).hexdigest() == 'dd01226e745e1530ceef2f79f9ae9c495064aede07d61089607667881e66b6a7'
    for p in (CODE / 'experiments').rglob('*'):   # namespace packages need nothing, but keep dirs importable
        pass
    for rel in DATA + CKPT:
        copy(rel)
    pm = OUT / 'data_pm97/data/etri'; pm.mkdir(parents=True, exist_ok=True)
    for p in PM97:
        shutil.copy2(p, pm / Path(p).name)
    # the code tree is its own git repository (training records `git rev-parse HEAD`)
    subprocess.check_call(['git', 'init', '-q'], cwd=CODE)
    subprocess.check_call(['git', 'add', '-A'], cwd=CODE)
    subprocess.check_call(['git', '-c', 'user.name=release', '-c', 'user.email=release@local',
                           'commit', '-q', '-m', 'ETRI E2E final release snapshot'], cwd=CODE)
    manifest = dict(
        source_repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        temporal_model_restored_from='ac62e26',
        code={rel: sha(CODE / rel) for rel in files},
        weights={rel: sha(CODE / rel) for rel in CKPT if rel.endswith('.pth')},
        data_pm97={Path(p).name: sha(pm / Path(p).name) for p in PM97})
    (OUT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(json.dumps(dict(code_files=len(files), weights=len(manifest['weights']),
                          size=subprocess.check_output(['du', '-sh', str(OUT)], text=True).split()[0])))


if __name__ == '__main__':
    main()
