"""Record design-check provenance and verify the last mask fix on train fixtures."""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_design as check
import numpy as np
import design

ROOT = check.ROOT
REPORT = ROOT/'reports/a2_design_preflight_20260921'
RUNTIME = ROOT/'work_dirs/a2_design_preflight_20260921'


def main():
    comparisons = []
    spec = importlib.util.spec_from_file_location('previous_design', RUNTIME/'check2/source/design.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    train, _ = check.datasets()
    for index in check.INDICES:
        item = train[index]
        scene, frame = item['scenario'], int(item['frame'])
        meta = json.loads((train.supervision_root/(scene+'.json')).read_text())
        path, = [Path(p) for p in meta['sources'] if p.endswith('/annotation/map.parquet')]
        frames, _, poses, _, lines, _ = check.read_scene_metadata(path.parent.parent)
        pos = {int(f): i for i, f in enumerate(frames)}[frame]
        local = [check.transform_points(line, np.linalg.inv(poses[pos])) for line in lines]
        args = local, check.grid_centers(), item['lane_valid'].numpy()[0]
        old, new = previous.geometry_targets(*args), design.geometry_targets(*args)
        fields = {}
        for key in ('offset', 'axis', 'offset_valid', 'axis_valid'):
            assert np.array_equal(old[key], new[key]), (index, key)
            fields[key] = hashlib.sha256(new[key].tobytes()).hexdigest()
        comparisons.append({'train_index': index, 'all_four_arrays_exactly_equal': True, 'sha256': fields})
    for attempt in ('check1', 'check2'):
        destination = REPORT/attempt/'source'
        destination.mkdir(exist_ok=True)
        for path in (RUNTIME/attempt/'source').glob('*.py'):
            shutil.copyfile(path, destination/path.name)
    destination = REPORT/'check3/source'
    destination.mkdir(exist_ok=True)
    for name in ('design.py', 'check_design.py'):
        shutil.copyfile(HERE/name, destination/name)
    exports = []
    for name in ('split_prototype.pth', 'fixtures.pth', 'reload.json'):
        path = RUNTIME/'check2'/name
        exports.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': check.sha_file(path)})
    for path in sorted(REPORT.glob('check*/train_geometry_overlay.png')):
        exports.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': check.sha_file(path)})
    receipt = {
        'scope': 'design preflight only; three fixed DEV-train rows; no optimizer updates',
        'attempts': {
            'check1': 'failed whole-model deepcopy hook ownership; all initial records retained',
            'check2': 'passed real-weight CUDA FP32/BF16, gradient routes, unchanged state, fresh-process strict export',
            'check3': 'passed CPU geometry fix for order-independent three-way ambiguity; no GPU rerun'},
        'final_geometry_targets_equal_GPU_checked_targets': comparisons,
        'SplitRead_class_source_changed_after_GPU_check': False,
        'runtime_artifacts_not_committed': exports,
        'not_measured': ['new DEV score', 'new server score', 'SplitRead RTX4090 latency/FLOPs',
                         'production optimizer groups and sample stream', 'training-time geometry head removal export'],
    }
    import ast
    old_tree = ast.parse((RUNTIME/'check2/source/design.py').read_text())
    new_tree = ast.parse((HERE/'design.py').read_text())
    def class_ast(tree, name):
        return ast.dump(next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name))
    assert class_ast(old_tree, 'SplitReadPlanner') == class_ast(new_tree, 'SplitReadPlanner')
    assert class_ast(old_tree, 'LaneGeometryHead') == class_ast(new_tree, 'LaneGeometryHead')
    check.atomic(REPORT/'verification_receipt.json', receipt)
    inventory = []
    for path in sorted(list(HERE.glob('*.py')) + list(REPORT.rglob('*'))):
        if path.is_file() and '__pycache__' not in str(path) and path.name != 'run_artifact_index.json':
            inventory.append({'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                              'sha256': check.sha_file(path)})
    check.atomic(REPORT/'run_artifact_index.json', {'source_commit': 'b3e967a0c692db8319ac29d7821f0bf3308ca542',
        'parent_checkpoint_sha256': check.PARENT_SHA, 'files': inventory})
    print(json.dumps({'final_targets_equal_check2': True, 'source_graph_unchanged': True,
                      'indexed_files': len(inventory)}, indent=2))


if __name__ == '__main__':
    main()
