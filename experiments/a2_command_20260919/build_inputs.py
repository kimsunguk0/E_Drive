"""Freeze exact raw-command/frame joins and exercise the test-input parser."""
import json
from collections import Counter
import numpy as np
from command_common import *
import command_data as data

def counts(values):
    return {label: int((values == i).sum()) for i, label in enumerate(data.LABELS)}

def main():
    assert not CACHE.exists(), 'Never overwrite a frozen input cache'
    train, tune = nominal.raw_datasets(False, 1)
    rows = np.sort(np.concatenate([train.rows, tune.rows]))
    assert len(np.unique(rows)) == len(rows)
    # Read only identity and provided command fields from the old cache for an audit.
    with np.load('/tmp/pm97/data/etri/ego_cache.npz', allow_pickle=False) as z:
        scene_names = z['scenarios'][z['scen_idx'][rows]]
        frames = z['frame'][rows]
        old_ids = z['meta'][rows]
        assert list(z['meta_labels']) == list(data.LABELS)
    command = np.full(len(rows), -1, np.int8)
    raw_sources = {}
    for scene in sorted(set(scene_names)):
        mask = scene_names == scene
        directory = ROOT/'data/etri/meta_train'/str(scene)/'meta'
        command[mask] = data.read_scene_commands(directory, frames[mask])
        raw_sources[str(scene)] = {name: sha(directory/name) for name in ('timestamps.parquet', 'command.parquet')}
    assert np.array_equal(old_ids, command), 'Raw supplied command differs from previous timestamp-mapped metadata'
    test_files = sorted(Path('/tmp/etri_test').glob('*/command.parquet'))
    assert len(test_files) == 1125
    test_ids = np.asarray([data.read_clip_command(p).argmax() for p in test_files])
    CACHE.mkdir(parents=True)
    np.savez_compressed(CACHE/'provided_commands.npz', row=rows, command=command)
    splits = {}
    for dataset in (train, tune):
        ids = command[np.searchsorted(rows, dataset.rows)]
        splits[dataset.split] = dict(rows=len(ids), rows_sha256=legacy.rows_sha(dataset.rows), counts=counts(ids))
    manifest = dict(schema_version=1, labels=list(data.LABELS), mirror=list(data.MIRROR),
        producer_sha256=sha(data.__file__), build_script_sha256=sha(__file__),
        split_manifest_sha256=sha(legacy.SPLIT), allowed_splits=['train', 'tune'],
        artifact=dict(path=str(CACHE/'provided_commands.npz'), sha256=sha(CACHE/'provided_commands.npz'), rows=len(rows)),
        source_policy='Current frame exact timestamp join to raw command.parquet; no pose, GT XY or vad_cmd derivation',
        old_meta_exact_agreement=True, splits=splits, raw_source_sha256=raw_sources,
        test_input_check=dict(clips=len(test_files), counts=counts(test_ids),
            only_provided_inputs=True, future_gt_used=False, parser='read_clip_command',
            source_sha256={p.parent.name: sha(p) for p in test_files}))
    (CACHE/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    REPORT.mkdir(parents=True, exist_ok=True)
    summary={k:v for k,v in manifest.items() if k!='raw_source_sha256'}
    summary['test_input_check']={k:v for k,v in manifest['test_input_check'].items() if k!='source_sha256'}
    summary['cache_manifest']=dict(path=str(CACHE/'manifest.json'), sha256=sha(CACHE/'manifest.json'))
    (REPORT/'INPUT_POLICY.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()
