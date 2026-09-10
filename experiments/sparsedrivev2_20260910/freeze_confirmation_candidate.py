"""Freeze the prechosen CE terminal for group-excluded confirmation.

This records the root experiment decision, not user permission. It never opens
held images or future labels. Existing evaluators independently enforce this
record before constructing the held dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

from evaluate_checkpoint import (inspect_checkpoint, read_rows, rows_sha,
    CONFIRM_TRAIN_SHA, CONFIRM_HELD_SHA, check_approval)
from relative_artifact import load_relative_artifact, require_confirmation_binding, file_sha, require


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--base', required=True)
    p.add_argument('--head', required=True)
    p.add_argument('--cache-manifest', required=True)
    p.add_argument('--train-rows', required=True)
    p.add_argument('--eval-rows', required=True)
    p.add_argument('--method-protocol', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    base, head = Path(a.base).resolve(), Path(a.head).resolve()
    require(base.name == head.name == 'last.pth', 'Only fixed terminal artifacts may be selected')
    for path in (base, head):
        result = json.loads((path.parent / 'result.json').read_text())
        require(result['status'] == 'completed', 'Training must be complete before freezing')
        require(result['terminal']['step'] == 2000, 'Both components must finish step2000')
    plan = inspect_checkpoint(base, population='tune', train_rows_path=a.train_rows)
    _, cache, receipt = load_relative_artifact(head, expected_sha256=file_sha(head),
        cache_manifest_path=a.cache_manifest, base_checkpoint_sha256=plan.receipt['checkpoint_sha256'],
        bank_sha256=plan.receipt['bank_sha256'], base_goal_mode='none')
    payload = torch.load(head, map_location='cpu', weights_only=True)
    arguments = payload['manifest']['arguments']
    settings = dict(objective='soft_ce', steps=2000, batch=128, seed=0, lr=.001,
        warmup=100, weight_decay=.0001, temperature=.1, eval_every=250)
    for key, value in settings.items():
        require(arguments[key] == value, 'Prechosen head setting differs: ' + key)
    require(plan.payload['step'] == receipt['relative_step'] == 2000, 'Wrong terminal step')
    require(plan.goal_mode == 'none', 'Prechosen base must omit coarse goal')
    require(rows_sha(read_rows(a.train_rows)) == CONFIRM_TRAIN_SHA, 'Wrong train171 rows')
    require(rows_sha(read_rows(a.eval_rows)) == CONFIRM_HELD_SHA, 'Wrong confirmation12 rows')
    protocol = json.loads(Path(a.method_protocol).read_text())
    require(protocol['chosen_objective'] == 'soft_ce' and protocol['held_results_observed'] is False,
        'A method decision made before held observation is required')
    record = dict(schema='sparsedrivev2_confirmation_candidate_v1', approved=True, frozen=True,
        population='confirmation12', decision_owner='root internal experiment decision',
        frozen_unix=time.time(), checkpoint_sha256=receipt['base_checkpoint_sha256'],
        bank_sha256=receipt['bank_sha256'], train_rows_sha256=CONFIRM_TRAIN_SHA,
        eval_rows_sha256=CONFIRM_HELD_SHA, split_sha256=plan.receipt['split_sha256'],
        relative_head_sha256=receipt['relative_checkpoint_sha256'],
        relative_source_sha256=receipt['relative_source_sha256'],
        relative_trainer_sha256=receipt['relative_trainer_sha256'],
        relative_cache_manifest_sha256=receipt['cache_manifest_sha256'],
        relative_objective='soft_ce', feature_goal_mode='selection', base_goal_mode='none',
        checkpoint_path=str(base), relative_head_path=str(head), bank_path=str(plan.bank),
        cache_manifest_path=str(Path(a.cache_manifest).resolve()),
        train_rows_path=str(Path(a.train_rows).resolve()), eval_rows_path=str(Path(a.eval_rows).resolve()),
        head_settings=settings, public_checkpoint_sha256=plan.receipt['public_checkpoint_sha256'],
        method_protocol_sha256=file_sha(a.method_protocol),
        method_protocol_path=str(Path(a.method_protocol).resolve()),
        base_source_sha256=plan.receipt['source_sha256'],
        freeze_source_sha256=file_sha(__file__),
        held_results_observed_before_freeze=False,
        interpretation='Group-excluded retraining confirmation; these sessions belonged to primary train203, not a historically untouched external test set.')
    check_approval(record, {'checkpoint_sha256': plan.receipt['checkpoint_sha256'],
        'bank_sha256': plan.receipt['bank_sha256'], 'train_rows_sha256': CONFIRM_TRAIN_SHA,
        'eval_rows_sha256': CONFIRM_HELD_SHA, 'split_sha256': plan.receipt['split_sha256']})
    require_confirmation_binding(record, receipt, cache)
    path = Path(a.output)
    with path.open('x') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'frozen_candidate': str(path.resolve()), 'sha256': file_sha(path),
        'head_sha256': receipt['relative_checkpoint_sha256'], 'held_labels_opened': False}, indent=2))


if __name__ == '__main__':
    main()
