# Frozen checkpoint evaluation and late parameter average

`evaluate_checkpoint.py` restores the checkpoint's captured source snapshot, checks public initialization and bank ancestry, and strictly loads raw or goal-wrapped state keys. Every predicted trajectory must equal an immutable bank row. It also verifies all candidate rows and masks.

The `--approved-candidate` option means an **internal frozen experimental selection record written by the root agent after tune selection**. It does not request additional user permission. The user already authorized training/evaluation; this guard prevents accidental held-label tuning and checkpoint/bank lineage mistakes. The evaluator never creates that record automatically.

## Tune and CPU audit

Run from the experiment worktree with `/NHNHOME/data/sukim/adcl/env/venv/bin/python`. Supply a new output directory.

```sh
python experiments/sparsedrivev2_20260910/evaluate_checkpoint.py \
  --checkpoint /absolute/path/to/frozen_checkpoint.pth \
  --output /absolute/path/to/new_evaluation
```

Expose exactly one assigned GPU with `CUDA_VISIBLE_DEVICES=0`, `1`, or `4` for actual evaluation. Default evaluation batch/precision come from the training manifest; batch 8/bf16 preserves the original evaluation setup. `--reference-predictions path/to/eval_002000.npz` requires bitwise equality of the saved rows/predictions/D3/candidate IDs and any common auxiliary arrays. This is the actual GPU parity check; CPU synthetic parity alone does not establish GPU parity.

`--audit-only` instead performs source/identity provenance and strict model loading entirely on CPU. It never creates a label dataset or runs model inference. No GPU was used for the implementation tests.

`--dump-coarse` additionally saves candidate IDs/scores/validity, final path/velocity IDs, and every coarse stage's IDs/scores. Each coarse stage is recorded **before** that stage's pruning; the next stage IDs show the previous pruning. Final IDs show the last pruning. The default output remains small.

`predictions.npz` always contains row IDs, scenario/session names, prediction, D3, shortlist oracle, selected bank row ID, per-point L2 and signed XY error. Metrics include prefix ADE, 3-second endpoint L2, marginal weighted absolute X/Y errors and a separately identified squared-error share.

## Confirmation12 guard

Pass `--population confirmation12`, explicit `--train-rows .../confirmation12_train.npy`, `--eval-rows .../confirmation12_held_rows.npy` and the internal `--approved-candidate candidate.json`. Every source checkpoint, including all sources of an average, must match the train171 ancestry and bank before the held dataset opens.

The internal record has this schema; root fills hashes from the final frozen artifacts:

```json
{
  "schema": "sparsedrivev2_confirmation_candidate_v1",
  "approved": true,
  "frozen": true,
  "population": "confirmation12",
  "checkpoint_sha256": "FINAL_CHECKPOINT_SHA256",
  "bank_sha256": "TRAIN171_BANK_SHA256",
  "train_rows_sha256": "701e7ea7b76acd6b0d99845a0f400c9b65a5c470131d2d3b6324192e09c28a35",
  "eval_rows_sha256": "2809febcd692040870821e2575062722226b4f58152e53b7eed27d9cc5aaa3b7",
  "split_sha256": "f4e0f30c5c243e03f007a8da53fdc3dfa85a6b21b96c53723d35f94d0fbc7936"
}
```

The receipt is not a way to override incompatible ancestry: a train203 checkpoint/bank still fails even with a matching file hash. Reserve136 is not an enabled population.

## Predeclared late average

```sh
python experiments/sparsedrivev2_20260910/average_checkpoints.py \
  --protocol reports/sparsedrivev2_20260910/late_average_protocol.json \
  --checkpoint /absolute/path/to/run_step001500.pth \
  --checkpoint /absolute/path/to/run_step001750.pth \
  --checkpoint /absolute/path/to/run_step002000.pth \
  --output /absolute/path/to/new_average.pth
```

Source receipts may be either `file.pth.json` or the archiver's actual `file.json`; their SHA and embedded steps are mandatory. Source manifests must be exactly equal. Learned parameters accumulate in float64, divide by three, then cast back once. Every persistent buffer, including bank and BatchNorm statistics, must be bitwise identical and is copied unchanged. Output creation never overwrites. The output has `result=None`, no optimizer, and a `.pth.json` hash sidecar.

Evaluation independently recomputes the entire average from its three archived sources before opening labels. Preserve the preregistration file, postprocessor version, source files/receipts, and original runtime snapshot: all are hash checked. The averaged result is supplementary to the unchanged terminal-step comparison; no evaluation score is invented for it.

Reusable APIs are `inspect_checkpoint`, `recorded_runtime`, `load_verified_model`, `build_dataset`, `evaluate_model`, and `check_reference`. The first three can validate/load a frozen model without a label dataset. They support both original and `base.*` goal wrapper keys through the recorded architecture.
