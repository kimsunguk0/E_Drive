# A2 waypoint temporal-memory read

One user-authorized experiment, following the frozen planner readout review.

The original QREFINE scene and pooled-motion planner remain. Before the existing
XY head, six decoded waypoint queries additionally read the four original
time-labelled motion grids (`4 × 192 × 128`). The new four-head attention uses
FP32 query/memory LayerNorm and a zero-initialized output projection. No new image
pass, frame, supplied-status token, loss, or teacher is introduced.

The image-derived pair tokens are passed explicitly through forward return values;
no persistent temporal cache or hidden feature capture is used. Supplied status
still only conditions the common scene query. Decoded planner queries inherit
scene information, so the new reader is not described as goal-independent.

The matched control is completed `A2-FRESH-NUIM-s1` at 20,554 updates, PREFIX
0.158707259. Its exact public-trunk/random-nontrunk initializer, DEV310/V0 rows,
nominal producer, batch16/micro8, seed1, augmentation, original multi-task+LEN loss,
fixed BN, AdamW, LR and cosine schedule are reused. Added module construction does
not advance the control RNG stream. Logged row hashes are compared with control.

Primary evaluation: terminal vs terminal. Scheduled evaluations are 3426, 6852,
10278, 13704, 17130, 20554. A selected intermediate is explicitly marked as chosen
on reused V0. The longer FRESH continuation is a practical candidate, not this
arm's matched training-budget control.

`preflight.py` checks exact initial outputs/loss, parent tensor and RNG parity,
real planning gradients through the new reader, isolated gradients to all four
history slots, status/goal isolation of motion/state/history, shared scene
consumers, effective-batch normalization, strict reload and whole-forward cost.
All preflight mutations are disposable. A separate two-update smoke precedes the
real run, which restarts from the original public initializer.

Run on one of physical GPU0–3 after checking it is free. The existing `launch.py`
records PID, source, GPU, command and immutable run paths. `collect.py` collects
planned results; it never launches another training run. No weight averaging,
continuation, FULL fit or official submission is automatic.

Reports: `reports/a2_temporal_read_20260920/`. Large checkpoints and row predictions
stay in `work_dirs/a2_temporal_read_20260920/`; text reports include hashes/paths.
