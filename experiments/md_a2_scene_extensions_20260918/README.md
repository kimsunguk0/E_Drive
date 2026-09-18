# Independent A2 scene extensions — 2026-09-18

Two matched-budget DEV arms follow the completed BASE/MH4 comparison.
The primary record is `reports/md_a2_scene_extensions_20260918/README_KO.md`.

- `A2-SIDE-SCENE-NOM`: four additional historical left/right images in the shared scene.
- `A2-QREFINE-NOM`: one additional image-dependent query/read, with BASE observations.

Both reuse the completed BASE initializer, nominal provided-status producer, optimizer
recipe, train310/V0 split, seed1, batch16, microbatch8 and 20,554-update budget.
They are independent arms, not a combined SIDE+QREFINE model or terminal continuation.

`preflight.py` checks actual images, geometry, flip, initial function parity and gradient
connectivity on physical GPU3. `train_scene_extension.py --smoke` performs two updates
and a full 1,998-row V0 evaluation. `capture_records.py` archives those checks and the
completed A2 FULL checkpoint identity. `launch.py` starts only GPUs1 and2, once, after
preflight and smoke success; it refuses existing run directories or occupied devices.

Core modification: `SharedSceneEncoder._view_geometry` accepts optional historical
camera/pose indices. Its default front-only path preserves the prior computation.
The SIDE cameras are `[2,2,1,1]`; CONTROL pose indices are `[0,2,0,2]`, corresponding
to frame offsets `[1,5,1,5]`, not adjacent entries in CONTROL `(1,2,5,10)`.

QREFINE computes first image read S0, then q1=q0+MLP(q0,S0), then image read S1.
Its output is S0+W(S1), with bias-free W initialized to zero. The query MLP is
nonzero initialized so it receives gradients after W's first update. Neither
query nor raw status is added as a feature value. The refined scene is consumed
by occupancy, lane and planning. The native front motion/state/history path stays
independent of provided status and of SIDE's new images within a forward pass.
All shared parameters are jointly trained, so additional scene gradients can of
course change the shared backbone and future motion predictions across updates.

The existing `[0,W)` image-validity mask is not exactly symmetric under the pixel
reflection `u -> W-1-u` at the outer one-pixel strip. Preflight checks the interior
mirror coordinates and confines all mask differences to that strip. It does not
change this legacy convention mid-comparison.
