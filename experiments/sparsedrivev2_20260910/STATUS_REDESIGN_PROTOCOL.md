# Status sensitivity and common temporal perception comparison

The active goal continues beyond the completed sensitivity diagnostic. GPU 0,
1 and 4 are the only allocated devices. Existing checkpoints and the original
TRAIN203/TUNE37 split remain immutable; reserve/held rows are not used here.

## Evidence driving the next experiment

The original frozen SDV2+relative CE model reproduces D3 0.1307486145 on all
1,998 tune rows (batch8). All three GPU shards reproduce the same predictions
and IDs bitwise. With the actual P7 b0 image-predicted status substituted into
both inputs, D3 becomes 1.170503 and the shortlist oracle 1.112857. Its paired
11-session bootstrap delta is +1.03975, CI [+0.83495,+1.20652]. Mean retained
original candidates are only 17.58/200. Vx bias ±0.1m/s alone yields D3
0.157206/0.160285; ±0.05 yields 0.136355/0.139645. These are measured frozen
interventions, not a precision threshold or a bound on a retrained student.

Full artifacts: reports/sparsedrivev2_status_20260910/full_shard{0,1,2}_v1 and
analysis_v3. The P7 input is the literal stored first four image-estimator
outputs, not a GT-corrected residual. P7 raw-time labels and SDV2 nominal
causal labels differ slightly and are documented in EMPIRICAL_STATUS_KO.md.

## Fixed three-arm comparison

| Arm | History pixels | Causal state entering common perception |
|---|---|---|
| A | repeat current front twice | none (zero constant) |
| B | front at -0.1 and -0.5 seconds | none (zero constant) |
| C | front at -0.1 and -0.5 seconds | vx,vy,ax,ay at image-fusion query only |

Every arm has the same architecture and public initialization, five image
encodings (three current, two history), temporal position/time encoding,
image-only state auxiliary head, shared image perception/occ/lane branches,
and new final relative score head. Spatial attention runs on P4 image tokens;
it has no access to candidates, planning queries, goal or future GT. A and B
differ only in history pixels. B and C differ only in the explicit common
perception condition. Current-image augmentation and data shuffle are matched.

The temporal projection and status-condition output start at zero. All public
learned tensors remain unchanged at construction. Initial planning outputs
should therefore agree across arms (state auxiliary predictions need not).
The image-only auxiliary state branch executes before raw-state conditioning;
its output is never fed into planning or final scoring.

The original SDV2 planner receives an internally constructed zero8 tensor.
Raw status and predicted status are absent from its direct input and the final
relative score features. The common fused image features are shared by the
actual current occupancy/lane supervision and planning. Goal is consumed only
by `rescore` after completed fixed-bank candidates are returned. The older
GoalConditionedSelector is NOT used: it adds goal to the planner status
embedding and does not satisfy this experiment's final-selection boundary.

Q7's allowance for indirect improvement of common perception features supports
testing C; this is not an organizer approval. Semantic-output-only interfaces
are not an additional rule. We must inspect actual code and evaluate whether
shared perception improves; merely renaming a status embedding is insufficient.
The old A2 gain was only about 0.0072, so no performance success is assumed.

## Training and decision

- Public initialization: original SparseDriveV2 NAVSIMv1 checkpoint SHA
  330072b133981f77b0b18b669216ad50b211552a82ea45ac52d507ec9021a735.
- Bank: original train203 P1024 x V1024 100m bank, SHA
  4aff9b40696f91383bfbec377f388e6209d619fa509e51a2019b19af977b5e04.
- Train 54,810 rows, tune 1,998 rows; unchanged grouped split. All training
  targets, normalizers and bank provenance remain train-only. Tune is a reused
  exploratory set; its results cannot certify untouched generalization.
- Seed 0, reset training RNG to 1 after all construction. 2,000 steps, batch16,
  evaluation batch8, fixed BN running statistics. AdamW weight decay .01,
  public backbone lr1e-5, public head lr1e-4, new modules lr1e-3, warmup100
  and cosine decay. No best-checkpoint selection: report terminal step2000.
- Loss: original coarse/fine D3 soft CE (temperature .1), plus .25 times
  occupancy/lane BCE and .1 times state SmoothL1 normalized by [20,5,3,3].
  Positive BCE weights are predeclared 4 and 8. Empty valid auxiliary masks
  contribute differentiable zero; label validity is never a model input.
- Existing 6-camera auxiliary masks intersect current-front3 visibility.
  Raster x-forward [-10,70], y-left [-32,32], grid64x48, z0/1 visibility.
- Check finite forward/backward, original public tensor reuse, fixed-bank
  coordinate identity, numerical parity at zero initialization, explicit
  goal/status routes, matched row order and real-vs-repeat pixel contract
  before full launches. Preserve failed canaries as failures.
- Judge actual D3, shortlist oracle, selection regret, 3-second L2, per-session
  paired changes, state errors and occ/lane IoU. An auxiliary metric improvement
  alone is not evidence of better planning. If the new candidate set cannot
  support .15, final reranking is insufficient. Inference cost and final B1
  evaluation remain required after the trained comparison.

The full goal is not completed by producing this protocol or by the frozen
diagnostic. The trained comparison, interpretation, inference/dataflow checks
and final report remain outstanding.
