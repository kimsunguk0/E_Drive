# Privileged GT7 versus GT24 diagnostic protocol

Status: fixed CPU-only proposal; no result yet.

This diagnostic asks whether the pose/timestamp-derived GT binary stop plus GT
history16 add information beyond GT physical `(vx,vy,ax,ay,yaw_rate)` and the same raw
provided 5-second goal, and separately whether the state5 time convention matters.
All three arms retain the identical `24→512→512→12` GELU/LayerNorm
model and parameter count. It is privileged, non-submittable, and not a deployable
planning result.

The primary inputs are the frozen canonical C1 GT24 train54,810/tune1,998 artifacts.
Their state and history targets use the same causal raw timestamp convention. The third
arm uses the separately frozen A1 nominal-time causal status overlay, joined exactly by
row/frame, only to replace state5; stop/history remain masked.

One normalizer is fitted to the full canonical GT24 train matrix only. `canonical_full24`
is unchanged. `canonical_masked7` zeros normalized indices 5 through 21: binary stop and
all 16 history fields. It retains normalized indices 0 through 4 and 22 through 23,
exactly raw-time state5 plus the same raw goal2. `a1_nominal_masked7` replaces only the
first five raw fields with A1 nominal-time causal state before applying that same
normalizer and mask. This separates the history+stop contrast from the time-contract
contrast without changing input shape or parameter count.

For seeds 0 and 1, all three arms share initialization and shuffled row order. Each uses
AdamW LR `1e-3`, weight decay `1e-4`, batch 1024, 60 epochs/3,240 optimizer updates,
per-step cosine decay, SmoothL1 beta 0.1, and LAST60 only. There is exactly one terminal
tune pass per arm. That pass saves every row identity, predicted six-point trajectory
and FP32 per-row official D3; no follow-up forward is needed.

The script must pin the original diagnostic helper, current D3/state-hash helper, A1
overlay helper/artifacts, GT train/tune artifacts/manifests, and the B200 normalizer SHA. It runs with empty CUDA
visibility and one CPU thread. Final136, image-model forwards, A1 training, additional
arms, epoch selection and hyperparameter sweeps are prohibited.

Interpretation is narrow: canonical-masked7 minus full24 estimates incremental predictive
value of GT stop/history; A1-nominal7 minus canonical7 estimates the timing-contract effect
on state5 under this fixed privileged MLP family and reused tune split. Neither contrast
proves images can estimate state, that the current image planner can exploit it, or that
the raw-goal route is compliant.
