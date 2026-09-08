# P7 train203 bank proxy: fixed-output localization

Status: read-only CPU recomputation from the already frozen K1024/M12 artifacts. This is the same exploratory proxy, not a new arm, model evaluation, selector tuning, or deployment result. No fit, model forward, GPU, final-row access, threshold, candidate-count change, or coordinate edit was performed.

## Immutable inputs and method

- Frozen feasibility result: `tune_feasibility.json`, SHA256 `32947a5bb83df6d12d8ab47427b854c761d3fcd10d4c62480dbd2e9b8e5585c7`.
- Frozen bank: `bank.npy`, SHA256 `ee0e2267d29bb2253c5e0542d52b01cc5a0537ee66af10b55188afa3605aa469`; manifest SHA256 `33d7e93c3c776363dc3636fca127a9b6a7bffbc622b6bfea61e9e5a030b8b16c`.
- Saved controls: C0 SHA256 `67279fe06120ef752d44c915dbc01d022244b94d60ba27577ed62d1bace1afd0`; C1 SHA256 `703f46945496e2de9d5a9386e8596aa0fb01d6f47e627f9c8db33b0b58b6f795`.
- ego5 SHA256 `700d9443cbf38aaaeffbbd5dab789b5e448adf931e15939baa9d37779e4095c1`; all 1,998 selected tune masks are valid and `fut5[:, :6]` equals saved GT bitwise at the exact row join.
- Execution receipt SHA256 `cf1d9b722c544cd3cacdd8c7e51d0c59e21f241829fe43649fd9d77d63d63e3c` records fit/evaluate rc0 and no GPU/final access.
- Candidate construction and selection are unchanged: the fixed top-12 is determined only by saved C prediction and the frozen bank; prefix selector is stable nearest prefix; endpoint selector is stable nearest 5 s endpoint; shortlist/full-bank oracles use GT only as coverage/regret diagnostics.
- D3 uses the frozen FP32 PyTorch calculation with weights `[11,11,5,5,2,2]/36`. Recomputed direct D3 differs from stored GPU D3 only by operation-level FP32 rounding: C0 652 elements, max `1.1920929e-7` m; C1 662, max `2.3841858e-7` m. Stored means remain C0 `0.3305861224`, C1 `0.3246189025`; all localized deltas below use one consistent frozen CPU calculation.
- Bootstrap: 10,000 shared draws, seed 20260908, uniform resampling of the same 11 whole sessions; each draw retains the frame-weighted estimator. No within-session frame bootstrap.

## Aggregate result and uncertainty

| metric (m D3) | C0 | C1 | two-base mean / shared-session 95% CI |
|---|---:|---:|---:|
| direct C | 0.33058614 | 0.32461891 | 0.32760252 |
| prefix-nearest | 0.34622130 | 0.34288949 | 0.34455539 |
| endpoint-nearest | 0.33384800 | 0.32683477 | 0.33034138 |
| M12 oracle | 0.19351162 | 0.19190860 | 0.19271011 |
| full-K oracle | 0.11486043 | 0.11486043 | 0.11486043 |
| prefix minus direct | +0.01563516 | +0.01827058 | +0.01695289 `[+0.01183869,+0.02493620]` |
| endpoint minus prefix | -0.01237330 | -0.01605472 | -0.01421403 `[-0.03330863,+0.00471405]` |
| endpoint minus direct | +0.00326186 | +0.00221586 | +0.00273887 `[-0.01436699,+0.02155140]` |
| M12 oracle minus direct | -0.13707452 | -0.13271031 | -0.13489240 `[-0.15387220,-0.11635417]` |

The endpoint rule recovers most of the prefix quantization loss in the point estimate, but remains slightly worse than untouched C and its paired CI includes zero. This is no net gain.

## Fixed buckets and count-weighted attribution

Bucket counts are steady 99, depart 24, nonstop 1,875. “Contribution” is the bucket's sum of per-row endpoint-minus-direct divided by all 1,998 rows, so the three contributions add to the aggregate delta.

| base / bucket | direct | prefix | endpoint | M12 oracle | endpoint-direct mean | contribution to all-row delta | endpoint-prefix |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 steady | 1.028565 | 1.009711 | 0.700590 | 0.596370 | -0.327974 | -0.016251 | -0.309121 |
| C0 depart | 0.935933 | 0.924432 | 0.841526 | 0.553688 | -0.094407 | -0.001134 | -0.082906 |
| C0 nonstop | 0.285984 | 0.303788 | 0.307986 | 0.167630 | +0.022001 | +0.020647 | +0.004198 |
| C1 steady | 1.012657 | 1.022659 | 0.708506 | 0.613313 | -0.304150 | -0.015071 | -0.314153 |
| C1 depart | 0.982757 | 0.950594 | 0.866777 | 0.596553 | -0.115980 | -0.001393 | -0.083818 |
| C1 nonstop | 0.279866 | 0.299219 | 0.299771 | 0.164479 | +0.019905 | +0.018680 | +0.000552 |

Endpoint selection materially helps the 123 steady/depart rows, but the small harm across 1,875 nonstop rows outweighs it. This count-weighted localization explains the aggregate regression; it is not a post-hoc selectable rule.

## Longitudinal/lateral diagnostic

These are official-time-weighted absolute coordinate errors, not additive components of Euclidean D3.

| base / path | longitudinal x (m) | lateral y (m) |
|---|---:|---:|
| C0 direct | 0.296556 | 0.084108 |
| C0 prefix | 0.308680 | 0.086566 |
| C0 endpoint | 0.306002 | 0.072570 |
| C1 direct | 0.290607 | 0.082354 |
| C1 prefix | 0.304376 | 0.087799 |
| C1 endpoint | 0.299461 | 0.071220 |

Relative to direct C, endpoint selection increases longitudinal absolute error by `+0.009446` m (C0) / `+0.008854` m (C1), while reducing lateral error by `-0.011538` / `-0.011134` m. Relative to prefix selection it improves both axes, but not enough to beat direct C under D3.

## Coverage, shortlist miss, and selector regret

| diagnostic (m D3 unless stated) | C0 | C1 |
|---|---:|---:|
| M12 oracle minus full-K oracle (shortlist miss) | 0.07865120 | 0.07704816 |
| prefix minus M12 oracle (selector regret) | 0.15270968 | 0.15098090 |
| endpoint minus M12 oracle (selector regret) | 0.14033636 | 0.13492616 |
| direct C strictly better than M12 oracle | 294/1998 (14.715%) | 297/1998 (14.865%) |

The full bank has substantial representation coverage, but this does not translate into selectable performance. Both shortlist miss and selector regret are large, and even the GT-best row among the fixed 12 loses to untouched direct C on about 15% of rows. A negative fixed-M12 proxy does not close the hypothesis of a learned multimodal generator whose modes can fall outside this local shortlist.

## The identical 1,663 change counts are genuine

The two seeds do not reuse the same prediction, shortlist, or choice masks. Prediction tensor SHAs are C0 `f7ecb019e55e936028d5ed4185685869769c03fde8cf8d0f03b525ae4ad3c65a` and C1 `92b99550d190b0e689a0200691bdd466f8e2f27023e61a28831c8c4ec10a64d1`. Shortlist SHAs are also distinct. Although each seed changes 1,663 endpoint IDs, the changed-row masks intersect on 1,523 rows, have union 1,803, and differ on 280 rows. Prefix choices match on 1,099 rows and endpoint choices on 1,590 rows. Thus the equal counts are coincidental at aggregate level, not reused C0 arrays.

Conclusion: the fixed endpoint selector is a better index rule than prefix-nearest for these exact M12 candidates, especially on steady/depart and lateral error, but it does not beat direct C overall. No adoption, accuracy, compliance, or learned multimodal claim follows from these saved-output diagnostics.
