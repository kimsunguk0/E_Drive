# Phase 5A — Factorized Path × Velocity Bank Oracle Sweep

Generated: 2026-09-04T15:02:58+09:00 KST

> CPU-only, train-only fitting. P/V selection used only the 8-scenario mini-val. The 38-scenario dev38 split was evaluated once after locking the configuration and is already experiment-contaminated; it is report-only, not an unbiased final-val estimate.

## Decision

**KEEP for coarse/fine scorer experiments** — selected `P=512, V=128` (65,025 virtual candidates).

- Mini-val D3: `0.104003`; legacy `0.114514`; delta `-0.010511`.
- Dev38 D3: `0.092941`; legacy `0.099634`; delta `-0.006694`.
- Dev38 5s endpoint / mean-trajectory oracle: `0.114108` / `0.260682`.
- Exact-zero candidate: `True`.

## Input provenance

| input | path | SHA256 |
|---|---|---|
| ego5 | `/NHNHOME/data/sukim/adcl/data/etri/ego_cache_5s.npz` | `700d9443cbf38aaaeffbbd5dab789b5e448adf931e15939baa9d37779e4095c1` |
| split | `/NHNHOME/data/sukim/adcl/data/etri/val_clips.npz` | `e56ee8803d8de63eb5fa700283afdddf6d735746625c8fcd781251e8902c60c0` |
| anchor | `/NHNHOME/data/sukim/adcl/h200_latest/artifacts/dense_vocab/etri_train330_vocab_k1024.npy` | `04c5f2fc20dffd489907d2f7a9e550a11ee2895974c6f276736d70bb3b074ced` |
| a0 | `/NHNHOME/data/sukim/adcl/data/etri/bank_A0_onetail.npz` | `4ccb7ac0358b977e707c121a200e621f2efd34d89c3abbdabf2247b67b291aa9` |
| a1 | `/NHNHOME/data/sukim/adcl/data/etri/bank_A1_multitail.npz` | `fbff5f18f997bbab9252905ceedea8ebbe1981c4eff9b8904eec893c832d0230` |
| sweep script | `scripts/sweep_factorized_bank.py` | `e8f1d6785813a6ed60d0e6f62edc0d737cc1dbbb15d672edb66647d47d6afa9c` |

Train fit rows: **17820** (`train_idx`, frame≥30); mini-val rows: **432**; dev38 rows: **2052**.
No test paths, rows, statistics, or thresholds were used.

## Deterministic approximation

Full pairwise k-medoids at N=17,820 and four P/V sizes would be wasteful. We use seeded MiniBatchKMeans in standardized feature space, then replace every centre with a unique real train trajectory. Geometry uses uniform arc-length q(u) features and sqrt-count quotas for straight/left/right/U-turn. Velocity uses `[S,r1..r10]` and quotas for creep/decel/cruise/accel. Stop is a separate exact-zero row. Thus all prototypes are train medoids (not learned synthetic coordinates); only cross-products are composed.

A first, rejected diagnostic clustered per-stratum z-scored `[S,r1..r10]` directly. At P512/V128 it produced mini D3 `0.281397` and dev38 report-only D3 `0.206473`, failing the gate badly. The constant r10 and low-variance normalized-progress dimensions distorted Euclidean distance. The reported sweep therefore clusters the same stored `[S,r]` medoids through the induced physical progress `s_i=S*r_i`, globally standardized and official-weighted. This rejected result is retained in the JSON manifest.

## Mini-val sweep (selection source)

| P | V | candidates | D3 | Δ vs legacy | endpoint5 | traj5 | wall s | gate |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 64 | 16 | 961 | 0.532729 | +0.418216 | 1.551465 | 1.270689 | 0.08 | FAIL |
| 64 | 32 | 1985 | 0.239933 | +0.125419 | 0.526794 | 0.604594 | 0.17 | FAIL |
| 64 | 64 | 4033 | 0.167845 | +0.053332 | 0.366988 | 0.456202 | 0.33 | FAIL |
| 64 | 128 | 8129 | 0.120098 | +0.005585 | 0.274532 | 0.359140 | 0.66 | FAIL |
| 128 | 16 | 1921 | 0.523202 | +0.408689 | 1.493821 | 1.239992 | 0.16 | FAIL |
| 128 | 32 | 3969 | 0.234154 | +0.119640 | 0.437288 | 0.578044 | 0.32 | FAIL |
| 128 | 64 | 8065 | 0.161600 | +0.047087 | 0.294377 | 0.426786 | 0.65 | FAIL |
| 128 | 128 | 16257 | 0.111765 | -0.002749 | 0.205531 | 0.335891 | 1.32 | PASS |
| 256 | 16 | 3841 | 0.515869 | +0.401355 | 1.446387 | 1.231882 | 0.31 | FAIL |
| 256 | 32 | 7937 | 0.225993 | +0.111479 | 0.392417 | 0.565028 | 0.65 | FAIL |
| 256 | 64 | 16129 | 0.154362 | +0.039848 | 0.247969 | 0.411233 | 1.30 | FAIL |
| 256 | 128 | 32513 | 0.107568 | -0.006946 | 0.164238 | 0.316000 | 2.62 | PASS |
| 512 | 16 | 7681 | 0.512748 | +0.398234 | 1.355836 | 1.233888 | 0.62 | FAIL |
| 512 | 32 | 15873 | 0.223657 | +0.109143 | 0.367874 | 0.560164 | 1.28 | FAIL |
| 512 | 64 | 32257 | 0.150648 | +0.036135 | 0.223789 | 0.403444 | 2.60 | FAIL |
| 512 | 128 | 65025 | 0.104003 | -0.010511 | 0.135045 | 0.307550 | 5.23 | PASS |

Selection rule was locked before dev38: among mini-val D3-gate passers, minimize 5s mean-trajectory oracle, then endpoint oracle, then candidate count. This is an oracle bank decision only; learnable coarse/fine ranking is not evaluated here.

## A0/A1 references and selected bank

| split/bank | D3 | endpoint5 | traj5 |
|---|---:|---:|---:|
| mini legacy/A0 | 0.114514 | 0.495206 | 0.393100 |
| mini A1 | 0.114514 | 0.473596 | 0.376752 |
| mini selected P×V | 0.104003 | 0.135045 | 0.307550 |
| dev38 legacy/A0 | 0.099634 | 0.477551 | 0.340789 |
| dev38 A1 | 0.099634 | 0.443044 | 0.326400 |
| dev38 selected P×V | 0.092941 | 0.114108 | 0.260682 |

## Dev38 bucket report (report-only)

| bucket | n | D3 | endpoint5 | traj5 | D3≤0.15 | endpoint≤1m |
|---|---:|---:|---:|---:|---:|---:|
| stop | 109 | 0.006170 | 0.016200 | 0.010077 | 1.000 | 1.000 |
| creep | 32 | 0.016706 | 0.013900 | 0.057208 | 1.000 | 1.000 |
| move | 1911 | 0.122048 | 0.147658 | 0.343916 | 0.738 | 0.992 |
| straight | 1679 | 0.092098 | 0.102289 | 0.257979 | 0.832 | 0.994 |
| left | 138 | 0.157304 | 0.230509 | 0.493891 | 0.470 | 1.000 |
| right | 171 | 0.156255 | 0.319110 | 0.466966 | 0.512 | 0.969 |
| uturn | 64 | 0.028910 | 0.023960 | 0.039435 | 0.939 | 1.000 |

## Artifacts and compute

- Selected bank: `/NHNHOME/data/sukim/adcl/data/etri/factorized_phase5a/factorized_P512_V128.npz`
- Selected bank SHA256: `459c97e313ddaf38a14847f894ae023a91249a5938aaf3b3f25a3733bee1a7e4`
- Machine: `Linux-6.8.0-100-generic-x86_64-with-glibc2.39`; sklearn `1.7.2`; threads `16`; total wall `69.96s`.
- Evaluation is chunked; no full `[samples,candidates,10,2]` distance tensor is retained.

## Axis oracle decomposition

| split | oracle | D3 | endpoint5 | traj5 |
|---|---|---:|---:|---:|
| mini | GT velocity + clustered geometry | 0.012405 | 0.053280 | 0.052808 |
| mini | GT geometry + clustered velocity | 0.132280 | 0.270885 | 0.363563 |
| dev38 | GT velocity + clustered geometry | 0.010500 | 0.045922 | 0.044895 |
| dev38 | GT geometry + clustered velocity | 0.115629 | 0.248696 | 0.302032 |

Exact self-reconstruction maximum absolute error: `3.815e-06` (mini), `3.815e-06` (dev38).

## Keep / kill interpretation

The bank passes the predeclared D3 gate and improves A1 5s coverage on mini-val, with the locked configuration also passing dev38. Promote it only to the next image-only coarse/fine scoring experiment; oracle success does not establish rankability or latency.
