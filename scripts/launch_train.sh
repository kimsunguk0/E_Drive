#!/usr/bin/env bash
# Launch SparseDrive training on this single-GPU host, reproducing the official
# optimisation setup exactly.
#
# ============================ BATCH / LR REASONING ==========================
# The configs are written for 8 GPUs:
#     total_batch_size = 64 ; num_gpus = 8 ; batch_size = 64 // 8 = 8
#     num_iters_per_epoch = length // (num_gpus * batch_size) = 28130 // 64 = 439
#     num_epochs = 100  ->  max_iters = 43900
#     stage1 lr = 4e-4 ; stage2 lr = 3e-4     (tuned for TOTAL batch 64)
#
# Putting all 64 samples on one GPU would be the clean way to keep total batch,
# iteration count and LR schedule identical... except IT CANNOT BE DONE:
#
#   ops/src/deformable_aggregation_cuda.cu:281
#     const int num_kernels = batch*num_pts*num_embeds*num_anchors*num_cams*num_scale
#
#   num_kernels is a plain int32, so it overflows to negative and the launch dies
#   with "RuntimeError: CUDA error: invalid configuration argument".
#   Per-sample kernel counts (num_embeds=256, num_cams=6, num_scale=4):
#
#     det head : num_pts=13  (7 fix_scale + 6 learnable), num_anchor=900
#                13*256*900*6*4   =  71,884,800/sample -> max batch 29
#     map head : num_pts=300 (num_sample 20 * len(fix_height) 5 * num_learnable 3,
#                see map_blocks.py:107),               num_anchor=100
#                300*256*100*6*4  = 184,320,000/sample -> max batch 11   <-- BINDING
#
#   So the real cap is 11, set by the MAP head, not the detection head.
#   Empirically verified: map bs=11 ok, bs=12 (2,211,840,000 > INT_MAX 2,147,483,647)
#   corrupts the CUDA context. The official recipe puts 8 samples on each of
#   8 GPUs, which is why upstream never reaches this limit.
#
#   Fixing the kernel properly would mean widening num_kernels AND the per-thread
#   `int idx` and all downstream index arithmetic to int64 — invasive, and a
#   mistake there yields silently wrong features rather than a crash. Not worth it
#   for a reproduction run, so we stay under the cap instead.
#
# Therefore: run a micro-batch and restore the official TOTAL batch with gradient
# accumulation, keeping lr EXACTLY as the config has it.
#     SAMPLES x ACCUM == 64,  ACCUM = 64 / SAMPLES
#     optimizer_config.type = GradientCumulativeFp16OptimizerHook
#       (fp16 is on, so the fp16-aware cumulative hook is required; it divides
#        loss by cumulative_iters so the gradient equals a true batch-64 step)
#
# Because the runner is ITER-based and every schedule quantity counts DATALOADER
# iterations, all of them scale by ACCUM to preserve the official number of
# OPTIMIZER STEPS (439 iters/epoch x 100 epochs = 43,900):
#     runner.max_iters          43,900 -> 43,900 * ACCUM
#     lr_config.warmup_iters       500 ->    500 * ACCUM
#     checkpoint/eval interval   8,780 ->  8,780 * ACCUM
# lr itself is NEVER touched; `--autoscale-lr` exists in train.py and is
# deliberately not used.
#
# NOTE: this needs patches/02-sparsedrive-honour-optimizer_config-type.diff —
# the repo hardcodes Fp16OptimizerHook and ignores optimizer_config["type"]
# (mmdet_train.py:141), which made the cumulative hook unreachable.
#
# ============================ WHY NOT dist_train.sh =========================
# The repo's tools/dist_train.sh uses `torch.distributed.launch`, which under
# torch >= 2.0 appends `--local-rank=0` (hyphen). train.py declares
# `--local_rank` (underscore), so argparse rejects it:
#     train.py: error: unrecognized arguments: --local-rank=0
# train.py only uses that value to populate the LOCAL_RANK env var when it is
# absent (tools/train.py:103-104), so `torch.distributed.run` (torchrun), which
# exports LOCAL_RANK and passes no such flag, is a drop-in that needs NO repo
# edit. Using `--launcher none` is not an option: that path goes through
# MMDataParallel and hits a genuine mmcv 1.7 / torch 2.x break —
#     mmcv/parallel/_functions.py:75 _get_stream(device)
#     AttributeError: 'int' object has no attribute 'type'
# because torch 2.1's _get_stream expects a torch.device, not an int.
#
# Usage:
#   ./launch_train.sh stage1
#   ./launch_train.sh stage2
#   SMOKE=500 ./launch_train.sh stage1          # short run, no checkpoints
#   RESUME=/path/latest.pth ./launch_train.sh stage1
#   SAMPLES=32 WORKERS=32 ./launch_train.sh stage1   # if VRAM/RAM demands it
set -uo pipefail

PERSIST=${PERSIST:-/home/pm97/workspace/sukim/adcl}
source "$PERSIST/scripts/env_vars.sh" --activate

STAGE=${1:?usage: launch_train.sh stage1|stage2}
REPO=$PERSIST/src/SparseDrive
CFG=${CFG_OVERRIDE:-projects/configs/sparsedrive_small_${STAGE}.py}
[ -f "$REPO/$CFG" ] || { echo "no such config: $CFG"; exit 1; }

# Official schedule differs per stage -- stage2 is NOT a longer stage1. Values
# read off the configs (total_batch_size / num_epochs / length[version]):
#   stage1: 8 GPUs x 8 = 64,  28130//64 = 439 iters/epoch, 100 epochs -> 43,900
#   stage2: 8 GPUs x 6 = 48,  28130//48 = 586 iters/epoch,  10 epochs ->  5,860
# Hardcoding the stage1 numbers here would silently give stage2 a 7.5x too long
# schedule AND the wrong cosine lr decay horizon.
case "$STAGE" in
  stage1) TOTAL_BATCH=64; ITERS_PER_EPOCH=439; EPOCHS=100 ;;
  stage2) TOTAL_BATCH=48; ITERS_PER_EPOCH=586; EPOCHS=10  ;;
  *) echo "unknown stage: $STAGE"; exit 1 ;;
esac
# We DID widen the kernel to int64 and verify it
# (patches/03-deformable-aggregation-int64-index.diff, correct to within the
# kernel's own atomicAdd non-determinism), which lifts the cap to VRAM. It was
# then BENCHMARKED AND REJECTED — throughput is flat in batch size:
#     bs=8  original kernel  0.901 s/iter   8.88 samples/s   14 GB   <-- best
#     bs=8  int64 kernel     1.015 s/iter   7.88 samples/s   14 GB
#     bs=32 int64 kernel     3.777 s/iter   8.47 samples/s   51 GB
#     bs=64 int64 kernel     7.697 s/iter   8.32 samples/s  100 GB
# Bigger batches buy nothing IN THROUGHPUT (the model is already saturated per
# sample), and int64 indexing costs a few percent. That was the whole basis for
# rejecting the int64 kernel on 2026-08-01 and capping at 11.
#
# REVERSED 2026-08-05, because throughput was the wrong criterion. This model is
# trained with GroupInBatchSampler: every batch POSITION is an independent
# temporal stream through one scene, popping one frame per dataloader iteration
# (group_in_batch_sampler.py, buffer_per_local_sample). So
#
#     number of independent scene streams = samples_per_gpu x world_size
#
# The official 8-GPU run gets 8x8 = 64 streams and one optimizer step sees 64
# DIFFERENT scenes, one frame each. At samples_per_gpu=8 with cumulative_iters=8
# we get 8 streams, and one optimizer step sees 8 scenes x 8 CONSECUTIVE frames.
# nuScenes is 2 Hz, so those 8 frames span 4 s of nearly identical imagery: the
# total batch is 64 either way, but our effective independent sample count is
# closer to 8. Gradient accumulation reproduces batch SIZE, never batch DIVERSITY.
#
# Measured cost of that deficit against the official log at matched steps:
#     grad_norm 1.30x, loss 1.17x, and widening as training progressed.
#
# samples_per_gpu=64 with accum=1 makes the optimizer step structurally identical
# to upstream's and removes gradient accumulation from the picture entirely. It
# needs the int64 kernel (int32 overflows past 11) and ~100-110 GB of VRAM.
#
# These are DEFAULTS, not just env overrides, on purpose: the watchdog's automatic
# resume re-invokes this script with no environment, so anything left as an
# override silently reverts on the first restart.
MAX_MICRO=${MAX_MICRO:-64}
# Default to the stage's full official batch so accum is 1 and the stream count
# matches upstream exactly (64 for stage1, 48 for stage2).
SAMPLES=${SAMPLES:-$TOTAL_BATCH}
WORKERS=${WORKERS:-32}          # nproc=128; 32 leaves room for another tenant
SMOKE=${SMOKE:-0}               # dataloader iterations, not optimizer steps
RESUME=${RESUME:-}
PORT=${PORT:-28651}

if [ "$SAMPLES" -gt "$MAX_MICRO" ]; then
  echo "REFUSING: samples_per_gpu=$SAMPLES exceeds MAX_MICRO=$MAX_MICRO."
  echo "          With the int64 kernel the limit is VRAM, not int32 overflow."
  echo "          Raise MAX_MICRO deliberately after checking free VRAM."
  exit 1
fi
# Guard against silently running the int64 batch sizes on the ORIGINAL kernel,
# which would overflow at >11 and produce 'invalid configuration argument'.
if [ "$SAMPLES" -gt 11 ]; then
  _so=$REPO/projects/mmdet3d_plugin/ops/deformable_aggregation_ext*.so
  if ! grep -qa 'int64_widened_marker' $_so 2>/dev/null; then
    echo "WARNING: samples_per_gpu=$SAMPLES needs the int64-widened deformable"
    echo "         aggregation kernel and the marker was not found in the .so."
    echo "         If this run dies with 'invalid configuration argument', rebuild"
    echo "         from patches/03-deformable-aggregation-int64-index.diff."
  fi
fi
if [ $((TOTAL_BATCH % SAMPLES)) -ne 0 ]; then
  echo "REFUSING: SAMPLES=$SAMPLES does not divide the official total batch $TOTAL_BATCH,"
  echo "          so gradient accumulation cannot reproduce it exactly."
  exit 1
fi
ACCUM=$((TOTAL_BATCH / SAMPLES))

# ITERS_PER_EPOCH / EPOCHS come from the per-stage case block above.
# Official saves every 20 epochs (stage1). That is 70,240 dataloader iters here = ~17 h
# uncontended and ~44 h while sharing the GPU, i.e. a crash could cost two days.
# Checkpoint FREQUENCY is a durability setting only — CheckpointHook just writes
# files and does not touch the optimisation — so shortening it does not affect
# reproduction. 5 epochs = 17,560 iters, ~20 checkpoints x 0.9 GB = 18 GB.
CKPT_EVERY_EPOCHS=${CKPT_EVERY_EPOCHS:-5}
OFFICIAL_STEPS=$((ITERS_PER_EPOCH * EPOCHS))            # 43900
# Same number of optimizer steps, expressed in dataloader iterations
MAX_ITERS=$((OFFICIAL_STEPS * ACCUM))
WARMUP_ITERS=$((500 * ACCUM))
CKPT_INTERVAL=$((ITERS_PER_EPOCH * CKPT_EVERY_EPOCHS * ACCUM))

STAMP=$(date +%F_%H%M%S)
# TAG isolates an experiment's work_dir and log from the production run, WITHOUT
# touching runner.max_iters -- which SMOKE does, and which silently invalidates
# any A/B comparison: lr_config is CosineAnnealing over runner.max_iters, so
# shortening it drops the lr to near min_lr and every variant then looks stable.
# For a controlled run, use TAG and kill the job once enough iters are logged.
#   TAG=torchattn CFG_OVERRIDE=..._torchattn.py RESUME=... ./launch_train.sh stage1
if [ "$SMOKE" != "0" ]; then
  WD=$SCRATCH/work_dirs/smoke_${STAGE}_b${SAMPLES}
  LOG=$PERSIST/logs/smoke_${STAGE}_b${SAMPLES}_${STAMP}.log
elif [ -n "${TAG:-}" ]; then
  WD=$SCRATCH/work_dirs/${STAGE}_${TAG}
  LOG=$PERSIST/logs/${STAGE}_${TAG}_${STAMP}.log
else
  WD=$SCRATCH/work_dirs/${STAGE}
  LOG=$PERSIST/logs/${STAGE}_${STAMP}.log
fi
mkdir -p "$WD" "$PERSIST/logs"

# ---------------------------- fp16 LOSS SCALE -------------------------------
# The config ships `fp16 = dict(loss_scale=32.0)` — a FIXED scale. In mmcv 1.7
# (Fp16OptimizerHook.__init__) a float loss_scale sets
#     self._scale_update_param = 32.0
# and the hook then calls `self.loss_scaler.update(self._scale_update_param)`
# every step, which FORCES torch's GradScaler back to 32.0 and destroys its
# automatic backoff. `loss_scale='dynamic'` leaves _scale_update_param = None so
# update() manages the scale properly (halve on overflow, grow when stable).
#
# Why this matters here: stage1 trained cleanly to iter 70,240 (loss 40 -> 18),
# then between iter ~95,000 and ~100,164 it DIVERGED and went permanently NaN:
#     iter  95,778  loss 18.4  grad_norm   304
#     iter  97,818  loss 34.6  grad_norm   810
#     iter  98,838  loss 35.4  grad_norm  1549
#     iter 100,980  loss 34.7  grad_norm   nan
#     iter 103,020+ loss nan   grad_norm   nan     (never recovers)
# With the scale pinned at 32 there is no backoff path once gradients overflow.
# NaN then becomes permanent rather than transient because this model carries
# state across frames (InstanceQueue, queue_length=4): once a NaN enters the
# cached instance features, every later forward is NaN even with clean weights.
# The checkpoint at 70,240 was verified clean (0 non-finite tensors, max |w| 77.2,
# well inside fp16 range) so recovery is a resume, not a restart.
#
# lr is NOT touched — the instruction is explicit that lr must not be lowered to
# paper over instability. If 'dynamic' still diverges, the next step is bf16
# (stable on H200) and that must be recorded in BUILD_LOG.md.
# The config now carries fp16.loss_scale as a DICT
# (patches/06-fp16-loss-scale-dict-enable-backoff.diff), which --cfg-options
# cannot express: a dict(...) literal fails to parse, and the nested form
# `fp16.loss_scale.init_scale=...` is rejected by mmcv because it would replace a
# float with a dict without _delete_=True. So leave LOSS_SCALE empty to use the
# config's dict; set it only to pass a scalar/'dynamic' override for probing.
LOSS_SCALE=${LOSS_SCALE:-}

OPTS=(
  data.samples_per_gpu=$SAMPLES
  data.workers_per_gpu=$WORKERS
  optimizer_config.type=GradientCumulativeFp16OptimizerHook
  optimizer_config.cumulative_iters=$ACCUM
  runner.max_iters=$MAX_ITERS
  lr_config.warmup_iters=$WARMUP_ITERS
  checkpoint_config.interval=$CKPT_INTERVAL
  evaluation.interval=$CKPT_INTERVAL
)
[ -n "$LOSS_SCALE" ] && OPTS+=(fp16.loss_scale=$LOSS_SCALE)

EXTRA=()
# Never let in-training validation gate a multi-day run.
# On 2026-08-02 the first in-training eval crashed outright (mmcv/torch DDP,
# patch 04) and training then sat idle for 14.5 h. With that patched, a STANDALONE
# eval of the same checkpoint got as far as "evaluating 3 categories..." (the map
# metric) and hung there for 14 hours without finishing. Either failure mode
# stalls training indefinitely, so validation is detached from the training loop
# and run separately with scripts/run_eval.sh, where a hang costs nothing.
# Set NOVAL=0 to re-enable once the map metric is understood.
NOVAL=${NOVAL:-1}
[ "$NOVAL" = "1" ] && EXTRA+=(--no-validate)
if [ "$SMOKE" != "0" ]; then
  # Short run: cap iters, log often, and push checkpoint/eval out of reach so the
  # smoke neither writes multi-GB checkpoints nor triggers a full val pass.
  OPTS+=(runner.max_iters=$SMOKE log_config.interval=10
         checkpoint_config.interval=100000000 evaluation.interval=100000000)
fi
[ -n "$RESUME" ] && EXTRA+=(--resume-from "$RESUME")

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# 128 cores exceed the installed OpenBLAS's precompiled thread limit and it
# SEGFAULTS rather than degrading (this killed kmeans_det). Cap it here too.
export OPENBLAS_NUM_THREADS=32 OMP_NUM_THREADS=32 MKL_NUM_THREADS=32

{
  echo "===================================================================="
  echo " stage        : $STAGE"
  echo " config       : $CFG"
  echo " work_dir     : $WD"
  echo " samples/gpu  : $SAMPLES  (micro-batch; hard cap $MAX_MICRO = map-head int32 overflow)"
  echo " grad accum   : $ACCUM   -> effective total batch $((SAMPLES * ACCUM)) == official $TOTAL_BATCH"
  echo " optimizer hk : GradientCumulativeFp16OptimizerHook (needs patch 02)"
  echo " max_iters    : $MAX_ITERS dataloader iters = $OFFICIAL_STEPS optimizer steps (official)"
  echo " warmup_iters : $WARMUP_ITERS (= 500 optimizer steps)"
  echo " ckpt/eval    : every $CKPT_INTERVAL iters (= $CKPT_EVERY_EPOCHS epochs)"
  echo " workers/gpu  : $WORKERS"
  echo " smoke iters  : ${SMOKE:-full} (dataloader iters)"
  echo " in-train val : $([ "$NOVAL" = 1 ] && echo DISABLED\ \(--no-validate\;\ use\ run_eval.sh\) || echo enabled)"
  echo " fp16 scale   : ${LOSS_SCALE:-from config: dict(init_scale=32.0, growth_interval=2000)}"
  echo " resume       : ${RESUME:-none}"
  echo " lr           : untouched from config (no --autoscale-lr)"
  echo " ARCH_LIST    : $TORCH_CUDA_ARCH_LIST"
  echo " data root    : $(readlink -f "$REPO/data/nuscenes")"
  echo " free VRAM    : $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
  echo " started      : $(date -Is)"
  echo "===================================================================="
} | tee -a "$LOG"

python -m torch.distributed.run --nproc_per_node=1 --master_port="$PORT" \
  tools/train.py "$CFG" --launcher pytorch \
  --work-dir "$WD" --seed 0 --deterministic \
  "${EXTRA[@]}" --cfg-options "${OPTS[@]}" >> "$LOG" 2>&1

rc=$?
echo "=== exited rc=$rc at $(date -Is) ===" | tee -a "$LOG"
exit $rc
