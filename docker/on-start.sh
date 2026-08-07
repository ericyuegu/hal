#!/usr/bin/env bash
# vast.ai cloud lifecycle for a single training run. scripts/launch_vast.py reads
# this file and passes its contents inline as the instance's onstart command (NOT
# baked into the image — it runs before the repo is cloned, so baking would strand
# the box on a stale copy). The image ships only the environment (deps in
# /opt/venv), so this script lays the code in at the exact git SHA, runs one
# training command, and tears the box down — there is no persistent state on the
# instance: datasets are fetched from R2, checkpoints stream back to R2, and logs
# go to W&B during the run.
#
# Driven entirely by env injected at create time:
#   HAL_GIT_SHA          commit to check out
#   HAL_TRAIN_CMD_B64    base64 of the training command (base64 survives the env string)
#   AWS_*, WANDB_API_KEY  R2 + W&B credentials
#   GITHUB_TOKEN         optional; only set when the repo/image is private
#   HAL_KEEP_ALIVE       optional; "1" disables all self-teardown (debug: leave box up)
# vast injects CONTAINER_ID + CONTAINER_API_KEY (a per-instance key) so the box
# can stop/destroy itself.
set -euo pipefail

log() { echo "[on-start] $*"; }

# vast does NOT inject env into this on-start shell, so recover it from two sources:
#   - account env-vars (R2 + W&B secrets, set in the vast console) — written to
#     /etc/environment, deliberately kept out of the per-instance config so they
#     don't leak into `show instance`/extra_env.
#   - the per-run `-e` vars (HAL_GIT_SHA, HAL_TRAIN_CMD_B64, HAL_KEEP_ALIVE) — these
#     live in the container's PID 1 environ (NUL-separated).
# Export both so this script and the training child inherit them.
set -a
[ -r /etc/environment ] && . /etc/environment 2>/dev/null || true
set +a
if [ -r /proc/1/environ ]; then
  while IFS= read -r -d '' kv; do
    case "$kv" in AWS_*=* | WANDB_*=* | GITHUB_TOKEN=* | HAL_*=*) export "$kv" ;; esac
  done < /proc/1/environ
fi
log "env check: AWS_ENDPOINT_URL=${AWS_ENDPOINT_URL:+set} WANDB_API_KEY=${WANDB_API_KEY:+set} HAL_GIT_SHA=${HAL_GIT_SHA:+set} HAL_TRAIN_CMD_B64=${HAL_TRAIN_CMD_B64:+set}"

# Teardown, gated on HAL_KEEP_ALIVE so a debug run leaves the box SSH-able. $1 is the
# vast verb (stop|destroy), $2 a human reason for the log.
teardown() {
  if [ "${HAL_KEEP_ALIVE:-0}" = "1" ]; then
    log "HAL_KEEP_ALIVE=1 — leaving instance up ($2); destroy manually when done"
    return
  fi
  log "$2 — ${1}ing instance"
  # `destroy` prompts for confirmation; onstart has no TTY, so without -y it aborts and
  # leaves the box billing after a successful run. `stop` has no prompt (and no -y flag).
  yes_flag=""
  [ "$1" = destroy ] && yes_flag="-y"
  VAST_API_KEY="$CONTAINER_API_KEY" vastai "$1" instance "$CONTAINER_ID" $yes_flag || true
}

# Any failure during boot (clone, sync, fetch) stops the box (or keeps it under
# HAL_KEEP_ALIVE) rather than leaving it idle-billing.
trap 'log "boot failed (line $LINENO)"; teardown stop "boot failure"; exit 1' ERR

# Fail loud + early if the injected inputs are missing (e.g. env recovery found
# nothing) instead of dying obscurely mid-clone.
: "${HAL_GIT_SHA:?missing — vast -e env not recovered from /proc/1/environ}"
: "${HAL_TRAIN_CMD_B64:?missing — vast -e env not recovered from /proc/1/environ}"
: "${AWS_ENDPOINT_URL:?missing — R2 creds not recovered from /proc/1/environ}"

# Persist creds so an interactive `ssh` peek (e.g. to --resume) sees them too.
# `|| true`: grep exits 1 on no match, which must not trip `set -e`.
env | grep -E '^(AWS_|WANDB_|GITHUB_TOKEN)=' >> /etc/environment || true

# Code at the exact SHA. Clone into a fresh /opt/hal (rm first so it's robust whether
# the image baked a repo there or not — the venv lives at /opt/venv, untouched). uv
# sync then installs the pure-Python hal into the prebuilt venv (fast, no compiler —
# a uv.lock mismatch would fail loud here and trip the trap).
log "cloning hal @ ${HAL_GIT_SHA}"
# Step out of /opt/hal before deleting it: the onstart shell can inherit /opt/hal as
# its cwd, and `rm -rf` on the cwd makes the next git command die with "Unable to read
# current working directory". Don't rely on the image WORKDIR being a safe value.
cd /
rm -rf /opt/hal
# Public repo clones anonymously; the ${GITHUB_TOKEN:+…@} prefix injects auth only
# if a token was set (private repo/image). Safe under `set -u`.
# github.com is reachable from a vast host only most of the time — during a github edge
# incident the connect times out after ~2 minutes and one blip burns the whole (billed)
# boot. Retry a few times, then fail loud.
clone_url="https://${GITHUB_TOKEN:+${GITHUB_TOKEN}@}github.com/ericyuegu/hal.git"
cloned=0
for attempt in 1 2 3; do
  if git clone --quiet "$clone_url" /opt/hal; then
    cloned=1
    break
  fi
  log "clone attempt ${attempt} failed (github.com unreachable); retrying in 30s"
  rm -rf /opt/hal
  sleep 30
done
if [ "$cloned" -ne 1 ]; then
  log "FATAL: cannot clone hal after 3 attempts — github.com unreachable from this host"
  teardown stop "git clone failed (github.com unreachable)"
  exit 1
fi
cd /opt/hal
git checkout --quiet "$HAL_GIT_SHA"
uv sync --locked

# Shared memory, provisioned BEFORE the (billed) fetch so a shm-incapable host aborts in
# seconds, not after downloading the ISO+dataset. vast's container /dev/shm defaults to 64MB.
# The trainer uses the 'file_system' tensor-sharing strategy, so per-batch worker IPC (several
# GB at high worker/batch counts) no longer rides /dev/shm — but StreamingDataset's own
# coordination arrays still need room beyond 64MB. Remount to 16g (compose uses the same), then
# FAIL FAST if it didn't take: a too-small /dev/shm otherwise only surfaces as a DataLoader
# worker dying at step 0 (RuntimeError: worker exited unexpectedly), wasting the whole boot.
mount -o remount,size=16g /dev/shm 2>/dev/null && log "/dev/shm -> 16g" || log "WARN: /dev/shm remount failed"
shm_mb=$(df -m /dev/shm | awk 'NR==2 {print $2}')
log "/dev/shm = ${shm_mb}MB"
if [ "${shm_mb:-0}" -lt 1024 ]; then
  log "FATAL: /dev/shm ${shm_mb}MB < 1GB (remount failed/undersized) — dataloader would die at step 0; aborting"
  teardown stop "insufficient /dev/shm (${shm_mb}MB)"
  exit 1
fi

# Headless GL context for the closed-loop Dolphin eval.
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &)
export DISPLAY=:99

# Preflight: the image ships CUDA 13. A host whose driver is too old (cuda_max_good < 13)
# fails torch's CUDA init (error 804: forward-compat unsupported on consumer GPUs), and the
# trainer's `DEVICE = "cuda" if is_available() else "cpu"` then silently runs ~100x slower on
# CPU and never self-stops. Assert here so a slipped-through box trips the boot trap -> stop.
log "preflight: asserting CUDA is available"
gpu_cap=$(uv run python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() is False — host driver too old for the CUDA-13 image'; print(''.join(map(str, torch.cuda.get_device_capability())))")
log "CUDA compute capability = sm_${gpu_cap}"

# Put compile files on the provisioned disk. A full /tmp can kill a compile worker and leave the
# parent process waiting forever.
export TORCHINDUCTOR_CACHE_DIR=/opt/hal-cache/torchinductor
export TRITON_CACHE_DIR=/opt/hal-cache/triton
export CUDA_CACHE_PATH=/opt/hal-cache/cuda
# Inductor also writes temporary files outside its persistent cache.
export TMPDIR=/opt/hal-cache/tmp
log "preflight: compile caches -> /opt/hal-cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"
for d in "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$TMPDIR"; do
  if ! touch "$d/.write-test" 2>/dev/null; then
    log "FATAL: compile cache ${d} is not writable — inductor would fall back to a full pool stall; aborting"
    teardown stop "compile cache unwritable (${d})"
    exit 1
  fi
  rm -f "$d/.write-test"
done
# Use one compile worker on Blackwell to limit cold-start disk use.
if [ "$gpu_cap" = 120 ]; then
  export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
  log "sm_120: TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS}"
fi
uv run python -c "import os, tempfile; from torch._inductor.codecache import cache_dir; assert cache_dir() == os.environ['TORCHINDUCTOR_CACHE_DIR']; assert tempfile.gettempdir() == os.environ['TMPDIR']; print('[on-start] python compile cache:', cache_dir(), 'temp:', tempfile.gettempdir())"
df -h /opt/hal-cache | while IFS= read -r line; do log "df: ${line}"; done
cache_free_gb=$(df -BG --output=avail /opt/hal-cache | awk 'NR==2 {gsub("G",""); print $1}')
log "compile-cache free = ${cache_free_gb}GB"
if [ "${cache_free_gb:-0}" -lt 10 ]; then
  log "FATAL: /opt/hal-cache has ${cache_free_gb}GB free (< 10GB) — a compile worker dying mid-write stalls the run at 0% GPU; aborting"
  teardown stop "insufficient compile-cache disk (${cache_free_gb}GB)"
  exit 1
fi

# Test the Blackwell compile path before downloading the dataset.
if [ "$gpu_cap" = 120 ] && [ "${HAL_SKIP_SM120_PROBE:-0}" != 1 ]; then
  log "sm_120: running compile smoke probe (600s hard timeout)"
  timeout 600 uv run notebooks/probe_sm120.py
fi

# Pull data only after the GPU and compiler checks pass.
log "fetching fixtures + dataset stats"
uv run fetch
uv run python -c "from hal import streams; [streams.pull_stats(s) for s in streams.ALL]"

# DataLoader workers pass tensor-storage handles to the main process over a unix socket
# via file descriptors; num_workers * prefetch_factor in-flight batches can exceed the
# container's default open-fd soft limit (often 1024), crashing mid-run with
# "RuntimeError: received 0 items of ancdata" in recvfds. Raise the soft limit to the
# hard cap so fd-passing has headroom.
ulimit -n "$(ulimit -Hn)" && log "open files (ulimit -n) -> $(ulimit -n)"

cmd="$(printf '%s' "$HAL_TRAIN_CMD_B64" | base64 -d)"
log "training: ${cmd}"
# Run training outside the trap so we can branch on its exit code: success destroys
# the box (checkpoints already in R2), non-zero stops it (keeps /opt/hal/train.log
# for inspection / --resume) — both subject to HAL_KEEP_ALIVE.
set +e
bash -c "$cmd" 2>&1 | tee /opt/hal/train.log
code=${PIPESTATUS[0]}
set -e

if [ "$code" -eq 0 ]; then
  teardown destroy "training succeeded (checkpoints in R2, logs in W&B)"
else
  teardown stop "training exited ${code}"
fi
