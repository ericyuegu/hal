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
git clone --quiet "https://${GITHUB_TOKEN:+${GITHUB_TOKEN}@}github.com/ericyuegu/hal.git" /opt/hal
cd /opt/hal
git checkout --quiet "$HAL_GIT_SHA"
uv sync --locked

# Datasets/fixtures from R2 (sha-pinned, idempotent); stats.json sits outside the
# streamed shards so pull it up front. Shards stream lazily during training.
log "fetching fixtures + dataset stats"
uv run fetch
uv run python -c "from hal import streams; [streams.pull_stats(s) for s in streams.ALL]"

# StreamingDataset + the PyTorch DataLoader coordinate workers through /dev/shm; vast's
# default 64MB is far too small (StreamingDataset shm arrays + batched-tensor IPC need a
# few hundred MB) and silently stalls the loader before the first batch. compose sets
# shm_size: 16gb for the same reason. Remount larger; tolerate a host that forbids it.
mount -o remount,size=4g /dev/shm 2>/dev/null && log "/dev/shm -> 4g" || log "WARN: could not enlarge /dev/shm"

# Headless GL context for the closed-loop Dolphin eval.
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &)
export DISPLAY=:99

# Preflight: the image ships CUDA 13. A host whose driver is too old (cuda_max_good < 13)
# fails torch's CUDA init (error 804: forward-compat unsupported on consumer GPUs), and the
# trainer's `DEVICE = "cuda" if is_available() else "cpu"` then silently runs ~100x slower on
# CPU and never self-stops. Assert here so a slipped-through box trips the boot trap -> stop.
log "preflight: asserting CUDA is available"
uv run python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() is False — host driver too old for the CUDA-13 image'"

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
