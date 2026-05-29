#!/usr/bin/env bash
# vast.ai cloud lifecycle for a single training run. Set as the instance's
# On-start command (docker/launch_vast.py wires onstart_cmd=/usr/local/bin/on-start.sh).
# The image ships only the environment (deps in /opt/venv), so this script lays
# the code in at the exact git SHA, runs one training command, and tears the box
# down — there is no persistent state on the instance: datasets are fetched from
# R2, checkpoints stream back to R2, and logs go to W&B during the run.
#
# Driven entirely by env injected at create time:
#   HAL_GIT_SHA          commit to check out
#   HAL_TRAIN_CMD_B64    base64 of the training command (base64 survives the env string)
#   GITHUB_TOKEN         clones the private repo (also used as the ghcr image login)
#   AWS_*, WANDB_API_KEY  R2 + W&B credentials
# vast injects CONTAINER_ID + CONTAINER_API_KEY (a per-instance key) so the box
# can stop/destroy itself.
set -euo pipefail

log() { echo "[on-start] $*"; }

# Any failure before training starts (clone, sync, fetch) stops the box for
# inspection rather than leaving it idle-billing.
trap 'log "boot failed (line $LINENO); stopping instance for inspection"; \
      VAST_API_KEY="$CONTAINER_API_KEY" vastai stop instance "$CONTAINER_ID" || true' ERR

# vast hides the -e env vars from interactive SSH sessions; persist the creds so a
# manual `ssh` peek (e.g. to --resume) sees them too.
env | grep -E '^(AWS_|WANDB_|GITHUB_TOKEN)=' >> /etc/environment

# Code at the exact SHA. The image bakes no repo, so this is a clean clone into the
# empty /opt/hal; uv sync then installs the pure-Python hal into the prebuilt venv
# (fast, no compiler — a uv.lock mismatch would fail loud here and trip the trap).
log "cloning hal @ ${HAL_GIT_SHA}"
git clone --quiet "https://${GITHUB_TOKEN}@github.com/ericyuegu/hal.git" /opt/hal
cd /opt/hal
git checkout --quiet "$HAL_GIT_SHA"
uv sync --locked

# Datasets/fixtures from R2 (sha-pinned, idempotent); stats.json sits outside the
# streamed shards so pull it up front. Shards stream lazily during training.
log "fetching fixtures + dataset stats"
uv run fetch
uv run python -c "from hal import streams; [streams.pull_stats(s) for s in streams.ALL]"

# Headless GL context for the closed-loop Dolphin eval.
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &)
export DISPLAY=:99

cmd="$(printf '%s' "$HAL_TRAIN_CMD_B64" | base64 -d)"
log "training: ${cmd}"
# Run the training command itself outside `set -e`/the trap so we can branch on its
# exit code: success destroys the box (checkpoints are already in R2), any non-zero
# exit stops it (keeps /opt/hal/train.log + the box for inspection / --resume).
set +e
bash -lc "$cmd" 2>&1 | tee /opt/hal/train.log
code=${PIPESTATUS[0]}
set -e

if [ "$code" -eq 0 ]; then
  log "training succeeded; destroying instance (checkpoints in R2, logs in W&B)"
  VAST_API_KEY="$CONTAINER_API_KEY" vastai destroy instance "$CONTAINER_ID"
else
  log "training exited ${code}; stopping instance for inspection"
  VAST_API_KEY="$CONTAINER_API_KEY" vastai stop instance "$CONTAINER_ID"
fi
