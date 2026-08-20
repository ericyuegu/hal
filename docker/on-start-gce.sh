# Google Compute Engine startup lifecycle. scripts/launch_gce.py prepends the
# non-secret HAL_* configuration before placing this script in instance metadata.
# Secret values are fetched at boot from Secret Manager with the VM service account.
set -euo pipefail

log() { echo "[hal-gce] $*"; }

finish() {
  code="$1"
  if [ "${HAL_KEEP_ALIVE:-0}" = "1" ]; then
    log "job exited ${code}; HAL_KEEP_ALIVE=1, leaving VM running"
    exit "$code"
  fi
  log "job exited ${code}; shutting VM down (delete it later to remove the boot disk)"
  shutdown -h now
  exit "$code"
}
trap 'code=$?; log "startup failed at line ${LINENO}"; finish "$code"' ERR

: "${HAL_GIT_SHA:?missing HAL_GIT_SHA}"
: "${HAL_TRAIN_CMD_B64:?missing HAL_TRAIN_CMD_B64}"
: "${HAL_GCP_PROJECT:?missing HAL_GCP_PROJECT}"
: "${HAL_SECRET_SPECS_B64:?missing HAL_SECRET_SPECS_B64}"
: "${HAL_IMAGE:?missing HAL_IMAGE}"

install -d -m 0700 /run/hal
: > /run/hal/job.env
chmod 0600 /run/hal/job.env
while IFS='=' read -r env_name secret_id; do
  [ -n "$env_name" ] || continue
  log "loading ${env_name} from Secret Manager secret ${secret_id}"
  printf '%s=' "$env_name" >> /run/hal/job.env
  gcloud secrets versions access latest --secret="$secret_id" --project="$HAL_GCP_PROJECT" >> /run/hal/job.env
  printf '\n' >> /run/hal/job.env
done < <(printf '%s' "$HAL_SECRET_SPECS_B64" | base64 -d)

command -v docker >/dev/null || { log "Docker is missing from the selected DLVM image"; false; }
nvidia-smi
log "pulling ${HAL_IMAGE}"
docker pull "$HAL_IMAGE"

# The runtime script lives outside the repository because the exact SHA has not
# been cloned yet. It executes inside the prebuilt HAL image and uses its /opt/venv.
cat > /run/hal/run.sh <<'HAL_RUNTIME'
#!/usr/bin/env bash
set -euo pipefail
log() { echo "[hal-container] $*"; }
trap 'code=$?; log "boot failed at line ${LINENO}"; exit "$code"' ERR

cd /
rm -rf /opt/hal
log "cloning HAL @ ${HAL_GIT_SHA}"
git clone --quiet "https://github.com/ericyuegu/hal.git" /opt/hal
cd /opt/hal
git checkout --quiet "$HAL_GIT_SHA"
uv sync --locked

log "fetching fixtures"
uv run fetch
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &)
export DISPLAY=:99
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable inside HAL container'"
ulimit -n "$(ulimit -Hn)" || true

cmd="$(printf '%s' "$HAL_TRAIN_CMD_B64" | base64 -d)"
log "training: ${cmd}"
bash -c "$cmd"
HAL_RUNTIME
chmod 0700 /run/hal/run.sh

set +e
docker run --rm --gpus all --ipc=host \
  --env-file /run/hal/job.env \
  -e HAL_GIT_SHA -e HAL_TRAIN_CMD_B64 \
  -v /run/hal/run.sh:/run/hal/run.sh:ro \
  "$HAL_IMAGE" bash /run/hal/run.sh 2>&1 | tee /var/log/hal-training.log
code=${PIPESTATUS[0]}
rm -f /run/hal/job.env
finish "$code"
