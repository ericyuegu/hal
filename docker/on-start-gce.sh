# Google Compute Engine startup lifecycle. scripts/launch_gce.py prepends the
# non-secret HAL_* configuration before placing this script in instance metadata.
# Secret values are fetched at boot from Secret Manager with the VM service account.
set -euo pipefail

log() { echo "[hal-gce] $*"; }

finish() {
  # Keep a late TERM from interrupting upload. Its timeout and the hard stop still apply.
  trap '' TERM INT
  code="$1"
  trap - EXIT ERR
  set +e
  systemctl stop hal-job-deadline.timer 2>/dev/null
  if [ -n "${HAL_RESULTS_PREFIX:-}" ]; then
    printf '%s\n' "$code" > /var/lib/hal/job-exit-code
    cp /var/log/hal-training.log /var/lib/hal/startup.log 2>/dev/null
    log "uploading results to ${HAL_RESULTS_PREFIX} (12 minute bound)"
    timeout --kill-after=30s 720s docker run --rm --entrypoint python \
      --env-file /run/hal/job.env -e PYTHONPATH=/run/hal \
      -v /run/hal:/run/hal:ro -v /var/lib/hal:/hal-output \
      "$HAL_IMAGE" /run/hal/results.py upload --root /hal-output --prefix "$HAL_RESULTS_PREFIX"
    upload_code=$?
    printf '%s\n' "$upload_code" > /var/lib/hal/upload-exit-code
    if [ "$upload_code" != 0 ]; then
      log "RESULT UPLOAD FAILED (${upload_code}); retain the boot disk for recovery"
      code=1
    fi
  fi
  rm -f /run/hal/job.env
  if [ "${HAL_KEEP_ALIVE:-0}" = "1" ]; then
    log "job exited ${code}; HAL_KEEP_ALIVE=1, leaving VM running"
    exit "$code"
  fi
  log "job exited ${code}; shutting VM down (delete it later to remove the boot disk)"
  shutdown -h now
  exit "$code"
}
trap 'finish "$?"' EXIT
trap 'exit 124' TERM INT

install -d -m 0700 /var/lib/hal /run/hal/hal
exec > >(tee -a /var/log/hal-training.log) 2>&1
# The kernel's service manager owns the timers even if this startup process hangs.
if [ "${HAL_MAX_RUN_SECONDS:-0}" -gt 0 ]; then
  systemd-run --unit=hal-hard-stop --on-boot="${HAL_MAX_RUN_SECONDS}s" /sbin/shutdown -h now
  cat > /run/hal/deadline.sh <<'HAL_DEADLINE'
#!/usr/bin/env bash
docker kill hal-training || true
HAL_DEADLINE
  systemd-run --unit=hal-job-deadline --on-boot="$((HAL_MAX_RUN_SECONDS - 900))s" \
    /bin/bash /run/hal/deadline.sh
fi

: "${HAL_GIT_SHA:?missing HAL_GIT_SHA}"
: "${HAL_TRAIN_CMD_B64:?missing HAL_TRAIN_CMD_B64}"
: "${HAL_GCP_PROJECT:?missing HAL_GCP_PROJECT}"
: "${HAL_SECRET_SPECS_B64:?missing HAL_SECRET_SPECS_B64}"
: "${HAL_IMAGE:?missing HAL_IMAGE}"

install -d -m 0700 /run/hal
printf '%s' "$HAL_RESULTS_HELPER_B64" | base64 -d > /run/hal/results.py
printf '%s' "$HAL_R2_HELPER_B64" | base64 -d > /run/hal/hal/r2.py
: > /run/hal/hal/__init__.py
: > /run/hal/job.env
chmod 0600 /run/hal/job.env
while IFS='=' read -r env_name secret_id; do
  [ -n "$env_name" ] || continue
  log "loading ${env_name} from Secret Manager secret ${secret_id}"
  printf '%s=' "$env_name" >> /run/hal/job.env
  gcloud secrets versions access latest --secret="$secret_id" --project="$HAL_GCP_PROJECT" >> /run/hal/job.env
  printf '\n' >> /run/hal/job.env
done < <(printf '%s' "$HAL_SECRET_SPECS_B64" | base64 -d)

if ! command -v docker >/dev/null; then
  log "installing Docker because the selected DLVM image does not include it"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install --yes --no-install-recommends docker.io
  command -v nvidia-ctk >/dev/null || { log "nvidia-ctk is missing from the selected DLVM image"; false; }
  nvidia-ctk runtime configure --runtime=docker
  systemctl enable docker
  systemctl restart docker
fi
nvidia-smi
registry="${HAL_IMAGE%%/*}"
if [[ "$registry" = *.pkg.dev ]]; then
  gcloud auth configure-docker "$registry" --quiet
fi
log "pulling ${HAL_IMAGE}"
docker pull "$HAL_IMAGE"
docker image inspect "$HAL_IMAGE" > /var/lib/hal/container-image.json

# The runtime script lives outside the repository because the exact SHA has not
# been cloned yet. It executes inside the prebuilt HAL image and uses its /opt/venv.
cat > /run/hal/run.sh <<'HAL_RUNTIME'
#!/usr/bin/env bash
set -euo pipefail
log() { echo "[hal-container] $*"; }
trap 'code=$?; log "boot failed at line ${LINENO}"; exit "$code"' ERR

install -d -m 0700 /hal-output/tmp
export TMPDIR=/hal-output/tmp

cd /
rm -rf /opt/hal
log "cloning HAL @ ${HAL_GIT_SHA}"
git clone --quiet "https://github.com/ericyuegu/hal.git" /opt/hal
cd /opt/hal
git checkout --quiet "$HAL_GIT_SHA"
[ "$(git rev-parse HEAD)" = "$HAL_GIT_SHA" ] || { log "HAL checkout SHA mismatch"; exit 1; }
uv sync --locked

if [ "${HAL_SKIP_FIXTURES:-0}" != "1" ]; then
  log "fetching fixtures"
  uv run fetch
fi
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/hal-output/xvfb.log 2>&1 &)
export DISPLAY=:99
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable inside HAL container'"
ulimit -n "$(ulimit -Hn)" || true

cmd="$(printf '%s' "$HAL_TRAIN_CMD_B64" | base64 -d)"
log "training: ${cmd}"
bash -c "$cmd"
HAL_RUNTIME
chmod 0700 /run/hal/run.sh

set +e
install -d -m 0700 /var/lib/hal
docker run --rm --name hal-training --entrypoint bash --gpus all --ipc=host \
  --env-file /run/hal/job.env \
  -e HAL_GIT_SHA -e HAL_TRAIN_CMD_B64 -e HAL_SKIP_FIXTURES \
  -v /var/lib/hal:/hal-output \
  -v /run/hal/run.sh:/run/hal/run.sh:ro \
  "$HAL_IMAGE" /run/hal/run.sh
code=$?
exit "$code"
