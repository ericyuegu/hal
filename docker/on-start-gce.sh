# Google Compute Engine startup lifecycle. scripts/launch_gce.py prepends the
# non-secret HAL_* configuration before placing this script in instance metadata.
# Secret values are fetched at boot from Secret Manager with the VM service account.
#
# The repository, replay shards, fixtures, and compiler caches live under
# /var/lib/hal on the persistent boot disk, so a kept-alive VM can run later
# commits with `scripts/gce_run.py` without re-downloading or recompiling.
set -euo pipefail

log() { echo "[hal-gce] $*"; }

finish() {
  code="$1"
  if [ "${HAL_KEEP_ALIVE:-0}" = "1" ]; then
    log "job exited ${code}; HAL_KEEP_ALIVE=1, leaving VM running"
    exit "$code"
  fi
  rm -f /run/hal/job.env
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
log "pulling ${HAL_IMAGE}"
docker pull "$HAL_IMAGE"
install -d -m 0755 /var/lib/hal /var/lib/hal/repo /var/lib/hal/cache /var/lib/hal/logs
printf '%s\n' "$HAL_IMAGE" > /var/lib/hal/image

# The container runtime lives outside the repository because the exact SHA has
# not been checked out yet. It runs inside the prebuilt image's /opt/venv.
cat > /var/lib/hal/container-run.sh <<'HAL_RUNTIME'
#!/usr/bin/env bash
set -euo pipefail
log() { echo "[hal-container] $*"; }
trap 'code=$?; log "boot failed at line ${LINENO}"; exit "$code"' ERR

cd /opt/hal
if [ ! -d .git ]; then
  log "cloning HAL"
  git clone --quiet "https://github.com/ericyuegu/hal.git" /tmp/hal-clone
  mv /tmp/hal-clone/.git .git
  rm -rf /tmp/hal-clone
fi
git config --global --add safe.directory /opt/hal
log "checking out HAL @ ${HAL_GIT_SHA}"
git fetch --quiet origin
git reset --quiet --hard "$HAL_GIT_SHA"
# Ignored paths (data/, runs/) hold the warm shards, fixtures, and run outputs.
git clean --quiet -fd
uv sync --locked

log "fetching fixtures"
uv run fetch
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &)
export DISPLAY=:99
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable inside HAL container'"
ulimit -n "$(ulimit -Hn)" || true

cmd="$(printf '%s' "$HAL_TRAIN_CMD_B64" | base64 -d)"
log "running: ${cmd}"
bash -c "$cmd"
HAL_RUNTIME

# hal-run SHA CMD_B64 runs one command in the HAL image against the warm disk.
# It holds a lock so two jobs never share the GPUs, and records the exit code
# beside the log so a detached client can report it.
cat > /usr/local/sbin/hal-run <<'HAL_RUN'
#!/usr/bin/env bash
set -euo pipefail
[ "$#" -eq 2 ] || { echo "usage: hal-run GIT_SHA CMD_B64" >&2; exit 2; }
exec 9>/var/lib/hal/run.lock
flock -n 9 || { echo "[hal-run] another job holds /var/lib/hal/run.lock" >&2; exit 75; }
[ -f /run/hal/job.env ] || { echo "[hal-run] /run/hal/job.env is missing; reboot to reload secrets" >&2; exit 78; }
log_file=/var/lib/hal/logs/job.log
rm -f /var/lib/hal/logs/job.exit
: > "$log_file"
stop_container() { docker stop --time 60 hal-job >/dev/null 2>&1 || true; }
trap stop_container TERM INT
set +e
# A private /dev/shm disappears with the container, so a killed job cannot
# leave Streaming or DataLoader shared memory behind for the next one.
# A systemd daemon-reload, such as the one the guest agent runs when OS Login
# configures SSH, drops the device cgroup rules that --gpus adds. New CUDA
# contexts in the running container then fail with "Failed to initialize NVML:
# Unknown Error" (nvidia-container-toolkit, cgroup v2, driver 580). Explicit
# --device rules survive the reload.
gpu_devices=()
for device in /dev/nvidia*; do [ -c "$device" ] && gpu_devices+=(--device "$device"); done
docker run --rm --name hal-job --gpus all "${gpu_devices[@]}" --shm-size=128g \
  --env-file /run/hal/job.env \
  -e HAL_GIT_SHA="$1" -e HAL_TRAIN_CMD_B64="$2" \
  -e TORCHINDUCTOR_CACHE_DIR=/opt/hal-cache/torchinductor \
  -e TORCHINDUCTOR_FX_GRAPH_CACHE=1 -e TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
  -e TRITON_CACHE_DIR=/opt/hal-cache/triton \
  -e CUDA_CACHE_PATH=/opt/hal-cache/cuda \
  -e UV_CACHE_DIR=/opt/hal-cache/uv \
  -v /var/lib/hal/repo:/opt/hal \
  -v /var/lib/hal/cache:/opt/hal-cache \
  -v /var/lib/hal/container-run.sh:/run/hal/container-run.sh:ro \
  "$(cat /var/lib/hal/image)" bash /run/hal/container-run.sh > >(tee -a "$log_file") 2>&1 &
wait $!
code=$?
set -e
echo "$code" > /var/lib/hal/logs/job.exit
exit "$code"
HAL_RUN
chmod 0755 /usr/local/sbin/hal-run

# A restart of a kept-alive VM reloads secrets and tooling but does not rerun
# the launch command; scripts/gce_run.py starts later jobs explicitly.
launch_marker="/var/lib/hal/launched-$(printf '%s %s' "$HAL_GIT_SHA" "$HAL_TRAIN_CMD_B64" | sha256sum | cut -c1-16)"
if [ -e "$launch_marker" ]; then
  log "launch command already ran on this disk; waiting for scripts/gce_run.py"
  exit 0
fi
touch "$launch_marker"
set +e
# The same systemd unit as scripts/gce_run.py, so --attach and --stop apply.
systemd-run --quiet --collect --wait --pipe --unit=hal-job /usr/local/sbin/hal-run "$HAL_GIT_SHA" "$HAL_TRAIN_CMD_B64"
code=$?
set -e
finish "$code"
