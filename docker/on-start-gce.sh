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
: "${HAL_LOCAL_SSD_COUNT:?missing HAL_LOCAL_SSD_COUNT}"

install -d -m 0700 /run/hal
: > /run/hal/job.env
chmod 0600 /run/hal/job.env
# `|| [ -n "$env_name" ]`: read returns non-zero on a final line with no trailing newline,
# which would drop that secret. The launcher terminates every line, so this is belt-and-braces.
loaded=0
while IFS='=' read -r env_name secret_id || [ -n "$env_name" ]; do
  [ -n "$env_name" ] || continue
  log "loading ${env_name} from Secret Manager secret ${secret_id}"
  printf '%s=' "$env_name" >> /run/hal/job.env
  gcloud secrets versions access latest --secret="$secret_id" --project="$HAL_GCP_PROJECT" >> /run/hal/job.env
  printf '\n' >> /run/hal/job.env
  loaded=$((loaded + 1))
done < <(printf '%s' "$HAL_SECRET_SPECS_B64" | base64 -d)
# Counted in the loop, not from job.env: a secret value may itself contain newlines.
log "loaded ${loaded} secrets"

# Stripe the Local SSDs into the dataset cache mount. These devices are why the run keeps up:
# the train split is read once per shuffled epoch, so page cache does not help and a
# persistent disk of this size cannot sustain the read rate. Assert the expected count rather
# than falling back to the boot disk, which would silently train at a third of the throughput.
mapfile -t ssd_devices < <(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null || true)
if [ "${#ssd_devices[@]}" -ne "$HAL_LOCAL_SSD_COUNT" ]; then
  log "FATAL: expected ${HAL_LOCAL_SSD_COUNT} Local SSD(s), found ${#ssd_devices[@]}"
  false
fi
if [ "$HAL_LOCAL_SSD_COUNT" -gt 1 ]; then
  log "striping ${HAL_LOCAL_SSD_COUNT} Local SSDs into /dev/md0"
  command -v mdadm >/dev/null || { apt-get update -qq && apt-get install -y -qq mdadm; }
  mdadm --create /dev/md0 --level=0 --raid-devices="$HAL_LOCAL_SSD_COUNT" "${ssd_devices[@]}"
  data_device=/dev/md0
else
  data_device="${ssd_devices[0]}"
fi
# lazy_itable_init=0 pays the inode-table write up front so it does not compete with the
# first epoch's reads; discard lets the SSD start with a clean map. GCE's Local SSD docs also
# suggest nobarrier, but it is deprecated for ext4 and a rejected mount option would fail the
# whole boot for a marginal gain, so it is left off.
mkfs.ext4 -F -E lazy_itable_init=0,discard "$data_device"
install -d -m 0777 /mnt/hal-data
mount -o discard,defaults "$data_device" /mnt/hal-data
chmod 0777 /mnt/hal-data
log "dataset cache: $(df -h --output=size,avail /mnt/hal-data | tail -1) on ${data_device}"

command -v docker >/dev/null || { log "Docker is missing from the selected DLVM image"; false; }
# The DLVM image can still be installing its driver when the metadata startup script fires,
# so nvidia-smi is not immediately truthful. Wait rather than tripping the ERR trap on a race.
for _ in $(seq 60); do
  nvidia-smi >/dev/null 2>&1 && break
  log "waiting for the NVIDIA driver"
  sleep 10
done
nvidia-smi
log "pulling ${HAL_IMAGE}"
docker pull "$HAL_IMAGE"
# Prove the container runtime can actually reach the GPU before spending 20 minutes on
# clone + fetch only to fail the CUDA assert.
docker run --rm --gpus all "$HAL_IMAGE" nvidia-smi -L

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

# Every byte the data path touches — streamed MDS shards, the ISO, the Dolphin AppImage —
# lands under the repo's data/, so point that at the Local SSD. A symlink (rather than a
# --cfg.data-root override) keeps streams.remote_for_local() resolving, since hal/paths.py
# uses os.path.abspath, not resolve.
ln -s /mnt/hal-data /opt/hal/data
log "data/ -> /mnt/hal-data ($(df -h --output=avail /mnt/hal-data | tail -1 | tr -d ' ') free)"

# --ipc=host means /dev/shm is the host's (~half of RAM), but assert it: an undersized
# /dev/shm only surfaces as a DataLoader worker dying at step 0, wasting the whole boot.
shm_mb=$(df -m /dev/shm | awk 'NR==2 {print $2}')
log "/dev/shm = ${shm_mb}MB"
[ "${shm_mb:-0}" -ge 1024 ] || { log "FATAL: /dev/shm ${shm_mb}MB < 1GB"; false; }

log "fetching fixtures + dataset stats"
uv run fetch
uv run python -c "from hal import streams; [streams.pull_stats(s) for s in streams.ALL]"
pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &)
export DISPLAY=:99
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable inside HAL container'"
ulimit -n "$(ulimit -Hn)" || true

cmd="$(printf '%s' "$HAL_TRAIN_CMD_B64" | base64 -d)"
log "training: ${cmd}"
bash -c "$cmd"
HAL_RUNTIME
chmod 0700 /run/hal/run.sh

# Under HAL_KEEP_ALIVE the container is left behind (`docker logs hal-train`, `docker commit`,
# `docker exec` into a still-populated /opt/hal) because that is the whole point of the flag.
run_flags=(--rm)
if [ "${HAL_KEEP_ALIVE:-0}" = "1" ]; then
  run_flags=()
fi

set +e
docker run "${run_flags[@]}" --name hal-train --gpus all --ipc=host \
  --env-file /run/hal/job.env \
  -e HAL_GIT_SHA -e HAL_TRAIN_CMD_B64 \
  -v /run/hal/run.sh:/run/hal/run.sh:ro \
  -v /mnt/hal-data:/mnt/hal-data \
  "$HAL_IMAGE" bash /run/hal/run.sh 2>&1 | tee /var/log/hal-training.log
code=${PIPESTATUS[0]}
# Keep the creds for an interactive `--resume` peek when the VM is deliberately left up;
# otherwise they die with the job.
[ "${HAL_KEEP_ALIVE:-0}" = "1" ] || rm -f /run/hal/job.env
finish "$code"
