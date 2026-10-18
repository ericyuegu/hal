# Run only through the transient unit from scripts/launch_o65_existing.py.
set -euo pipefail
umask 077
job="/var/lib/hal-o65/${HAL_RUN_NAME}"
output="$job/output"
control="$job/control"
scratch="$job/scratch"
training="${HAL_RUN_NAME}-training"
uploader="${HAL_RUN_NAME}-upload"
export DOCKER_CONFIG="$control/docker"

log() { echo "[hal-o65] $*"; }

prepare() {
  install -d -m 0700 "$output" "$control/hal" "$scratch"
  printf '%s' "$HAL_RESULTS_HELPER_B64" | base64 -d > "$control/results.py"
  printf '%s' "$HAL_R2_HELPER_B64" | base64 -d > "$control/hal/r2.py"
  : > "$control/hal/__init__.py"
}

credentials() {
  : > "$control/job.env.partial"
  while IFS='=' read -r env_name secret_id; do
    printf '%s=' "$env_name" >> "$control/job.env.partial"
    timeout --kill-after=2s 10s gcloud secrets versions access latest \
      --secret="$secret_id" --project="$HAL_GCP_PROJECT" >> "$control/job.env.partial"
    printf '\n' >> "$control/job.env.partial"
  done < <(printf '%s' "$HAL_SECRET_SPECS_B64" | base64 -d)
  mv "$control/job.env.partial" "$control/job.env"
}

remove_container() {
  timeout --kill-after=5s 20s docker rm -f "$1" 2>/dev/null
}

case "${1:?expected run, finish, upload or clean}" in
run)
  prepare
  exec > >(tee -a "$output/job.log") 2>&1
  exec 9>/var/lib/hal-o65/pilot.lock
  flock -n 9 || { log 'another pilot is running'; exit 1; }
  for unit in hal-netplay-runner.service hal-netplay-health.service; do
    state="$(systemctl show "$unit" --property=ActiveState --value)"
    case "$state" in inactive|failed) ;; *) log "${unit} is ${state}; cleanup is pending"; exit 1 ;; esac
  done
  containers="$(docker ps --format '{{.Names}}')"
  if printf '%s\n' "$containers" | grep -q '^hal-netplay-'; then
    log 'netplay containers are still running; cleanup is pending'
    exit 1
  fi
  docker info --format '{{json .Runtimes}}' | grep -q '"nvidia":'
  nvidia-smi
  # Reuse the cached image. Refuse a download with an unmeasured disk requirement.
  docker image inspect "$HAL_IMAGE" > "$output/container-image.json"
  available="$(df -B1 --output=avail "$job" | tail -1)"
  if [ "$available" -lt 42949672960 ]; then
    log "need at least 40 GiB free; found ${available} bytes"
    exit 1
  fi
  credentials
  cat > "$control/run.sh" <<'HAL_RUNTIME'
#!/usr/bin/env bash
set -euo pipefail
install -d -m 0700 /hal-scratch/tmp
export TMPDIR=/hal-scratch/tmp
export TORCHINDUCTOR_CACHE_DIR=/hal-scratch/inductor
export TRITON_CACHE_DIR=/hal-scratch/triton
cd /
rm -rf /opt/hal
git clone --quiet https://github.com/ericyuegu/hal.git /opt/hal
cd /opt/hal
git checkout --quiet --detach "$HAL_GIT_SHA"
[ "$(git rev-parse HEAD)" = "$HAL_GIT_SHA" ]
uv sync --locked
Xvfb :99 -screen 0 1280x720x24 >/hal-output/xvfb.log 2>&1 &
export DISPLAY=:99
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable inside HAL container'"
bash docker/o65-pilot.sh --output /hal-output/pilot --checkpoint-retention compact
HAL_RUNTIME
  docker run --rm --pull=never --name "$training" --entrypoint bash --gpus all --ipc=host \
    --log-driver=local --log-opt max-size=10m --log-opt max-file=2 \
    --env-file "$control/job.env" -e HAL_GIT_SHA \
    -v "$output:/hal-output" -v "$scratch:/hal-scratch" -v "$control:/hal-control:ro" \
    "$HAL_IMAGE" /hal-control/run.sh
  ;;
finish)
  prepare
  printf '%s\n' "${SERVICE_RESULT:-unknown}" > "$output/service-result"
  printf '%s %s\n' "${EXIT_CODE:-unknown}" "${EXIT_STATUS:-unknown}" > "$output/job-exit-code"
  set +e
  # systemd reserves 900 seconds after its 85,500-second runtime limit.
  timeout --kill-after=5s 810s /bin/bash "$0" upload
  upload_code=$?
  printf '%s\n' "$upload_code" > "$output/upload-exit-code"
  timeout --kill-after=5s 60s /bin/bash "$0" clean
  clean_code=$?
  if [ "$upload_code" != 0 ]; then
    log "RESULT UPLOAD FAILED (${upload_code}); results remain in ${output}"
    exit "$upload_code"
  fi
  exit "$clean_code"
  ;;
upload)
  prepare
  # Docker containers use a different cgroup from the service's Docker client.
  if docker container inspect "$training" >/dev/null 2>&1; then
    remove_container "$training"
  fi
  if [ ! -f "$control/job.env" ]; then credentials; fi
  log "uploading o65/pilot-results/${HAL_RUN_NAME}/"
  timeout --kill-after=5s 720s docker run --rm --pull=never --name "$uploader" --entrypoint python \
    --log-driver=local --log-opt max-size=10m --log-opt max-file=2 \
    --env-file "$control/job.env" -e PYTHONPATH=/hal-control \
    -v "$control:/hal-control:ro" -v "$output:/hal-output" \
    "$HAL_IMAGE" /hal-control/results.py upload --root /hal-output \
    --prefix "o65/pilot-results/$HAL_RUN_NAME"
  ;;
clean)
  set +e
  code=0
  for container in "$training" "$uploader"; do
    if docker container inspect "$container" >/dev/null 2>&1; then
      remove_container "$container" || code=1
    fi
  done
  rm -rf -- "$scratch" "$output/pilot/delivery" "$output/pilot/tmp" "$output/.upload" "$control/docker"
  rm -f -- "$control/job.env" "$control/job.env.partial"
  exit "$code"
  ;;
*) exit 2 ;;
esac
