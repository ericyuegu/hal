#!/usr/bin/env bash
set -euo pipefail

log() { echo "[hal-netplay] $*"; }
metadata() {
  curl --fail --silent --show-error \
    -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

project=$(metadata hal-netplay-project)
image=$(metadata hal-netplay-image)
git_sha=$(metadata hal-netplay-git-sha)
secret=$(metadata hal-netplay-secret)
slots=$(metadata hal-netplay-slots)
drain_timeout=$(metadata hal-netplay-drain-timeout)

[[ $git_sha =~ ^[0-9a-f]{40}$ ]] || { log "invalid Git SHA"; exit 2; }
[[ $slots =~ ^[1-8]$ ]] || { log "invalid slot count"; exit 2; }
[[ $drain_timeout =~ ^[1-9][0-9]*$ ]] || { log "invalid drain timeout"; exit 2; }

command -v docker >/dev/null || { log "Docker is missing from the selected GPU image"; exit 1; }
command -v nvidia-smi >/dev/null || { log "the NVIDIA driver is missing from the selected GPU image"; exit 1; }
systemctl start docker
docker info >/dev/null
nvidia-smi

install -d -m 0700 /run/hal-netplay
install -d -m 0755 /var/cache/hal-netplay /var/lib/hal-netplay/replays
log "reading runner environment from Secret Manager"
gcloud secrets versions access latest --secret="$secret" --project="$project" \
  > /run/hal-netplay/runner.env
chmod 0600 /run/hal-netplay/runner.env
if [[ ! -s /run/hal-netplay/runner.env ]]; then
  log "the runner environment secret is empty"
  exit 1
fi

registry_host=${image%%/*}
if [[ $registry_host == *.pkg.dev ]]; then
  gcloud auth configure-docker "$registry_host" --quiet
fi
log "pulling $image"
docker pull "$image"

stop_seconds=$((drain_timeout + 30))
unit_timeout=$((drain_timeout + 60))
cat > /etc/systemd/system/hal-netplay-runner.service <<EOF
[Unit]
Description=HAL netplay runner
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service

[Service]
Restart=always
RestartSec=5
TimeoutStopSec=${unit_timeout}
ExecStartPre=-/usr/bin/docker rm -f hal-netplay-runner
ExecStart=/usr/bin/docker run --rm --name hal-netplay-runner --gpus all --ipc=host --env-file /run/hal-netplay/runner.env -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video -v /var/cache/hal-netplay:/root/.cache/hal-netplay -v /var/lib/hal-netplay:/var/lib/hal-netplay ${image} hal-netplay-runner --slots ${slots} --compiled --drain-timeout ${drain_timeout} --replay-dir /var/lib/hal-netplay/replays --status-path /var/lib/hal-netplay/runner-status.json --git-sha ${git_sha}
ExecStop=/usr/bin/docker stop --time=${stop_seconds} hal-netplay-runner

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/hal-netplay-health.service <<EOF
[Unit]
Description=HAL netplay host health endpoint
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service

[Service]
Restart=always
RestartSec=5
ExecStartPre=-/usr/bin/docker rm -f hal-netplay-health
ExecStart=/usr/bin/docker run --rm --name hal-netplay-health --network host -v /var/lib/hal-netplay:/var/lib/hal-netplay:ro ${image} python -m hal.netplay_service.host_health --status-path /var/lib/hal-netplay/runner-status.json --host 0.0.0.0 --port 9101 --max-age 10
ExecStop=/usr/bin/docker stop --time=10 hal-netplay-health

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now hal-netplay-health.service
systemctl enable --now hal-netplay-runner.service
log "runner service started"
