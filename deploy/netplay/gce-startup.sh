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
[[ $slots =~ ^([1-9]|1[0-6])$ ]] || { log "invalid slot count"; exit 2; }
[[ $drain_timeout =~ ^[1-9][0-9]*$ ]] || { log "invalid drain timeout"; exit 2; }

command -v docker >/dev/null || { log "Docker is missing from the selected GPU image"; exit 1; }
command -v nvidia-smi >/dev/null || { log "the NVIDIA driver is missing from the selected GPU image"; exit 1; }
systemctl start docker
docker info >/dev/null
nvidia-smi

install -d -m 0700 /run/hal-netplay
install -d -m 0755 /var/cache/hal-netplay /var/lib/hal-netplay/replays

# CUDA and NVENC alone do not supply the GLX libraries needed by Dolphin.
driver_version=$(dpkg-query -W -f='${Version}' libnvidia-compute-580-server)
[[ $(nvidia-smi --query-gpu=driver_version --format=csv,noheader) == "${driver_version%%-*}" ]] || {
  log "expected one GPU with the installed NVIDIA 580 server driver"; exit 1;
}
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  xserver-xorg-core "xserver-xorg-video-nvidia-580-server=$driver_version" \
  "libnvidia-gl-580-server=$driver_version" "libnvidia-common-580-server=$driver_version" \
  xauth mesa-utils
systemctl restart nvidia-cdi-refresh.service
pci=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader)
bus_id=$(python3 -c 'import sys; domain,bus,tail=sys.argv[1].split(":"); device,function=tail.split("."); print(f"PCI:{int(bus,16)}@{int(domain,16)}:{int(device,16)}:{int(function,16)}")' "$pci")
cat > /etc/X11/hal-netplay.conf <<EOF
Section "ServerFlags"
    Option "AutoAddDevices" "False"
    Option "AutoAddGPU" "False"
EndSection
Section "Device"
    Identifier "HAL GPU"
    Driver "nvidia"
    BusID "$bus_id"
    Option "AllowEmptyInitialConfiguration" "True"
EndSection
Section "Screen"
    Identifier "HAL Screen"
    Device "HAL GPU"
    DefaultDepth 24
    SubSection "Display"
        Depth 24
        Virtual 1920 1080
    EndSubSection
EndSection
EOF
python3 - <<'PY'
import secrets
import subprocess
from pathlib import Path

path = Path("/run/hal-netplay/Xauthority")
path.touch(mode=0o600)
subprocess.run(["xauth", "-f", str(path), "add", ":90", "MIT-MAGIC-COOKIE-1", secrets.token_hex(16)], check=True)
entries = subprocess.check_output(["xauth", "-f", str(path), "nlist"]).decode()
# The container has its own hostname. Keep the cookie, widen only its family.
entries = "".join("ffff" + line[4:] + "\n" for line in entries.splitlines())
subprocess.run(["xauth", "-f", str(path), "nmerge", "-"], input=entries.encode(), check=True)
PY
cat > /etc/systemd/system/hal-netplay-display.service <<'EOF'
[Unit]
Description=HAL NVIDIA display
After=network.target
[Service]
ExecStart=/usr/lib/xorg/Xorg :90 -config /etc/X11/hal-netplay.conf -auth /run/hal-netplay/Xauthority -nolisten tcp -noreset
Restart=on-failure
RestartSec=2
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now hal-netplay-display.service
for _ in {1..100}; do
  if DISPLAY=:90 XAUTHORITY=/run/hal-netplay/Xauthority glxinfo -B > /run/hal-netplay/glxinfo 2>/dev/null; then break; fi
  sleep 0.1
done
grep -q 'OpenGL vendor string: NVIDIA Corporation' /run/hal-netplay/glxinfo || {
  log "NVIDIA Xorg display is not ready"; exit 1;
}
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
After=docker.service network-online.target hal-netplay-display.service
Wants=network-online.target
Requires=docker.service hal-netplay-display.service

[Service]
Restart=always
RestartSec=5
TimeoutStopSec=${unit_timeout}
ExecStartPre=-/usr/bin/docker rm -f hal-netplay-runner
ExecStart=/usr/bin/docker run --rm --name hal-netplay-runner --gpus all --ipc=host --env-file /run/hal-netplay/runner.env -e APPIMAGE_EXTRACT_AND_RUN=1 -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video,display -e HAL_NETPLAY_STREAM_DISPLAY=:90 -e XAUTHORITY=/run/hal-netplay/Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /run/hal-netplay/Xauthority:/run/hal-netplay/Xauthority:ro -v /var/cache/hal-netplay:/root/.cache/hal-netplay -v /var/lib/hal-netplay:/var/lib/hal-netplay ${image} hal-netplay-runner --slots ${slots} --compiled --graphics-backend OGL --drain-timeout ${drain_timeout} --replay-dir /var/lib/hal-netplay/replays --status-path /var/lib/hal-netplay/runner-status.json --git-sha ${git_sha}
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
