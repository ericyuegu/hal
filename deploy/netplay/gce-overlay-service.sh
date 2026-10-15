#!/usr/bin/env bash
set -euo pipefail

# Run on the guest after the runner image has been pulled.
image=${1:?usage: gce-overlay-service.sh IMAGE}
root=/var/lib/hal-overlay
install -d -m 0755 "$root/releases"
if [[ ! -L $root/current ]]; then
  sha=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image")
  [[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "image has no source revision" >&2; exit 1; }
  release="$root/releases/seed-$sha"
  install -d -m 0755 "$release"
  docker run --rm --entrypoint tar "$image" \
    -C /opt/hal -cf - hal deploy/netplay/overlay-fonts.conf \
    -C /usr/local/share/fonts schibsted-grotesk | tar -xf - -C "$release"
  ln -sfn schibsted-grotesk "$release/fonts"
  printf '%s\n' "$sha" > "$release/git-sha"
  ln -s "$release" "$root/current"
fi

cat > /etc/systemd/system/hal-netplay-overlay.service <<EOF
[Unit]
Description=HAL Twitch overlay
After=docker.service hal-netplay-runner.service
Requires=docker.service hal-netplay-runner.service
PartOf=hal-netplay-runner.service

[Service]
Restart=always
RestartSec=3
TimeoutStopSec=20
ExecStartPre=-/usr/bin/docker rm -f hal-netplay-overlay
ExecStart=/usr/bin/docker run --rm --name hal-netplay-overlay --gpus all --network container:hal-netplay-runner -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video,display -e DISPLAY=:90 -e XAUTHORITY=/run/hal-netplay/Xauthority -e PYTHONPATH=/opt/hal-overlay/current -e FONTCONFIG_FILE=/opt/hal-overlay/current/deploy/netplay/overlay-fonts.conf -e HAL_OVERLAY_RELEASE_FILE=/opt/hal-overlay/current/git-sha -v /tmp/.X11-unix/X90:/tmp/.X11-unix/X90:ro -v /run/hal-netplay/Xauthority:/run/hal-netplay/Xauthority:ro -v /var/lib/hal-netplay:/var/lib/hal-netplay -v /var/lib/hal-overlay:/opt/hal-overlay:ro --workdir /tmp ${image} sh -c 'fc-cache -f /opt/hal-overlay/current/fonts && exec python -m hal.netplay_service.overlay --display :90 --status-path /var/lib/hal-netplay/runner-status.json'
ExecStop=/usr/bin/docker stop --time=15 hal-netplay-overlay

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable hal-netplay-overlay.service
systemctl restart hal-netplay-overlay.service
