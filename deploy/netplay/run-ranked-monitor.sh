#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage: sudo $0 PLAYER_CONTAINER RUN_DIRECTORY" >&2
  exit 2
fi
monitor_player=$1
monitor_run=$(realpath "$2")
monitor_checkout=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
monitor_name="${monitor_player}-monitor"
monitor_output="$monitor_run/monitor"

if [[ $(docker inspect --format '{{.State.Running}}' "$monitor_player") != true ]]; then
  echo "The player is stopped. Start monitoring after the next authorized player start." >&2
  exit 1
fi
for monitor_file in obs-stats.json value.json; do
  test -f "$monitor_run/$monitor_file"
done
monitor_image=$(docker inspect --format '{{.Image}}' "$monitor_player")
mkdir -p "$monitor_output"
chown 65534:65534 "$monitor_output"
chmod 0755 "$monitor_output"

# No GPU, secrets, network, writable player files, or permission to signal its root processes.
docker run --detach --init --name "$monitor_name" \
  --network=none --pid="container:$monitor_player" \
  --user=65534:65534 --cap-drop=ALL --security-opt=no-new-privileges \
  --read-only --memory=96m --cpus=0.25 --pids-limit=32 \
  --log-opt=max-size=1m --log-opt=max-file=2 \
  --mount "type=bind,src=$monitor_run,dst=/ranked,readonly" \
  --mount "type=bind,src=$monitor_output,dst=/monitor" \
  --mount "type=bind,src=$monitor_checkout/hal,dst=/monitor-source/hal,readonly" \
  --env=PYTHONPATH=/monitor-source --env=PYTHONDONTWRITEBYTECODE=1 \
  --workdir=/tmp --entrypoint=/opt/venv/bin/python "$monitor_image" \
  -m hal.scripts.ranked_monitor --run-dir /ranked --output /monitor --processes
