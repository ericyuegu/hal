#!/usr/bin/env bash
# The existing HAL runtime image omits the toolchain needed by the native simulator.
set -euo pipefail
missing=0
for tool in gcc make readelf; do
  command -v "$tool" >/dev/null || missing=1
done
if [ "$missing" = 1 ]; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install --yes --no-install-recommends gcc make binutils libc6-dev
fi
uv run python experiments/o65/pilot.py "$@"
