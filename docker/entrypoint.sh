#!/usr/bin/env bash
# Bring up a virtual display so the closed-loop Dolphin eval/roundtrip can get a
# GL context headlessly, then hand off to the requested command. Used for local
# `docker run` and vast.ai's "Docker ENTRYPOINT" launch mode.
set -euo pipefail

if [ -z "${DISPLAY:-}" ]; then
  Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &
  export DISPLAY=:99
fi

exec "$@"
