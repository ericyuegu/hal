#!/usr/bin/env bash
# vast.ai template On-start script. The instance is stateless: datasets are
# fetched from R2 into the (ephemeral) container fs, and checkpoints are pushed
# back to R2 by the training loop — so there's no persistent volume to manage.
# vast skips the image ENTRYPOINT in Jupyter/SSH modes, so we do its work here.
set -euo pipefail

cd /opt/hal
uv run fetch                      # AWS_*/R2 env vars; idempotent (sha match)

pgrep -x Xvfb >/dev/null || (Xvfb :99 -screen 0 1280x720x24 >/tmp/xvfb.log 2>&1 &)
export DISPLAY=:99
echo "[on-start] hal ready in /opt/hal (DISPLAY=:99; checkpoints -> R2)"
