#!/usr/bin/env bash
# Launch one bounded debug pilot. The generic launcher checks publication before create.
set -euo pipefail
cd "$(dirname "$0")/.."
sha="$(git rev-parse HEAD)"
name="hal-o65-proxy-${sha:0:8}"
image="us-west1-docker.pkg.dev/centering-star-502613-k3/hal-netplay/hal-netplay-runner@sha256:93b281d12c0614516eca8b5a508dae2723da80e53933e9616bb08d6e926f3016"
if [ "$#" -gt 1 ] || { [ "$#" = 1 ] && [ "$1" != --dry-run ]; }; then
  echo 'Usage: bash scripts/launch_o65.sh [--dry-run]' >&2
  exit 2
fi
if [ "$#" = 0 ]; then
  uv run python scripts/o65_delivery.py --action check-remote
fi
uv run python scripts/launch_gce.py \
  --project centering-star-502613-k3 --zone us-central1-b --name "$name" \
  --machine-type g4-standard-48 --gpu-count 1 \
  --disk 500 --disk-type hyperdisk-balanced --image "$image" \
  --no-spot --max-run-hours 24 --skip-fixtures \
  --results-prefix "o65/pilot-results/$name" "$@" \
  -- bash docker/o65-pilot.sh --output /hal-output/pilot
