#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  echo "usage: deploy/netplay/run-host.sh [environment-file]"
  exit 0
fi
if (( $# > 1 )); then
  echo "run-host.sh accepts at most one environment file" >&2
  exit 2
fi
env_file=${1:-"$script_dir/.env"}
if [[ ! -f $env_file ]]; then
  echo "environment file does not exist: $env_file" >&2
  exit 2
fi

set -a
source "$env_file"
set +a
required=(HAL_GIT_SHA HAL_NETPLAY_API_URL HAL_NETPLAY_RUNNER_TOKEN AWS_ENDPOINT_URL AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_BUCKET)
for variable in "${required[@]}"; do
  if [[ -z ${!variable:-} ]]; then
    echo "environment variable is not set: $variable" >&2
    exit 2
  fi
done
if [[ $HAL_NETPLAY_API_URL == https://* ]]; then
  for variable in CF_ACCESS_CLIENT_ID CF_ACCESS_CLIENT_SECRET; do
    if [[ -z ${!variable:-} ]]; then
      echo "environment variable is not set: $variable" >&2
      exit 2
    fi
  done
fi
if ! command -v uv >/dev/null; then
  echo "required command is not installed: uv" >&2
  exit 2
fi

state_dir=${HAL_NETPLAY_STATE_DIR:-"$repo_dir/runs/netplay"}
mkdir -p "$state_dir/replays"
cd "$repo_dir"
uv sync --locked
exec uv run hal-netplay-runner \
  --slots "${HAL_NETPLAY_SLOTS:-1}" \
  --slippi-port "${HAL_NETPLAY_SLIPPI_PORT:-51441}" \
  --compiled \
  --graphics-backend "${HAL_NETPLAY_GRAPHICS_BACKEND:-Vulkan}" \
  --replay-dir "$state_dir/replays" \
  --status-path "$state_dir/runner-status.json" \
  --git-sha "$HAL_GIT_SHA"
