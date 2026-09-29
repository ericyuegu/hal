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
commands=(uv Xvfb xsetroot)
if [[ ${HAL_NETPLAY_STREAM:-1} != 0 ]]; then
  : "${HAL_NETPLAY_STREAM_DISPLAY:?set a dedicated NVIDIA Xorg display}"
  commands+=(obs openbox glxinfo pulseaudio)
fi
for command in "${commands[@]}"; do
  if ! command -v "$command" >/dev/null; then
    echo "required command is not installed: $command" >&2
    exit 2
  fi
done

state_dir=${HAL_NETPLAY_STATE_DIR:-"$repo_dir/runs/netplay"}
mkdir -p "$state_dir/replays"
cd "$repo_dir"
uv sync --locked
runner=(uv run hal-netplay-runner \
  --slots "${HAL_NETPLAY_SLOTS:-1}" \
  --slippi-port "${HAL_NETPLAY_SLIPPI_PORT:-51441}" \
  --display-base "${HAL_NETPLAY_DISPLAY_BASE:-100}" \
  --compiled \
  --graphics-backend "${HAL_NETPLAY_GRAPHICS_BACKEND:-Vulkan}" \
  --replay-dir "$state_dir/replays" \
  --status-path "$state_dir/runner-status.json" \
  --git-sha "$HAL_GIT_SHA")
if [[ ${HAL_NETPLAY_STREAM:-1} == 0 ]]; then runner+=(--no-stream); fi
if [[ ${HAL_TWITCH_BANDWIDTH_TEST:-0} == 1 ]]; then runner+=(--twitch-bandwidth-test); fi
exec "${runner[@]}"
