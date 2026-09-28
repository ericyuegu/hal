#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  echo "usage: deploy/netplay/run-local.sh [environment-file]"
  exit 0
fi
if (( $# > 1 )); then
  echo "run-local.sh accepts at most one environment file" >&2
  exit 2
fi
env_file=${1:-"$script_dir/.env"}
if [[ ! -f $env_file ]]; then
  echo "environment file does not exist: $env_file" >&2
  exit 2
fi
for command in flock npm uv curl sha256sum; do
  if ! command -v "$command" >/dev/null; then
    echo "required command is not installed: $command" >&2
    exit 2
  fi
done

set -a
source "$env_file"
set +a
state_dir=${HAL_NETPLAY_STATE_DIR:-"$repo_dir/runs/netplay/local"}
lock_file=${HAL_NETPLAY_LOCAL_LOCK:-"$state_dir/run-local.lock"}
mkdir -p "$state_dir"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "local netplay is already running" >&2
  exit 2
fi

runner_token=${HAL_NETPLAY_RUNNER_TOKEN:-dev-runner-token}
admin_token=${HAL_NETPLAY_ADMIN_TOKEN:-dev-admin-token}
local_stream=${HAL_NETPLAY_LOCAL_STREAM:-0}
if [[ $local_stream == 1 && ${HAL_TWITCH_BANDWIDTH_TEST:-0} != 1 ]]; then
  echo "local streaming requires HAL_TWITCH_BANDWIDTH_TEST=1" >&2
  exit 2
fi
stream_key=${HAL_NETPLAY_LOCAL_TWITCH_KEY:-local-stream-disabled}
commands=(Xvfb xsetroot)
if [[ $local_stream == 1 ]]; then commands+=(ffmpeg pulseaudio); fi
for command in "${commands[@]}"; do
  if ! command -v "$command" >/dev/null; then
    echo "required command is not installed: $command" >&2
    exit 2
  fi
done
runner_digest=$(printf %s "$runner_token" | sha256sum | cut -d' ' -f1)
admin_digest=$(printf %s "$admin_token" | sha256sum | cut -d' ' -f1)
process_groups=()
stop() {
  trap - EXIT INT TERM
  for group in "${process_groups[@]}"; do kill -TERM -- "-$group" 2>/dev/null || true; done
  for _ in {1..50}; do
    local alive=false
    for group in "${process_groups[@]}"; do kill -0 -- "-$group" 2>/dev/null && alive=true; done
    [[ $alive == false ]] && break
    sleep 0.1
  done
  for group in "${process_groups[@]}"; do kill -KILL -- "-$group" 2>/dev/null || true; done
  wait "${process_groups[@]}" 2>/dev/null || true
}
trap stop EXIT INT TERM

set -m
cd "$repo_dir/web/netplay-api"
npm exec wrangler dev -- --local --ip 127.0.0.1 --port 8787 \
  --persist-to "$state_dir/worker" \
  --var "RUNNER_TOKEN_SHA256:$runner_digest" \
  --var "ADMIN_TOKEN_SHA256:$admin_digest" \
  --var "TWITCH_STREAM_KEY:$stream_key" </dev/null &
process_groups+=("$!")
for _ in {1..120}; do
  curl --fail --silent http://127.0.0.1:8787/v1/capacity >/dev/null 2>&1 && break
  kill -0 -- "-${process_groups[0]}" 2>/dev/null || exit 1
  sleep 0.25
done
curl --fail --silent http://127.0.0.1:8787/v1/capacity >/dev/null

cd "$repo_dir/web/netplay"
npm run dev -- --host 127.0.0.1 --port 3000 </dev/null &
process_groups+=("$!")

cd "$repo_dir"
runner=(uv run hal-netplay-runner --slots "${HAL_NETPLAY_SLOTS:-1}" --compiled --git-sha "${HAL_GIT_SHA:?set HAL_GIT_SHA}")
runner+=(--display-base "${HAL_NETPLAY_DISPLAY_BASE:-100}")
if [[ -n ${HAL_NETPLAY_LOCAL_ASSETS:-} ]]; then runner+=(--local-assets "$HAL_NETPLAY_LOCAL_ASSETS"); fi
if [[ $local_stream == 0 ]]; then runner+=(--no-stream); fi
if [[ ${HAL_TWITCH_BANDWIDTH_TEST:-0} == 1 ]]; then runner+=(--twitch-bandwidth-test); fi
HAL_NETPLAY_API_URL=http://127.0.0.1:8787 HAL_NETPLAY_RUNNER_TOKEN=$runner_token "${runner[@]}" </dev/null &
process_groups+=("$!")
set +m

wait -n "${process_groups[@]}"
