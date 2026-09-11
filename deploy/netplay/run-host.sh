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

required_variables=(
  HAL_GIT_SHA
  HAL_NETPLAY_POLICY
  HAL_NETPLAY_USER_JSON_A
  HAL_ISO_PATH
  HAL_NETPLAY_EMULATOR_PATH
  HAL_NETPLAY_ALLOWED_ORIGINS
  HAL_NETPLAY_ALLOWED_HOSTS
  CLOUDFLARE_TUNNEL_TOKEN
  AWS_ENDPOINT_URL
  AWS_ACCESS_KEY_ID
  AWS_SECRET_ACCESS_KEY
  AWS_BUCKET
)
for variable in "${required_variables[@]}"; do
  if [[ -z ${!variable:-} ]]; then
    echo "environment variable is not set: $variable" >&2
    exit 2
  fi
done
for command in uv xvfb-run cloudflared; do
  if ! command -v "$command" >/dev/null; then
    echo "required command is not installed: $command" >&2
    exit 2
  fi
done

state_dir=${HAL_NETPLAY_STATE_DIR:-"$repo_dir/runs/netplay"}
mkdir -p "$state_dir/replays"
cd "$repo_dir"
uv sync --extra netplay-server --locked

process_ids=()
stop() {
  trap - EXIT INT TERM
  if (( ${#process_ids[@]} )); then
    kill -TERM "${process_ids[@]}" 2>/dev/null || true
    wait "${process_ids[@]}" 2>/dev/null || true
  fi
}
trap stop EXIT INT TERM

HAL_NETPLAY_DATABASE="$state_dir/queue.sqlite3" \
HAL_NETPLAY_CAPACITY=1 \
HAL_NETPLAY_RUNNER_STATUS="$state_dir/runner-status.json" \
uv run hal-netplay-api --host 127.0.0.1 --port 8080 &
process_ids+=("$!")

xvfb-run -a uv run hal-netplay-runner "$HAL_NETPLAY_POLICY" \
  --compiled \
  --database "$state_dir/queue.sqlite3" \
  --user-jsons "$HAL_NETPLAY_USER_JSON_A" \
  --slippi-ports 51441 \
  --iso-path "$HAL_ISO_PATH" \
  --dolphin-path "$HAL_NETPLAY_EMULATOR_PATH" \
  --replay-dir "$state_dir/replays" \
  --status-path "$state_dir/runner-status.json" \
  --git-sha "$HAL_GIT_SHA" &
process_ids+=("$!")

cloudflared tunnel --no-autoupdate run \
  --token "$CLOUDFLARE_TUNNEL_TOKEN" &
process_ids+=("$!")

wait -n "${process_ids[@]}"
