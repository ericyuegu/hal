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

policy_override=${HAL_NETPLAY_POLICY_OVERRIDE:-}
set -a
source "$env_file"
set +a
if [[ -n $policy_override ]]; then
  export HAL_NETPLAY_POLICY=$policy_override
fi

required_variables=(
  HAL_GIT_SHA
  HAL_NETPLAY_POLICY
  HAL_NETPLAY_USER_JSON_A
  HAL_ISO_PATH
  HAL_NETPLAY_EMULATOR_PATH
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

export HAL_NETPLAY_ALLOWED_ORIGINS=${HAL_NETPLAY_ALLOWED_ORIGINS:-http://127.0.0.1:3000,http://localhost:3000}
export HAL_NETPLAY_ALLOWED_HOSTS=${HAL_NETPLAY_ALLOWED_HOSTS:-127.0.0.1,localhost}

user_jsons=("$HAL_NETPLAY_USER_JSON_A")
slippi_ports=("${HAL_NETPLAY_SLIPPI_PORT_A:-51441}")
if [[ -n ${HAL_NETPLAY_USER_JSON_B:-} ]]; then
  user_jsons+=("$HAL_NETPLAY_USER_JSON_B")
  slippi_ports+=("${HAL_NETPLAY_SLIPPI_PORT_B:-51442}")
fi
if (( ${#user_jsons[@]} == 2 )) && [[ ${user_jsons[0]} == "${user_jsons[1]}" || ${slippi_ports[0]} == "${slippi_ports[1]}" ]]; then
  echo "netplay workers need distinct Slippi credentials and ports" >&2
  exit 2
fi
capacity=${#user_jsons[@]}
printf -v user_jsons_csv '%s,' "${user_jsons[@]}"
printf -v slippi_ports_csv '%s,' "${slippi_ports[@]}"
user_jsons_csv=${user_jsons_csv%,}
slippi_ports_csv=${slippi_ports_csv%,}

required_commands=(uv xvfb-run)
if [[ -n ${CLOUDFLARE_TUNNEL_TOKEN:-} && ${HAL_NETPLAY_NO_TUNNEL:-0} != 1 ]]; then
  required_commands+=(cloudflared)
fi
for command in "${required_commands[@]}"; do
  if ! command -v "$command" >/dev/null; then
    echo "required command is not installed: $command" >&2
    exit 2
  fi
done

state_dir=${HAL_NETPLAY_STATE_DIR:-"$repo_dir/runs/netplay"}
export TMPDIR=${HAL_NETPLAY_TMPDIR:-"$state_dir/tmp"}
mkdir -p "$state_dir/replays" "$TMPDIR"
cd "$repo_dir"
uv sync --extra netplay-server --locked

process_groups=()
stop() {
  local alive
  local process_group

  trap - EXIT INT TERM
  for process_group in "${process_groups[@]}"; do
    kill -TERM -- "-$process_group" 2>/dev/null || true
  done
  for _ in {1..20}; do
    alive=false
    for process_group in "${process_groups[@]}"; do
      if kill -0 -- "-$process_group" 2>/dev/null; then
        alive=true
        break
      fi
    done
    if [[ $alive == false ]]; then
      break
    fi
    sleep 0.1
  done
  for process_group in "${process_groups[@]}"; do
    kill -KILL -- "-$process_group" 2>/dev/null || true
  done
  wait "${process_groups[@]}" 2>/dev/null || true
}
trap stop EXIT INT TERM

set -m
HAL_NETPLAY_DATABASE="$state_dir/queue.sqlite3" \
HAL_NETPLAY_CAPACITY="$capacity" \
HAL_NETPLAY_RUNNER_STATUS="$state_dir/runner-status.json" \
uv run hal-netplay-api --host 127.0.0.1 --port 8080 </dev/null &
process_groups+=("$!")

xvfb-run -a uv run hal-netplay-runner "$HAL_NETPLAY_POLICY" \
  --compiled \
  --database "$state_dir/queue.sqlite3" \
  --user-jsons "$user_jsons_csv" \
  --slippi-ports "$slippi_ports_csv" \
  --iso-path "$HAL_ISO_PATH" \
  --dolphin-path "$HAL_NETPLAY_EMULATOR_PATH" \
  --replay-dir "$state_dir/replays" \
  --status-path "$state_dir/runner-status.json" \
  --git-sha "$HAL_GIT_SHA" </dev/null &
process_groups+=("$!")

if [[ -n ${CLOUDFLARE_TUNNEL_TOKEN:-} && ${HAL_NETPLAY_NO_TUNNEL:-0} != 1 ]]; then
  cloudflared tunnel --no-autoupdate run \
    --token "$CLOUDFLARE_TUNNEL_TOKEN" </dev/null &
  process_groups+=("$!")
fi
set +m

wait -n "${process_groups[@]}"
