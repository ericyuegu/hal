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
for command in flock npm; do
  if ! command -v "$command" >/dev/null; then
    echo "required command is not installed: $command" >&2
    exit 2
  fi
done

lock_file=${HAL_NETPLAY_LOCAL_LOCK:-"$repo_dir/runs/netplay/run-local.lock"}
mkdir -p "$(dirname -- "$lock_file")"
exec 9>"$lock_file"
if ! flock -n 9; then
  echo "local netplay is already running" >&2
  exit 2
fi

process_groups=()
stop() {
  local alive
  local process_group

  trap - EXIT INT TERM
  for process_group in "${process_groups[@]}"; do
    kill -TERM -- "-$process_group" 2>/dev/null || true
  done
  # run-host.sh needs time to stop its own child process groups first.
  for _ in {1..50}; do
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
"$script_dir/run-host.sh" "$env_file" </dev/null &
process_groups+=("$!")

cd "$repo_dir/web/netplay"
if [[ ! -x node_modules/.bin/vinext ]]; then
  npm ci
fi
NEXT_PUBLIC_HAL_API_URL=http://127.0.0.1:8080 \
npm run dev -- --host 127.0.0.1 --port 3000 </dev/null &
process_groups+=("$!")
set +m

wait -n "${process_groups[@]}"
