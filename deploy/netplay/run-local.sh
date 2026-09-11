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
if ! command -v npm >/dev/null; then
  echo "required command is not installed: npm" >&2
  exit 2
fi

env_file=${1:-"$script_dir/.env"}
if [[ ! -f $env_file ]]; then
  echo "environment file does not exist: $env_file" >&2
  exit 2
fi

process_ids=()
stop() {
  trap - EXIT INT TERM
  if (( ${#process_ids[@]} )); then
    kill -TERM "${process_ids[@]}" 2>/dev/null || true
    wait "${process_ids[@]}" 2>/dev/null || true
  fi
}
trap stop EXIT INT TERM

"$script_dir/run-host.sh" "$env_file" &
process_ids+=("$!")

cd "$repo_dir/web/netplay"
if [[ ! -x node_modules/.bin/vinext ]]; then
  npm ci
fi
NEXT_PUBLIC_HAL_API_URL=http://127.0.0.1:8080 \
npm run dev -- --host 127.0.0.1 --port 3000 &
process_ids+=("$!")

wait -n "${process_ids[@]}"
