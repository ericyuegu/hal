#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/../.." && pwd)

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  echo "usage: deploy/netplay/deploy-frontend.sh [environment-file]"
  exit 0
fi
if (( $# > 1 )); then
  echo "deploy-frontend.sh accepts at most one environment file" >&2
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
if [[ -z ${HAL_NETPLAY_PUBLIC_API_URL:-} ]]; then
  echo "environment variable is not set: HAL_NETPLAY_PUBLIC_API_URL" >&2
  exit 2
fi
if ! command -v npm >/dev/null; then
  echo "required command is not installed: npm" >&2
  exit 2
fi

cd "$repo_dir/web/netplay"
npm ci
NEXT_PUBLIC_HAL_API_URL="$HAL_NETPLAY_PUBLIC_API_URL" npm run build
npx wrangler deploy --config dist/server/wrangler.json
