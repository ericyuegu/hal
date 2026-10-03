#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} != --maintenance || $# != 2 || ( $2 != on && $2 != off ) ]]; then
  echo 'usage: deploy/netplay/deploy-api.sh --maintenance on|off' >&2
  exit 2
fi
maintenance=$2
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir/web/netplay-api"
npm ci
npm test
npm run typecheck
npx wrangler deploy --var "MAINTENANCE:$maintenance" --var EDGE_CACHE:on
