#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_dir/web/netplay"
npm ci
npm run build
npx wrangler deploy --config dist/server/wrangler.json
