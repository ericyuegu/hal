#!/usr/bin/env bash
# Run the bounded debug pilot on the lent GPU VM after operator cleanup.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python -m scripts.launch_o65_existing "$@"
