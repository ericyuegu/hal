#!/usr/bin/env bash
# The netplay runner owns one X server per slot and their cleanup.
set -euo pipefail

exec "$@"
