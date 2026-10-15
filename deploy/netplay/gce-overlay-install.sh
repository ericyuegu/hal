#!/usr/bin/env bash
set -euo pipefail

# Only the overlay service may be restarted here. Gameplay owns the shared display/audio.
sha=${1:?usage: gce-overlay-install.sh SHA ARCHIVE}
archive=${2:?usage: gce-overlay-install.sh SHA ARCHIVE}
[[ $sha =~ ^[0-9a-f]{40}$ ]] || { echo "invalid release revision" >&2; exit 2; }
root=/var/lib/hal-overlay
[[ -L $root/current ]] || { echo "the independent overlay service is not installed" >&2; exit 1; }
previous=$(readlink "$root/current")
release="$root/releases/$sha"
staging=$(mktemp -d "$root/releases/staging-XXXXXXXX")
trap 'rm -rf "$staging"' EXIT
tar -xf "$archive" -C "$staging"
ln -s deploy/netplay/fonts "$staging/fonts"
printf '%s\n' "$sha" > "$staging/git-sha"
image=$(docker inspect --format '{{.Config.Image}}' hal-netplay-overlay)
docker run --rm --entrypoint python --workdir /tmp -e PYTHONPATH=/release -v "$staging:/release:ro" \
  "$image" -c 'from hal.netplay_service import overlay; from hal.netplay_service import obs; print("overlay imports passed")'
if [[ -d $release ]]; then
  diff -qr "$staging" "$release" >/dev/null || { echo "release revision already has different content" >&2; exit 1; }
else
  mv "$staging" "$release"
fi
ln -s "releases/$sha" "$root/next"
mv -Tf "$root/next" "$root/current"
rollback() {
  echo "overlay activation failed; restoring the previous release" >&2
  ln -s "$previous" "$root/next"
  mv -Tf "$root/next" "$root/current"
  systemctl restart hal-netplay-overlay.service
}
trap 'rollback; rm -rf "$staging"' ERR
started=$(python3 -c 'import time; print(time.time())')
systemctl restart hal-netplay-overlay.service
for _ in {1..60}; do
  if python3 - "$sha" "$started" <<'PY'
import json
import sys
import time
from pathlib import Path

try:
    state = json.loads(Path('/var/lib/hal-netplay/runner-status.overlay-service.json').read_text())
    started = float(sys.argv[2])
    valid = state['git_sha'] == sys.argv[1] and started <= state['updated_at'] <= time.time() < state['updated_at'] + 3
    lease = json.loads(Path('/var/lib/hal-netplay/runner-status.stream-lease.json').read_text())
    if lease['stream'] is not None and 0 <= time.time() - lease['updated_at'] < 30:
        output = json.loads(Path('/var/lib/hal-netplay/runner-status.overlay.obs.json').read_text())
        valid = valid and output['updated_at'] >= started and output['stream']['outputActive']
except (OSError, ValueError, KeyError):
    valid = False
sys.exit(0 if valid else 1)
PY
  then
    trap - ERR
    rm -f "$archive"
    echo "overlay release $sha is running; gameplay was not restarted"
    exit 0
  fi
  sleep 0.5
done
false
