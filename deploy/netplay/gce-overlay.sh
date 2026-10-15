#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )) || [[ $2 != --project || $4 != --zone ]]; then
  echo "usage: deploy/netplay/gce-overlay.sh NAME --project PROJECT --zone ZONE" >&2
  exit 2
fi
name=$1
project=$3
zone=$5
[[ $name =~ ^[a-z0-9-]+$ && $project =~ ^[a-z0-9-]+$ && $zone =~ ^[a-z0-9-]+$ ]] || exit 2
repo=$(git rev-parse --show-toplevel)
git -C "$repo" diff --quiet HEAD -- hal deploy/netplay/fonts deploy/netplay/overlay-fonts.conf || {
  echo "commit the overlay source and fonts before deploying" >&2; exit 1;
}
sha=$(git -C "$repo" rev-parse HEAD)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
remote="/tmp/hal-overlay-$sha-$$"
git -C "$repo" archive --format=tar HEAD hal deploy/netplay/fonts deploy/netplay/overlay-fonts.conf > "$work/bundle.tar"
gcloud compute scp "$work/bundle.tar" "$name:$remote.tar" --project="$project" --zone="$zone" --quiet
gcloud compute scp "$repo/deploy/netplay/gce-overlay-install.sh" "$name:$remote.sh" --project="$project" --zone="$zone" --quiet
gcloud compute ssh "$name" --project="$project" --zone="$zone" \
  --command="sudo bash $remote.sh $sha $remote.tar; result=\$?; rm -f $remote.sh; exit \$result"
