#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: deploy/netplay/gce-down.sh NAME --project PROJECT [options]

Options:
  --zone ZONE                 Compute zone (default: us-central1-a)
  --drain-timeout SECONDS     Runner drain deadline (default: 900)
  --managed                   Remove a size-one managed instance group
  --force                     Delete even when the runner cannot drain
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
if (( $# == 0 )); then usage >&2; exit 2; fi
name=$1
shift
project=
zone=us-central1-a
drain_timeout=900
managed=0
force=0
while (( $# )); do
  case $1 in
    --project|--zone|--drain-timeout)
      (( $# >= 2 )) || { echo "missing value for $1" >&2; exit 2; }
      option=${1#--}
      option=${option//-/_}
      printf -v "$option" '%s' "$2"
      shift 2
      ;;
    --managed) managed=1; shift ;;
    --force) force=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n $project ]] || { echo "missing --project" >&2; exit 2; }
[[ $drain_timeout =~ ^[1-9][0-9]*$ ]] || { echo "--drain-timeout must be a positive integer" >&2; exit 2; }
command -v gcloud >/dev/null || { echo "gcloud is not installed" >&2; exit 2; }

drain() {
  local instance=$1
  if timeout "$((drain_timeout + 120))" gcloud compute ssh "$instance" --project="$project" --zone="$zone" \
      --command="sudo systemctl stop hal-netplay-runner.service" --quiet; then
    return 0
  fi
  if (( force )); then
    echo "warning: runner drain failed on $instance; continuing because --force was set" >&2
    return 0
  fi
  echo "runner drain failed on $instance; refusing deletion (use --force to override)" >&2
  return 1
}

if (( ! managed )); then
  drain "$name"
  gcloud compute instances delete "$name" --project="$project" --zone="$zone" --quiet
  exit 0
fi

instances=$(gcloud compute instance-groups managed list-instances "$name" --project="$project" --zone="$zone" --format='value(instance)')
while IFS= read -r resource; do
  [[ -n $resource ]] || continue
  drain "${resource##*/}"
done <<< "$instances"
gcloud compute instance-groups managed resize "$name" --project="$project" --zone="$zone" --size=0 --quiet
gcloud compute instance-groups managed delete "$name" --project="$project" --zone="$zone" --quiet
gcloud compute instance-templates delete "${name}-template" --project="$project" --quiet || true
gcloud compute health-checks delete "${name}-health" --project="$project" --quiet || true
gcloud compute firewall-rules delete "${name}-health" --project="$project" --quiet || true
