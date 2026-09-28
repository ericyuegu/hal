#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'EOF'
usage: deploy/netplay/gce-up.sh NAME --project PROJECT --image IMAGE --secret SECRET --service-account ACCOUNT [options]

Options:
  --zone ZONE                 Compute zone (default: us-central1-a)
  --machine-type TYPE         Machine type (default: g4-standard-48)
  --git-sha SHA               Full image Git SHA; default is the image tag
  --slots N                   Runner slots, 1-8 (default: 1)
  --drain-timeout SECONDS     Graceful drain deadline (default: 900)
  --image-family FAMILY       GPU-ready boot image family
  --image-project PROJECT     Boot image project
  --managed                   Create a size-one managed instance group
  --virtual-workstation       Attach the RTX Virtual Workstation accelerator variant
EOF
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
if (( $# == 0 )); then usage >&2; exit 2; fi
name=$1
shift
project=
image=
secret=
service_account=
zone=us-central1-a
machine_type=g4-standard-48
git_sha=
slots=1
drain_timeout=900
image_family=common-cu129-ubuntu-2404-nvidia-580
image_project=deeplearning-platform-release
managed=0
virtual_workstation=0
while (( $# )); do
  case $1 in
    --project|--image|--secret|--service-account|--zone|--machine-type|--git-sha|--slots|--drain-timeout|--image-family|--image-project)
      (( $# >= 2 )) || { echo "missing value for $1" >&2; exit 2; }
      option=${1#--}
      option=${option//-/_}
      printf -v "$option" '%s' "$2"
      shift 2
      ;;
    --managed) managed=1; shift ;;
    --virtual-workstation) virtual_workstation=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ $name =~ ^[a-z]([-a-z0-9]{0,61}[a-z0-9])?$ ]] || { echo "invalid resource name: $name" >&2; exit 2; }
for required in project image secret service_account; do
  [[ -n ${!required} ]] || { echo "missing --${required//_/-}" >&2; exit 2; }
done
if [[ -z $git_sha ]]; then git_sha=${image##*:}; fi
[[ $git_sha =~ ^[0-9a-f]{40}$ ]] || { echo "image must have a full Git SHA tag, or pass --git-sha" >&2; exit 2; }
[[ $slots =~ ^[1-8]$ ]] || { echo "--slots must be between 1 and 8" >&2; exit 2; }
[[ $drain_timeout =~ ^[1-9][0-9]*$ ]] || { echo "--drain-timeout must be a positive integer" >&2; exit 2; }
command -v gcloud >/dev/null || { echo "gcloud is not installed" >&2; exit 2; }

metadata="hal-netplay-project=$project,hal-netplay-image=$image,hal-netplay-git-sha=$git_sha,hal-netplay-secret=$secret,hal-netplay-slots=$slots,hal-netplay-drain-timeout=$drain_timeout"
common=(
  "--project=$project" "--machine-type=$machine_type"
  "--image-family=$image_family" "--image-project=$image_project"
  "--boot-disk-size=100GB" "--boot-disk-type=hyperdisk-balanced"
  "--maintenance-policy=TERMINATE" "--restart-on-failure"
  "--service-account=$service_account" "--scopes=cloud-platform"
  "--metadata=$metadata" "--metadata-from-file=startup-script=$script_dir/gce-startup.sh"
  "--tags=hal-netplay-health" "--labels=app=hal-netplay,workload=runner"
)
if (( virtual_workstation )); then
  common+=("--accelerator=type=nvidia-rtx-pro-6000-vws,count=1")
fi

if (( ! managed )); then
  gcloud compute instances create "$name" --zone="$zone" "${common[@]}" --quiet
  echo "created $name; logs: gcloud compute instances get-serial-port-output $name --project=$project --zone=$zone"
  exit 0
fi

template="${name}-template"
health="${name}-health"
firewall="${name}-health"
gcloud compute health-checks create http "$health" --project="$project" --port=9101 \
  --request-path=/healthz --check-interval=10s --timeout=5s --unhealthy-threshold=3 --healthy-threshold=2 --quiet
gcloud compute firewall-rules create "$firewall" --project="$project" --network=default \
  --allow=tcp:9101 --source-ranges=35.191.0.0/16,130.211.0.0/22 --target-tags=hal-netplay-health --quiet
gcloud compute instance-templates create "$template" "${common[@]}" --quiet
gcloud compute instance-groups managed create "$name" --project="$project" --zone="$zone" \
  --template="$template" --size=1 --health-check="$health" --initial-delay=2400 --quiet
echo "created managed group $name; inspect: gcloud compute instance-groups managed list-instances $name --project=$project --zone=$zone"
