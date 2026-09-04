# HAL

Training superhuman AI for *Super Smash Bros. Melee* via imitation learning and RL.

HAL is named after both [HAL Laboratory](https://en.wikipedia.org/wiki/HAL_Laboratory), the developer of Super Smash Bros. Melee, and [HAL 9000](https://en.wikipedia.org/wiki/HAL_9000), the infamous robot villain from 2001: A Space Odyssey.

Blog: https://ericyuegu.com/melee-pt1.

## Quick start

Setup venv:
```bash
uv sync
source .venv/bin/activate
```

To download ready-made datasets and emulator for training and eval, request keys for the S3 bucket from the maintainer [@ericyuegu](https://github.com/ericyuegu).

You can copy keys to `.env` or your `.bashrc`.
```
source .env
uv run fetch    # will download to `<repo_root>/data/` by default
```

Training experiments reside as single files under `experiments/`.
```
uv run experiments/001_flow_matching_baseline.py
```

To launch experiments on cloud, wrap your local training command with a launcher script:
```
uv run scripts/launch_vast.py --max-price 1.0 -- uv run experiments/001_flow_matching_baseline.py
```

Modal is the default fixed-hardware option. One-time setup requires an authenticated
Modal profile and a `hal` Secret with the R2 and W&B credentials:
```bash
uv run modal setup
source .env
uv run modal secret create hal \
  AWS_ENDPOINT_URL="$AWS_ENDPOINT_URL" \
  AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  AWS_BUCKET="$AWS_BUCKET" \
  WANDB_API_KEY="$WANDB_API_KEY"
```
The launcher defaults to one B200, 32 requested/48 maximum CPU cores, 128 GiB
RAM, a 384 GiB memory limit, and a 2 TiB ephemeral SSD. It submits a detached
Function by default:
```bash
uv run scripts/launch_modal.py --dry-run -- uv run experiments/028_onehot_controller.py
uv run scripts/launch_modal.py -- uv run experiments/028_onehot_controller.py
uv run scripts/launch_modal.py --wait -- uv run experiments/028_onehot_controller.py
```
Modal can preempt GPU Functions, and each Function attempt is limited to 24 hours.
The launcher gives an input ten retries and records its run name in the automatically
created `hal-modal-state` Volume. A replacement attempt adds `--resume <run>` only
after `runs/<run>/latest.pt` is present in R2. A normal nonzero training exit is
recorded as terminal and is not run again. PyTorch's training and inference compiler
caches share the persistent `hal-modal-compile-cache-v1` Volume; temporary compiler
files remain on the Function's ephemeral SSD. Arbitrary commands must opt out with
`--no-auto-resume`. Use `uv run scripts/launch_modal.py --help` for resource,
region, timeout, image, state, compile-cache, and retry options.

Google Compute Engine is also supported. The launcher uses your interactive
`gcloud auth login` session (no service-account key file) and reads job secrets
from Secret Manager through the VM's attached service account:
```
uv run scripts/launch_gce.py --dry-run --zone us-central1-a -- uv run experiments/001_flow_matching_baseline.py
uv run scripts/launch_gce.py --zone us-central1-a --service-account hal-jobs@PROJECT.iam.gserviceaccount.com -- uv run experiments/001_flow_matching_baseline.py
```
The service account needs `roles/secretmanager.secretAccessor` on the secrets
listed by `--secret`. See `uv run scripts/launch_gce.py --help` for GPU, Spot,
network, disk, and lifecycle options.


## Data

### Raw datasets

From the Slippi Discord server:

- ranked-anonymized-1-116248: https://drive.google.com/file/d/1pFjgh1dapX34s0T-Q1TC7JUO-qjYbQZf/view
- ranked-anonymized-2-151807: https://drive.google.com/file/d/1jEIzvhpV3778J2s2-Np9vCVqSLf9lZnk/view
- ranked-anonymized-3-128787: https://drive.google.com/file/d/1glzlkAPxHC58oXZljJXQV8dsTBKmlhkE/view
- ranked-anonymized-4-148358: https://drive.google.com/file/d/1qdIZUW4Er_Vu6rD3-VUvyak3lKa1KxVk/view
- ranked-anonymized-5-133261: https://drive.google.com/file/d/1Hqmj6C8g1BzuRAIqOrQcMDL0MX4GtffE/view
- ranked-anonymized-6-171694: https://drive.google.com/file/d/1g8yZ-Q4ldyhDEmXLSPBoWxywJRMRVGc3/view

### Data preprocessing

To create your own training datasets from `.slp` files, there are 3 helpful scripts in `hal/scripts/`:

```bash
# step 1: indexing - supports directly reading from .7z archives on-the-fly
uv run hal/scripts/build_index.py --archive data/raw/dev.7z --output data/processed/dev/index.jsonl

# step 2: filtering
uv run hal/scripts/filter.py --index data/processed/dev/index.jsonl --output data/processed/dev/paths.txt

# step 3: materializing
uv run hal/scripts/materialize.py --paths-file data/processed/dev/paths.txt --index data/processed/dev/index.jsonl --output data/processed/dev/mds
```
