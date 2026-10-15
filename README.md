# HAL

HAL trains *Super Smash Bros. Melee* policies on Slippi replays and plays them in Dolphin.

- [Blog post](https://ericyuegu.com/hillclimbing-melee)
- [Model weights](https://huggingface.co/ericyuegu/hal-base-250m-rc)
- [Dataset](https://huggingface.co/datasets/ericyuegu/hal-mds-dataset)

## Setup

You need Linux, an NVIDIA GPU, [uv](https://docs.astral.sh/uv/), and a Melee 1.02 NTSC ISO.

```bash
git clone https://github.com/ericyuegu/hal && cd hal
uv sync
uv run fetch --name dolphin-exiai                  # headless Dolphin for CPU matches
uv run fetch --name Slippi_Online-x86_64.AppImage  # Slippi Dolphin with a window
export HAL_ISO_PATH=/path/to/ssbm.iso
curl -LO https://huggingface.co/ericyuegu/hal-base-250m-rc/resolve/main/hal-250m.hal
```

## Play

Plug your controller into port 1 of a GameCube controller adapter. On Linux, Dolphin needs the adapter's [udev rule](https://dolphin-emu.org/docs/guides/how-use-official-gc-controller-adapter-wii-u/).

```bash
uv run hal-policy play hal-250m.hal --character FOX --stage FINAL_DESTINATION
```

Dolphin opens with HAL on port 2. Pick your character, and HAL starts the match. Dolphin waits for each of HAL's actions, so HAL plays with no input delay. Replays go to `replays/local-play/`.

To watch HAL play a level-9 CPU without a window:

```bash
uv run hal-policy eval hal-250m.hal --n-matches 1
```

## Private R2 bucket

The maintainer keeps the training corpus, test fixtures, and checkpoints in a private Cloudflare R2 bucket named `hal`. Setup and play work without it. Everything below reads or writes the bucket and needs its keys in `.env`, copied from `.env.example`:

- `uv run fetch` without `--name` downloads the pinned ISO and the test datasets.
- Integration tests use those fixtures.
- Training streams the corpus from `s3://hal/processed/`, downloads the player-identity file, and uploads checkpoints to `s3://hal/runs/`. The Hugging Face dataset holds the same corpus files, but the training code reads only from R2.
- Closed-loop evaluation during training uses the pinned ISO.
- `--resume` downloads checkpoints, and `hal-policy export` downloads per-source feature statistics.
- The cloud launchers pass the keys to their jobs.
- The netplay service downloads its ISO, policy, and Slippi accounts.
- `hal/scripts/publish_mds.py` uploads datasets.

Training also logs to W&B, so `.env` needs `WANDB_API_KEY`.

## Train

```bash
source .env
uv run experiments/059_muon_action_sequence.py train --proxy   # 17M parameters
uv run experiments/059_muon_action_sequence.py train           # 247M parameters
uv run hal-policy export runs/<run>/final.pt policy.hal
```

To train in the cloud, put a launcher in front of the same command:

```bash
uv run scripts/launch_modal.py -- uv run experiments/059_muon_action_sequence.py train
```

Modal needs `uv run modal setup` and a `hal` secret with the variables from `.env`. `launch_vast.py` and `launch_gce.py` take the same form. Run any launcher with `--help` for its options.

## Build a dataset

```bash
uv run hal/scripts/build_index.py --archive replays.7z --output data/processed/mine/index.jsonl
uv run hal/scripts/filter.py --index data/processed/mine/index.jsonl --output data/processed/mine/paths.txt
uv run hal/scripts/materialize.py --replay-format policy-world --paths-file data/processed/mine/paths.txt \
  --index data/processed/mine/index.jsonl --output data/processed/mine/mds
```

## Serve

```bash
cp deploy/netplay/.env.example deploy/netplay/.env
deploy/netplay/run-local.sh
```

[deploy/netplay/README.md](deploy/netplay/README.md) covers hosting.

## Layout

| Path | Contents |
| --- | --- |
| `hal/` | Library code for data, models, inference, emulation, evaluation, and the netplay service |
| `experiments/` | Training scripts, with 059 as the maintained recipe |
| `scripts/` | Cloud launchers and benchmarks |
| `docs/` | Inference and evaluation details, starting at [inference.md](docs/inference.md) |
| `archive/` | Earlier experiments and notes |

Before you send a change, run the checks in [AGENTS.md](AGENTS.md).
