# HAL

Training superhuman AI for *Super Smash Bros. Melee*. 

This project is under active development and is not ready for public use. 

Blog post: https://ericyuegu.com/melee-pt1

# Setup

This project targets Python ≥ 3.14 on Ubuntu 20.04+. Dependencies are managed by [uv](https://docs.astral.sh/uv/).

`peppi-py` (the slp parser used by the data pipeline) is pulled from a fork and built from source via `maturin`, so a Rust toolchain is required:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable --profile minimal
. "$HOME/.cargo/env"
uv sync
```

The first `uv sync` will compile `peppi-py` (~35s); subsequent syncs reuse the cached build.

For macOS, `libmelee` requires a system installation of enet:
```bash
brew install enet
CFLAGS="-I$(brew --prefix enet)/include" \
LDFLAGS="-L$(brew --prefix enet)/lib -lenet" \
uv sync
```

## Dolphin emulator

Download the latest Slippi ExiAI AppImage (e.g. `Slippi_Online-x86_64-ExiAI.AppImage`) into `~/data/ssbm/` and extract it once:

```bash
chmod +x ~/data/ssbm/Slippi_Online-x86_64-ExiAI.AppImage
( cd ~/data/ssbm && ./Slippi_Online-x86_64-ExiAI.AppImage --appimage-extract )
```

`libmelee` should be pointed at `~/data/ssbm/squashfs-root/AppRun`. The ExiAI build forces a Null video backend, so it runs headless with no X display required. To build the emulator from source instead, follow the instructions [here](https://github.com/ericyuegu/slippi-Ishiiruka/tree/ubuntu-20.04).

## Downloading data

You can obtain raw `.slp` files from the [Slippi Discord](https://discord.gg/qaHgPwpr) server.

# HOW-TO

Paths to the repo, Dolphin, ISO, and replay directory are resolved by `hal/local_paths.py` from environment variables, with defaults that match the layout above (`~/data/ssbm/...`). To override, copy `.env.example` to `.env` and edit, or `export` the variables in your shell profile.

## Processing replays to MDS format

```bash
uv run python hal/data/process_replays.py --replay_dir /path/to/replays --output_dir /path/to/mds
```

## Training

```bash
uv run python hal/training/simple_trainer.py --n_gpus 1 --data.data_dir /path/to/mds --arch GPTv5Controller-512-6-8-dropout
```

## Evaluation

```bash
uv run python hal/eval/eval.py --model_dir /path/to/model_dir --n_workers 1
```
