# HAL

Training superhuman AI for *Super Smash Bros. Melee*. 

This project is under active development and is not ready for public use. 

Blog post: https://ericyuegu.com/melee-pt1

# Setup

This project has been tested for Python 3.11 on Ubuntu 20.04 LTS. 

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For macOS, `libmelee` requires a system installation of enet. 
```bash
brew install enet
CFLAGS="-I/opt/homebrew/Cellar/enet/1.3.18/include" LDFLAGS="-L/opt/homebrew/Cellar/enet/1.3.18/lib -lenet" pip install melee --no-cache-dir
pip install -r requirements_macos.txt
```

## Building Dolphin emulator

An AppImage is provided in the `emulator` directory and can be called directly from `libmelee`. 

To build the emulator from source, follow the instructions [here](https://github.com/ericyuegu/slippi-Ishiiruka/tree/ubuntu-20.04).

## Downloading data

You can obtain raw `.slp` files from the [Slippi Discord](https://discord.gg/qaHgPwpr) server.

# HOW-TO

I recommend modifying the constants in `hal/local_paths.py` to point to your local directories for the repo, Dolphin, and the Melee ISO.

## Processing replays to MDS format

```bash
python hal/data/process_replays.py --replay_dir /path/to/replays --output_dir /path/to/mds
```

## Training

```bash
python hal/training/simple_trainer.py --n_gpus 1 --data.data_dir /path/to/mds --arch GPTv5Controller-512-6-8-dropout
```

## Evaluation

```bash
python hal/eval/eval.py --model_dir /path/to/model_dir --n_workers 1
```
