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
