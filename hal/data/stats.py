import json
from pathlib import Path

import attr

from hal.local_paths import REPO_DIR


@attr.s(auto_attribs=True, frozen=True)
class FeatureStats:
    """Contains mean, std, median, min, and max for each feature."""

    mean: float
    std: float
    min: float
    max: float
    # median: float


def load_dataset_stats(path: str | Path) -> dict[str, FeatureStats]:
    """Load the dataset statistics from a JSON file."""
    if not Path(path).is_absolute():
        # Support relative paths in config when reloading from arbitrary dir / jupyter notebook
        path = Path(REPO_DIR) / path
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dataset_stats = {}
    for k, v in data.items():
        # Explode in case of missing fields
        feature_stats = FeatureStats(mean=v["mean"], std=v["std"], min=v["min"], max=v["max"])
        dataset_stats[k] = feature_stats
    return dataset_stats
