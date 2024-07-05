import json
from typing import Dict

import attr


@attr.s(auto_attribs=True, frozen=True)
class FeatureStats:
    """Contains mean, std, median, min, and max for each feature."""

    mean: float
    std: float
    min: float
    max: float
    median: float


def load_dataset_stats(path: str) -> Dict[str, FeatureStats]:
    """Load the dataset statistics from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    dataset_stats = {k: FeatureStats(**v) for k, v in data.items()}
    return dataset_stats
