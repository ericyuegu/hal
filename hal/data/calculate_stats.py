import argparse
import json

import numpy as np
import numpy.ma as ma
from constants import NP_MASK_VALUE
from loguru import logger
from streaming import StreamingDataset


def calculate_statistics_for_mds(input_path: str, output_path: str, max_examples: int | None) -> None:
    """Calculate and save statistics for each feature to a JSON."""
    dataset = StreamingDataset(local=input_path, remote=None, batch_size=1, shuffle=False)
    statistics = {}

    for i, example in enumerate(dataset):
        if max_examples is not None and i >= max_examples:
            break

        for field_name, field_data in example.items():
            if field_name not in statistics:
                statistics[field_name] = {
                    "count": 0,
                    "mean": 0,
                    "M2": 0,
                    "min": float("inf"),
                    "max": float("-inf"),
                    "skipped": 0,
                }

            numpy_array = ma.masked_greater_equal(field_data, NP_MASK_VALUE)

            if numpy_array.dtype == object or not np.issubdtype(numpy_array.dtype, np.number):
                statistics[field_name]["skipped"] += numpy_array.size
                continue

            valid_data = numpy_array.compressed()  # Get only non-masked values
            if valid_data.size == 0:
                statistics[field_name]["skipped"] += numpy_array.size
                continue

            feature_stats = statistics[field_name]
            feature_stats["count"] += valid_data.size
            delta = valid_data - feature_stats["mean"]
            feature_stats["mean"] += np.sum(delta) / feature_stats["count"]
            delta2 = valid_data - feature_stats["mean"]
            feature_stats["M2"] += np.sum(delta * delta2)
            feature_stats["min"] = min(feature_stats["min"], np.min(valid_data))
            feature_stats["max"] = max(feature_stats["max"], np.max(valid_data))

        if i % 1000 == 0:
            logger.info(f"Processed {i} examples")

    for field_name, feature_stats in statistics.items():
        if feature_stats["count"] > 0:
            feature_stats["std"] = np.sqrt(feature_stats["M2"] / feature_stats["count"])
        else:
            logger.warning(f"No valid numeric data for {field_name}")

        del feature_stats["M2"]

        total = feature_stats["count"] + feature_stats["skipped"]
        feature_stats["skipped_percentage"] = (feature_stats["skipped"] / total) * 100 if total > 0 else 0

        for key, value in feature_stats.items():
            if isinstance(value, np.number):
                feature_stats[key] = value.item()

    try:
        logger.info(f"Saving statistics to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f)
    except OSError as e:
        logger.error(f"Error saving statistics: {e}")

    logger.info("Statistics calculation completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to the input dataset")
    parser.add_argument("--output_path", type=str, help="Path to the output JSON file")
    parser.add_argument("--max_examples", type=int, default=None, help="Maximum number of examples to process")
    args = parser.parse_args()

    calculate_statistics_for_mds(args.input_path, args.output_path, args.max_examples)
