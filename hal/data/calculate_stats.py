import argparse
import json

import numpy as np
import numpy.ma as ma
from loguru import logger
from streaming import StreamingDataset

from hal.constants import NP_MASK_VALUE


def calculate_statistics_for_mds(input_path: str, output_path: str) -> None:
    """Calculate and save statistics for each feature to a JSON."""
    dataset = StreamingDataset(local=input_path, remote=None, batch_size=1, shuffle=False)
    statistics = {}

    for i, example in enumerate(dataset):
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

            numpy_array = ma.masked_equal(field_data, NP_MASK_VALUE)

            if numpy_array.dtype == object or not np.issubdtype(numpy_array.dtype, np.number):
                statistics[field_name]["skipped"] += numpy_array.size
                continue

            stats = statistics[field_name]
            stats["count"] += numpy_array.size
            delta = numpy_array - stats["mean"]
            stats["mean"] += np.sum(delta) / stats["count"]
            delta2 = numpy_array - stats["mean"]
            stats["M2"] += np.sum(delta * delta2)
            stats["min"] = min(stats["min"], np.min(numpy_array))
            stats["max"] = max(stats["max"], np.max(numpy_array))

        if i % 1000 == 0:
            logger.info(f"Processed {i} examples")

    for field_name, stats in statistics.items():
        if stats["count"] > 0:
            stats["std"] = np.sqrt(stats["M2"] / stats["count"])
        else:
            logger.warning(f"No valid numeric data for {field_name}")

        del stats["M2"]

        total = stats["count"] + stats["skipped"]
        stats["skipped_percentage"] = (stats["skipped"] / total) * 100 if total > 0 else 0

        for key, value in stats.items():
            if isinstance(value, np.number):
                stats[key] = value.item()

    try:
        logger.info(f"Saving statistics to {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f)
    except IOError as e:
        logger.error(f"Error saving statistics: {e}")

    logger.info("Statistics calculation completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to the input dataset")
    parser.add_argument("--output_path", type=str, help="Path to the output JSON file")
    args = parser.parse_args()

    calculate_statistics_for_mds(args.input_path, args.output_path)
