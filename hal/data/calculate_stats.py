import argparse
import json

import numpy as np
import pyarrow as pa
from loguru import logger
from pyarrow import parquet as pq


def calculate_statistics_for_features(input_path: str, output_path: str) -> None:
    """Calculate and save statistics for each feature to a JSON."""
    table: pa.Table = pq.read_table(input_path)
    statistics = {}

    for field in table.schema:
        logger.info(f"Calculating statistics for {field.name}")
        column = table[field.name]
        numpy_array = column.to_numpy()

        stats = {
            "mean": float(np.mean(numpy_array)),
            "std": float(np.std(numpy_array)),
            "min": float(np.min(numpy_array)),
            "max": float(np.max(numpy_array)),
            "median": float(np.median(numpy_array)),
        }

        statistics[field.name] = stats

    logger.info(f"Saving statistics to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(statistics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="Path to the input parquet file")
    parser.add_argument("--output_path", type=str, help="Path to the output JSON file")
    args = parser.parse_args()

    calculate_statistics_for_features(args.input_path, args.output_path)
