from pathlib import Path
from typing import Dict
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from hal.data.preprocessing import pyarrow_table_to_np_dict
from hal.data.schema import SCHEMA


class MmappedParquetDataset(Dataset):
    """Memory mapped parquet dataset for DDP training."""

    def __init__(self, input_path: str, input_len: int, target_len: int, mask_multi_uuid: bool = True) -> None:
        """
        Initialize the dataset.

        Args:
            input_path (str): Path to the parquet file.
            input_len (int): Length of the input sequence.
            target_len (int): Length of the target sequence.
            cache_size (int): Number of items to cache in memory.

        Raises:
            ValueError: If input parameters are invalid.
            FileNotFoundError: If the input file doesn't exist.
        """
        if input_len <= 0 or target_len <= 0:
            raise ValueError("input_len and target_len must be positive integers")
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self.input_path = input_path
        self.input_len = input_len
        self.target_len = target_len
        self.trajectory_len = input_len + target_len
        self.mask_multi_uuid = mask_multi_uuid

        self._parquet_table: Optional[pa.Table] = None

    @property
    def parquet_table(self) -> pa.Table:
        """Lazy-load the parquet table."""
        if self._parquet_table is None:
            self._parquet_table = pq.read_table(self.input_path, schema=SCHEMA, memory_map=True)
        return self._parquet_table

    def __len__(self) -> int:
        return len(self.parquet_table) - self.trajectory_len

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        chunked_table = self.parquet_table[index : index + self.trajectory_len]
        feature_array_by_name = pyarrow_table_to_np_dict(chunked_table)

        # Truncate to the first uuid
        if self.mask_multi_uuid:
            first_uuid = feature_array_by_name["replay_uuid"][0]
            mask = feature_array_by_name["replay_uuid"] == first_uuid
            for key in feature_array_by_name:
                feature_array_by_name[key] = feature_array_by_name[key][mask]

        return feature_array_by_name
