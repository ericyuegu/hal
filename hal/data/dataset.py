from pathlib import Path
from typing import Dict
from typing import Optional

import attr
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from hal.data.preprocessing import preprocess_inputs_v0
from hal.data.preprocessing import preprocess_targets_v0
from hal.data.preprocessing import pyarrow_table_to_np_dict
from hal.data.schema import SCHEMA


@attr.s(auto_attribs=True, frozen=True)
class ReplayFilter:
    """Filter for replay."""

    replay_uuid: Optional[str] = None
    stage: Optional[str] = None
    character: Optional[str] = None


class MmappedParquetDataset(Dataset):
    """Memory mapped parquet dataset for DDP training.

    If sequence spans multiple replays, `truncate_replay_end` will truncate to the first replay.
    """

    def __init__(self, input_path: str, input_len: int, target_len: int, truncate_replay_end: bool = False) -> None:
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
        self.mask_multi_uuid = truncate_replay_end

        self.parquet_table = pq.read_table(self.input_path, schema=SCHEMA, memory_map=True)

    def __len__(self) -> int:
        return len(self.parquet_table) - self.trajectory_len

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        chunked_table = self.parquet_table[index : index + self.trajectory_len]

        # Truncate to the first replay
        if self.mask_multi_uuid:
            first_uuid = chunked_table["replay_uuid"][0].as_py()
            mask = pa.compute.equal(chunked_table["replay_uuid"], first_uuid)
            chunked_table = chunked_table.filter(mask)

        feature_array_by_name = pyarrow_table_to_np_dict(chunked_table)
        inputs = preprocess_inputs_v0(feature_array_by_name)
        targets = preprocess_targets_v0(feature_array_by_name)
        return inputs, targets
