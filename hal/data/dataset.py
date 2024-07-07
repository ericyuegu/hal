from pathlib import Path
from typing import Dict
from typing import Final
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

    def __init__(
        self,
        input_path: str,
        input_len: int,
        target_len: int,
        truncate_replay_end: bool = False,
        replay_filter: Optional[ReplayFilter] = None,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            input_path (str): Path to the parquet file.
            input_len (int): Length of the input sequence.
            target_len (int): Length of the target sequence.
            truncate_replay_end (bool): Whether to truncate sequences at replay boundaries.
            replay_filter (Optional[ReplayFilter]): Filter for replays.

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

        self.replay_filter = replay_filter
        self.filtered_indices = self._apply_filter()

        self.player: Final[str] = "p1"

    def _apply_filter(self) -> np.ndarray:
        if self.replay_filter is None:
            return np.arange(len(self.parquet_table) - self.trajectory_len)

        filter_conditions = []

        if self.replay_filter.replay_uuid is not None:
            filter_conditions.append(
                pa.compute.equal(self.parquet_table["replay_uuid"], self.replay_filter.replay_uuid)
            )

        if self.replay_filter.stage is not None:
            filter_conditions.append(pa.compute.equal(self.parquet_table["stage"], self.replay_filter.stage))

        if self.replay_filter.character is not None:
            filter_conditions.append(
                pa.compute.or_(
                    pa.compute.equal(self.parquet_table["p1_character"], self.replay_filter.character),
                    pa.compute.equal(self.parquet_table["p2_character"], self.replay_filter.character),
                )
            )

        if filter_conditions:
            combined_filter = pa.compute.and_(*filter_conditions)
            filtered_indices = np.where(combined_filter.to_numpy())[0]
            valid_indices = filtered_indices[filtered_indices < len(self.parquet_table) - self.trajectory_len]
            return valid_indices
        else:
            return np.arange(len(self.parquet_table) - self.trajectory_len)

    def __len__(self) -> int:
        return len(self.filtered_indices)

    def __getitem__(self, index: int) -> Dict[str, np.ndarray]:
        actual_index = self.filtered_indices[index]
        chunked_table = self.parquet_table[actual_index : actual_index + self.trajectory_len]

        # Truncate to the first replay
        if self.mask_multi_uuid:
            first_uuid = chunked_table["replay_uuid"][0].as_py()
            mask = pa.compute.equal(chunked_table["replay_uuid"], first_uuid)
            chunked_table = chunked_table.filter(mask)

        feature_array_by_name = pyarrow_table_to_np_dict(chunked_table)
        inputs = preprocess_inputs_v0(feature_array_by_name, player=self.player)
        targets = preprocess_targets_v0(feature_array_by_name, player=self.player)
        return {
            "inputs": inputs,
            "targets": targets,
        }
