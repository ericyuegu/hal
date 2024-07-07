from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Tuple

import attr
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from hal.data.constants import IDX_BY_CHARACTER_STR
from hal.data.constants import IDX_BY_STAGE_STR
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
        include_both_players: bool = True,  # New parameter
    ) -> None:
        """
        Initialize the dataset.

        Args:
            input_path (str): Path to the parquet file.
            input_len (int): Length of the input sequence.
            target_len (int): Length of the target sequence.
            truncate_replay_end (bool): Whether to truncate sequences at replay boundaries.
            replay_filter (Optional[ReplayFilter]): Filter for replays.
            include_both_players (bool): Whether to include both player 1 and player 2 perspectives.

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
        self.truncate_replay_end = truncate_replay_end
        self.include_both_players = include_both_players
        self.player_perspectives = ["p1", "p2"] if include_both_players else ["p1"]

        self.parquet_table = pq.read_table(self.input_path, schema=SCHEMA, memory_map=True)

        self.replay_filter = replay_filter
        self.filtered_indices = self._apply_filter()

    def _apply_filter(self) -> np.ndarray:
        if self.replay_filter is None:
            return np.arange(len(self.parquet_table) - self.trajectory_len)

        filter_conditions = []

        if self.replay_filter.replay_uuid is not None:
            filter_conditions.append(
                pa.compute.equal(self.parquet_table["replay_uuid"], self.replay_filter.replay_uuid)
            )

        if self.replay_filter.stage is not None:
            stage_idx = IDX_BY_STAGE_STR[self.replay_filter.stage]
            filter_conditions.append(pa.compute.equal(self.parquet_table["stage"], stage_idx))

        if self.replay_filter.character is not None:
            character_idx = IDX_BY_CHARACTER_STR[self.replay_filter.character]
            filter_conditions.append(
                pa.compute.or_(
                    pa.compute.equal(self.parquet_table["p1_character"], character_idx),
                    pa.compute.equal(self.parquet_table["p2_character"], character_idx),
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
        return len(self.filtered_indices) * len(self.player_perspectives)

    def __getitem__(self, index: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        player_index = index % len(self.player_perspectives)
        actual_index = self.filtered_indices[index // len(self.player_perspectives)]
        input_table_chunk = self.parquet_table[actual_index : actual_index + self.input_len]
        target_table_chunk = self.parquet_table[actual_index + self.input_len : actual_index + self.trajectory_len]

        # Truncate to the first replay
        if self.truncate_replay_end:
            first_uuid = input_table_chunk["replay_uuid"][0].as_py()
            mask = pa.compute.equal(input_table_chunk["replay_uuid"], first_uuid)
            input_table_chunk = input_table_chunk.filter(mask)
            target_table_chunk = target_table_chunk.filter(mask)

        input_features_by_name = pyarrow_table_to_np_dict(input_table_chunk)
        target_features_by_name = pyarrow_table_to_np_dict(target_table_chunk)
        player = self.player_perspectives[player_index]
        inputs = preprocess_inputs_v0(input_features_by_name, player=player)
        targets = preprocess_targets_v0(target_features_by_name, player=player)
        return inputs, targets
