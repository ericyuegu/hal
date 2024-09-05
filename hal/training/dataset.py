from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import pyarrow.parquet as pq
from loguru import logger
from tensordict import TensorDict
from torch.utils.data import Dataset

from hal.data.constants import IDX_BY_CHARACTER_STR
from hal.data.constants import IDX_BY_STAGE_STR
from hal.data.schema import SCHEMA
from hal.data.stats import load_dataset_stats
from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig
from hal.training.utils import pyarrow_table_to_tensordict
from hal.training.zoo.preprocess.registry import InputPreprocessRegistry
from hal.training.zoo.preprocess.registry import TargetPreprocessRegistry


def _create_filters_from_replay_filter(data_config: DataConfig) -> Optional[List[List[Tuple[str, str, Any]]]]:
    filter_config = data_config.replay_filter

    # Handle replay_uuid filter separately as it's the most specific
    if filter_config.replay_uuid:
        return [[("replay_uuid", "=", filter_config.replay_uuid)]]

    filters = []

    # Add stage filter if present
    if filter_config.stage:
        stage_filter = ("stage", "=", IDX_BY_STAGE_STR[filter_config.stage])
        filters.append(stage_filter)

    # Create character filters
    character_filters = []
    for player in ["ego", "opponent"]:
        character = getattr(filter_config, f"{player}_character")
        if character:
            character_idx = IDX_BY_CHARACTER_STR[character]
            character_filters.extend([("p1_character", "=", character_idx), ("p2_character", "=", character_idx)])

    # Combine filters based on what's present
    if filters and character_filters:
        # Both stage and character filters: AND stage with each character filter
        return [filters + [cf] for cf in character_filters]
    elif character_filters:
        # Only character filters: each in its own list
        return [[cf] for cf in character_filters]
    elif filters:
        # Only stage filter
        return [filters]
    else:
        # No filters
        return None


def load_filtered_parquet_as_tensordict(
    input_path: str | Path,
    data_config: DataConfig,
) -> TensorDict:
    filters = _create_filters_from_replay_filter(data_config) or None
    table = pq.read_table(input_path, schema=SCHEMA, filters=filters)
    num_unique_replays = len(table["replay_uuid"].unique())
    logger.info(f"Loaded {num_unique_replays} replays from {input_path}")
    tensordict = pyarrow_table_to_tensordict(table)
    return tensordict


class InMemoryDataset(Dataset):
    def __init__(
        self,
        tensordict: TensorDict,
        stats_path: Path,
        data_config: DataConfig,
        embed_config: EmbeddingConfig,
    ) -> None:
        self.tensordict = tensordict
        self.stats_by_feature_name = load_dataset_stats(stats_path)
        self.data_config = data_config
        self.embed_config = embed_config

        self.input_len = data_config.input_len
        self.target_len = data_config.target_len
        self.trajectory_len = self.input_len + self.target_len
        self.include_both_players = data_config.include_both_players

        self.input_preprocessing_fn = InputPreprocessRegistry.get(self.embed_config.input_preprocessing_fn)
        self.target_preprocessing_fn = TargetPreprocessRegistry.get(self.embed_config.target_preprocessing_fn)

    def __len__(self) -> int:
        if self.include_both_players:
            return 2 * len(self.tensordict) - self.trajectory_len
        return len(self.tensordict) - self.trajectory_len

    def __getitem__(self, idx: int) -> TensorDict:
        assert isinstance(idx, int), "Index must be an integer."
        assert 0 <= idx < len(self), "Index out of bounds."

        # TODO eric: dynamically determine player perspective
        player = "p1"

        sample: TensorDict = self.tensordict[idx : idx + self.trajectory_len]
        inputs = self.input_preprocessing_fn(sample, self.data_config, player, self.stats_by_feature_name)
        targets = self.target_preprocessing_fn(sample, player)

        return TensorDict(
            {
                "inputs": inputs,
                "targets": targets,
            }  # type: ignore
        )

    # TODO implement __getitems__ for batched sampling
