import random
from pathlib import Path
from typing import Optional
from typing import cast

import numpy as np
import torch
from streaming import StreamingDataset
from tensordict import TensorDict

from hal.constants import Player
from hal.constants import VALID_PLAYERS
from hal.data.stats import load_dataset_stats
from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig
from hal.training.preprocess.registry import InputPreprocessRegistry
from hal.training.preprocess.registry import TargetPreprocessRegistry


class HALStreamingDataset(StreamingDataset):
    def __init__(
        self,
        local: Optional[str],
        remote: Optional[str],
        batch_size: int,
        shuffle: bool,
        data_config: DataConfig,
        embed_config: EmbeddingConfig,
        stats_path: Path,
    ) -> None:
        super().__init__(local=local, remote=remote, batch_size=batch_size, shuffle=shuffle)
        self.stats_by_feature_name = load_dataset_stats(stats_path)
        self.data_config = data_config
        self.embed_config = embed_config
        self.input_len = data_config.input_len
        self.trajectory_len = data_config.input_len + data_config.target_len

        self.input_preprocessing_fn = InputPreprocessRegistry.get(self.embed_config.input_preprocessing_fn)
        self.target_preprocessing_fn = TargetPreprocessRegistry.get(self.embed_config.target_preprocessing_fn)

    def get_td_from_sample(self, sample: dict[str, np.ndarray]) -> TensorDict:
        episode_len = len(sample["frame"])
        random_start_idx = random.randint(0, episode_len - self.trajectory_len)
        sample_slice = {
            k: torch.from_numpy(v[random_start_idx : random_start_idx + self.trajectory_len].copy())
            for k, v in sample.items()
        }
        return TensorDict(sample_slice, batch_size=(self.trajectory_len,))

    def __getitem__(self, idx: int | slice | list[int] | np.ndarray) -> TensorDict:
        sample = super().__getitem__(idx)
        sample_td = self.get_td_from_sample(sample)

        player_perspective = cast(Player, random.choice(VALID_PLAYERS))
        inputs = self.input_preprocessing_fn(
            sample_td, self.data_config, player_perspective, self.stats_by_feature_name
        )
        # TODO refactor this to be less confusing / hacky
        targets = self.target_preprocessing_fn(sample_td[1:], player_perspective)

        return TensorDict(
            {
                "inputs": inputs,
                "targets": targets,
            },
            # TODO refactor this to be less confusing / footgun-y
            batch_size=(self.input_len,),
        )
