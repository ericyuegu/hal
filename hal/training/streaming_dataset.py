import random
from pathlib import Path
from typing import Optional

from streaming import StreamingDataset
from tensordict import TensorDict

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
        self.trajectory_len = data_config.input_len + data_config.target_len

        self.input_preprocessing_fn = InputPreprocessRegistry.get(self.embed_config.input_preprocessing_fn)
        self.target_preprocessing_fn = TargetPreprocessRegistry.get(self.embed_config.target_preprocessing_fn)

    def get_td_from_sample(self, sample: dict) -> TensorDict:
        episode_len = len(sample["frame"])
        random_start_idx = random.randint(0, episode_len - self.trajectory_len)
        td = TensorDict(sample, batch_size=(episode_len,))[random_start_idx : random_start_idx + self.trajectory_len]
        return td

    def __getitem__(self, idx: int) -> TensorDict:
        sample = super().__getitem__(idx)
        sample_td = self.get_td_from_sample(sample)

        player_perspective = random.choice(VALID_PLAYERS)
        inputs = self.input_preprocessing_fn(
            sample_td, self.data_config, player_perspective, self.stats_by_feature_name
        )
        targets = self.target_preprocessing_fn(sample_td, player_perspective)

        return TensorDict(
            {
                "inputs": inputs,
                "targets": targets,
            },
            batch_size=(self.trajectory_len,),
        )
