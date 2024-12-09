import random
from typing import Optional
from typing import cast

import numpy as np
from streaming import StreamingDataset
from tensordict import TensorDict

from hal.constants import Player
from hal.constants import VALID_PLAYERS
from hal.training.config import DataConfig
from hal.training.config import EmbeddingConfig
from hal.training.preprocess.preprocessor import Preprocessor


class HALStreamingDataset(StreamingDataset):
    def __init__(
        self,
        local: Optional[str],
        remote: Optional[str],
        batch_size: int,
        shuffle: bool,
        data_config: DataConfig,
        embedding_config: EmbeddingConfig,
    ) -> None:
        super().__init__(local=local, remote=remote, batch_size=batch_size, shuffle=shuffle)
        self.preprocessor = Preprocessor(data_config=data_config, embedding_config=embedding_config)
        self.seq_len = self.preprocessor.seq_len

    def __getitem__(self, idx: int | slice | list[int] | np.ndarray) -> TensorDict:
        """Expects episode features to match data/schema.py."""
        episode_features_by_name = super().__getitem__(idx)
        sample_td = self.preprocessor.sample_from_episode(episode_features_by_name)

        player_perspective = cast(Player, random.choice(VALID_PLAYERS))
        inputs = self.preprocessor.preprocess_inputs(sample_td, player_perspective)
        # targets = self.preprocessor.preprocess_targets(sample_td, player_perspective)

        return TensorDict(
            {
                "inputs": inputs,
                # "targets": targets,  # type: ignore
            },
            batch_size=(self.seq_len,),
        )
