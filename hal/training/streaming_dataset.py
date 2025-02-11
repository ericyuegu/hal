import random
from typing import Optional
from typing import cast

import numpy as np
from streaming import StreamingDataset
from tensordict import TensorDict

from hal.constants import Player
from hal.constants import VALID_PLAYERS
from hal.preprocess.preprocessor import Preprocessor
from hal.training.config import DataConfig


class HALStreamingDataset(StreamingDataset):
    def __init__(
        self,
        local: Optional[str],
        remote: Optional[str],
        batch_size: int,
        shuffle: bool,
        data_config: DataConfig,
        debug: bool = False,
    ) -> None:
        super().__init__(local=local, remote=remote, batch_size=batch_size, shuffle=shuffle)
        self.preprocessor = Preprocessor(data_config=data_config)
        self.seq_len = self.preprocessor.seq_len
        self.debug = debug

    def __getitem__(self, idx: int | slice | list[int] | np.ndarray) -> TensorDict:
        """Expects episode features to match data/schema.py."""
        episode_features_by_name = super().__getitem__(idx)
        sample_T = self.preprocessor.sample_from_episode(episode_features_by_name, debug=self.debug)

        player_perspective = "p1" if self.debug else cast(Player, random.choice(VALID_PLAYERS))
        inputs_T = self.preprocessor.preprocess_inputs(sample_T, player_perspective)
        targets_T = self.preprocessor.preprocess_targets(sample_T, player_perspective)

        feature_names = set(inputs_T.keys()) | set(targets_T.keys())  # type: ignore
        offset_keys = set(self.preprocessor.frame_offsets_by_feature.keys())
        assert all(
            feature in feature_names for feature in offset_keys
        ), f"Features with offsets must exist in sample. Missing: {offset_keys - feature_names}\nAvailable: {feature_names}"
        inputs_L = self.preprocessor.offset_features(inputs_T)
        targets_L = self.preprocessor.offset_features(targets_T)

        return TensorDict(
            {
                "inputs": inputs_L,
                "targets": targets_L,  # type: ignore
            },
            batch_size=(self.seq_len,),
        )
