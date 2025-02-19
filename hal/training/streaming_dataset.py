import random
from typing import Optional
from typing import Sequence
from typing import cast

import numpy as np
from streaming import Stream
from streaming import StreamingDataset
from tensordict import TensorDict

from hal.constants import Player
from hal.constants import VALID_PLAYERS
from hal.preprocess.preprocessor import Preprocessor
from hal.training.config import DataConfig


class HALStreamingDataset(StreamingDataset):
    def __init__(
        self,
        streams: Optional[Sequence[Stream]],
        local: Optional[str],
        remote: Optional[str],
        batch_size: int,
        shuffle: bool,
        data_config: DataConfig,
        debug: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(streams=streams, local=local, remote=remote, batch_size=batch_size, shuffle=shuffle, **kwargs)
        self.preprocessor = Preprocessor(data_config=data_config)
        self.seq_len = self.preprocessor.seq_len
        self.debug = debug

    def __getitem__(self, idx: int | slice | list[int] | np.ndarray) -> TensorDict:
        """Expects episode features to match data/schema.py."""
        episode_features_by_name = super().__getitem__(idx)
        sample_T = self.preprocessor.sample_td_from_episode(episode_features_by_name, debug=self.debug)

        player_perspective = "p1" if self.debug else cast(Player, random.choice(VALID_PLAYERS))
        inputs_T = self.preprocessor.preprocess_inputs(sample_T, player_perspective)
        targets_T = self.preprocessor.preprocess_targets(sample_T, player_perspective)

        inputs_L = self.preprocessor.offset_inputs(inputs_T)
        targets_L = self.preprocessor.offset_targets(targets_T)

        return TensorDict(
            {
                "inputs": inputs_L,
                "targets": targets_L,  # type: ignore
            },
            batch_size=(self.seq_len,),
        )
