# %%
from pathlib import Path

import attr
from tensordict import TensorDict

from hal.data.process_replays import process_replay
from hal.training.config import DataConfig
from hal.training.io import Checkpoint
from hal.training.io import load_config_from_artifact_dir
from hal.training.models.registry import Arch
from hal.training.preprocess.preprocessor import Preprocessor

# %%
# We want to load a .slp file from closed loop eval and run it back through data preprocessing pipeline & model forward pass to verify that the stack is working correctly
run = "2024-12-09_18-53-21/arch@GPTv1-12-4-dropout_local_batch_size@256_n_samples@4194304"
artifact_dir = Path(f"/opt/projects/hal2/runs/{run}")
replay_path = f"/opt/projects/hal2/replays/{run}/Game_20241215T162052.slp"

np_dict = process_replay(replay_path, check_damage=False)
assert np_dict is not None
episode_td = TensorDict(np_dict, batch_size=(len(np_dict["frame"]),))

# %%
config = load_config_from_artifact_dir(artifact_dir)
config = attr.evolve(config, data=DataConfig(data_dir="/opt/projects/hal2/data/mang0"))
preprocessor = Preprocessor(data_config=config.data, embedding_config=config.embedding)

# %%
arch = Arch.get(config.arch, preprocessor=preprocessor)
ckpt = Checkpoint(arch, config, artifact_dir, keep_ckpts=config.keep_ckpts)
ckpt.restore(idx=None, device="cpu")
model = ckpt.model

# %%
traj_len = preprocessor.trajectory_sampling_len
for i in range(len(episode_td) - traj_len):
    trajectory_td = episode_td[i : i + traj_len]
    offset_td = preprocessor.offset_features(trajectory_td)
    inputs = preprocessor.preprocess_inputs(offset_td, "p1")
    inputs = inputs.unsqueeze(0)
    preds = model(inputs)
    postprocessed_preds = preprocessor.postprocess_preds(preds[0, -1])
    print(postprocessed_preds)
    break

# %%


# %%
