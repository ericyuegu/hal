import functools
import subprocess
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import TypeVar

import numpy as np
import pyarrow as pa
import torch
from tensordict import TensorDict

from hal.training.config import EmbeddingConfig
from hal.training.zoo.preprocess.registry import InputPreprocessRegistry

T = TypeVar("T")


def get_git_repo_root() -> Path:
    cmd = subprocess.check_output("git rev-parse --show-toplevel".split(" "))
    root_dir = Path(cmd.decode("utf-8").strip(" \n"))
    assert root_dir.is_dir()
    return root_dir


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def report_module_weights(m: torch.nn.Module):
    weights = [(k, tuple(v.shape)) for k, v in m.named_parameters()]
    weights.append((f"Total ({len(weights)})", (sum(np.prod(x[1]) for x in weights),)))
    width = max(len(x[0]) for x in weights)
    return "\n".join(f"{k:<{width}} {np.prod(s):>10} {str(s):>16}" for k, s in weights)


def repeater(it: Iterable[T]) -> Iterator[T]:
    """Helper function to repeat an iterator in a memory efficient way."""
    while True:
        for x in it:
            yield x


def time_format(t: float) -> str:
    t = int(t)
    hours = t // 3600
    mins = (t // 60) % 60
    secs = t % 60
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def move_tensors_to_device(inputs: T, device: str, non_blocking=True) -> T:
    if isinstance(inputs, dict):
        return {k: move_tensors_to_device(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)(move_tensors_to_device(v, device) for v in inputs)
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=non_blocking)
    else:
        return inputs


def pyarrow_table_to_np_dict(table: pa.Table) -> Dict[str, np.ndarray]:
    """
    Convert pyarrow table to dictionary of numpy arrays.

    Use copy=True to ensure that the numpy arrays are not views of the original data for safe downstream processing.
    """
    return {name: np.array(col.to_numpy(), copy=True) for name, col in zip(table.column_names, table.columns)}


def pyarrow_table_to_tensordict(table: pa.Table) -> TensorDict:
    return TensorDict(
        {name: torch.from_numpy(col.to_numpy()) for name, col in zip(table.column_names, table.columns)},
        batch_size=len(table),
    )


def get_input_size_from_config(config: EmbeddingConfig) -> int:
    """Get the size of the materialized input dimensions from the embedding config."""
    numeric_feature_count = InputPreprocessRegistry.get_num_features(config.input_preprocessing_fn)
    return (
        numeric_feature_count
        + config.stage_embedding_dim
        + 2 * (config.character_embedding_dim + config.action_embedding_dim)
    )
