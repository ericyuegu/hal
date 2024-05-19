import functools
import subprocess

from pathlib import Path
from typing import Iterable, Any, TypeVar

import numpy as np
import torch
import os

T = TypeVar("T")


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def get_git_repo_root() -> Path:
    cmd = subprocess.check_output("git rev-parse --show-toplevel".split(" "))
    root_dir = Path(cmd.decode("utf-8").strip(" \n"))
    assert root_dir.is_dir()
    return root_dir


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def get_exp_name(config) -> str:
    return '_'.join(f'{k}@{v}' for k, v in sorted(vars(config).items()) if k not in
                    ('arch', 'cloud', 'dataset', 'num_data_workers', 'n_val_samples', 'keep_ckpts', 'report_len',
                     'betas', 'eps', 'wd', 'debug'))


def report_module_weights(m: torch.nn.Module):
    weights = [(k, tuple(v.shape)) for k, v in m.named_parameters()]
    weights.append((f'Total ({len(weights)})', (sum(np.prod(x[1]) for x in weights),)))
    width = max(len(x[0]) for x in weights)
    return '\n'.join(f'{k:<{width}} {np.prod(s):>10} {str(s):>16}' for k, s in weights)


def repeater(it: Iterable):
    """Helper function to repeat an iterator in a memory efficient way."""
    while True:
        for x in it:
            yield x


def time_format(t: float) -> str:
    t = int(t)
    hours = t // 3600
    mins = (t // 60) % 60
    secs = t % 60
    return f'{hours:02d}:{mins:02d}:{secs:02d}'


def move_tensors_to_device(inputs: T, device: str, non_blocking=True) -> T:
    if isinstance(inputs, dict):
        return {k: move_tensors_to_device(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)(move_tensors_to_device(v, device) for v in inputs)
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device, non_blocking=non_blocking)
    else:
        return inputs
