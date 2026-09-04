"""Compatibility names for experiment-051 v5 loader checkpoints.

New code must import :mod:`hal.training.physical_shard_loader`. This module
exists only because existing checkpoints pickle these two qualified names.
Remove it when experiment-051 v5 checkpoints are no longer supported.
"""

from hal.training.physical_shard_loader import GenerationDescriptor
from hal.training.physical_shard_loader import PhysicalRow

__all__ = ["GenerationDescriptor", "PhysicalRow"]
