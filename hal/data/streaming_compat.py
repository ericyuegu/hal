from multiprocessing import resource_tracker
from typing import Any

from streaming.base.shared.memory import SharedMemory


def _register(self: SharedMemory, name: str, resource_type: str) -> Any:
    if resource_type == "shared_memory":
        return None
    return resource_tracker._resource_tracker.register(name, resource_type)


def _unregister(self: SharedMemory, name: str, resource_type: str) -> Any:
    if resource_type == "shared_memory":
        return None
    return resource_tracker._resource_tracker.unregister(name, resource_type)


def patch_streaming_resource_tracker() -> None:
    SharedMemory.fix_register = _register
    SharedMemory.fix_unregister = _unregister
