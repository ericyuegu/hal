"""Fingerprint the committed schema-3 loader cursor and its next batches.

Run this against the matching loader test in the recorded control checkout and
the refactored checkout. Both must print the same digest.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_test(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("loader_case", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load loader test {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_file", type=Path)
    args = parser.parse_args()
    module = _load_test(args.test_file.resolve())

    loader = module._loader(19)
    try:
        iterator = iter(loader)
        for _ in range(37):
            next(iterator)
        state = loader.state_dict()
        next_batches = []
        for _ in range(20):
            batch = next(iterator)
            next_batches.append(
                {"ids": batch.replay_ids, "sha256": hashlib.sha256(batch.values.numpy().tobytes()).hexdigest()}
            )
    finally:
        loader.close()

    restored = module._loader(19)
    try:
        restored.load_state_dict(state)
        iterator = iter(restored)
        restored_batches = []
        for _ in range(20):
            batch = next(iterator)
            restored_batches.append(
                {"ids": batch.replay_ids, "sha256": hashlib.sha256(batch.values.numpy().tobytes()).hexdigest()}
            )
    finally:
        restored.close()
    if next_batches != restored_batches:
        raise AssertionError("next batches differ after loader restore")

    normalized = {
        "state": {
            key: [
                (
                    slot.slot,
                    slot.locator.source,
                    slot.locator.shard,
                    slot.locator.row,
                    slot.epoch,
                    slot.replay_checksum,
                )
                for slot in value
            ]
            if key == "slots"
            else value
            for key, value in state.items()
        },
        "next_batches": next_batches,
    }
    print(hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest())


if __name__ == "__main__":
    main()
