"""Validate the compact policy encoding against every source replay."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tyro
from tqdm import tqdm

from hal.data.mds import open_shard
from hal.data.mds import read_shard_index
from hal.data.policy_schema import PLAYER_PREFIXES
from hal.data.policy_schema import assert_policy_replay_equal
from hal.data.policy_schema import encode_policy_replay
from hal.scripts.project_policy_mds import DEFAULT_SCRATCH
from hal.wire import MASK_INT32


def audit_policy_mds(
    src: Path,
    *,
    splits: tuple[str, ...] = ("train", "val", "test"),
    scratch: Path = DEFAULT_SCRATCH,
) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    report = {}
    with TemporaryDirectory(dir=scratch) as run_scratch:
        for split in splits:
            shards = read_shard_index(src, split)
            expected = sum(int(info["samples"]) for info in shards)
            rows = 0
            frames = 0
            source_bytes = 0
            compact_bytes = 0
            nana_present = {prefix: 0 for prefix in PLAYER_PREFIXES if prefix.endswith("_nana")}
            action_max = -1
            action_over_511_frames = 0
            action_over_511_replays = 0
            with tqdm(total=expected, desc=f"audit {split}", unit="replay") as bar:
                for info in shards:
                    with open_shard(src, split, info, Path(run_scratch)) as reader:
                        for shard_row, source in enumerate(reader):
                            shard = info["raw_data"]["basename"]
                            where = f"{split} row {rows}, {shard} row {shard_row}"
                            try:
                                compact = encode_policy_replay(source, f"{split}:{rows}")
                                assert_policy_replay_equal(source, compact, where)
                            except (TypeError, ValueError) as error:
                                raise ValueError(f"{where}: {error}") from error
                            frames += len(source["frame"])
                            source_bytes += sum(
                                value.nbytes for value in source.values() if isinstance(value, np.ndarray)
                            )
                            compact_bytes += sum(
                                value.nbytes for value in compact.values() if isinstance(value, np.ndarray)
                            )
                            for prefix in nana_present:
                                nana_present[prefix] += int(compact[f"{prefix}_present"])
                            replay_has_large_action = False
                            for prefix in PLAYER_PREFIXES:
                                action = np.asarray(source[f"{prefix}_action"])
                                valid = action[action != MASK_INT32]
                                if valid.size:
                                    action_max = max(action_max, int(valid.max()))
                                    large = int((valid > 511).sum())
                                    action_over_511_frames += large
                                    replay_has_large_action |= large > 0
                            action_over_511_replays += int(replay_has_large_action)
                            rows += 1
                            bar.update(1)
            report[split] = {
                "rows": rows,
                "frames": frames,
                "source_array_bytes": source_bytes,
                "compact_array_bytes": compact_bytes,
                "array_ratio": source_bytes / compact_bytes,
                "nana_present_replays": nana_present,
                "action_max": action_max,
                "action_over_511_frames": action_over_511_frames,
                "action_over_511_replays": action_over_511_replays,
            }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    tyro.cli(audit_policy_mds)
