"""Adapt the common loader measurement code to the immutable pre-refactor API."""

import argparse
import inspect
from pathlib import Path

from hal.training.runs import source_git_sha


def main(source: Path, output: Path) -> None:
    code = source.read_text()
    replacements = {
        "from hal.representation.features import BASE_ACTION_PROJECTION": (
            "from hal.training.features import BASE_ACTION_PROJECTION"
        ),
        "from hal.training.buffered_mds_replay_loader import BufferedMDSReplayLoader": (
            "from hal.training.physical_shard_loader import PhysicalShardReplayLoader as BufferedMDSReplayLoader"
        ),
        "from hal.training.buffered_mds_replay_loader import ": "from hal.training.physical_shard_loader import ",
        "from hal.training.runs import source_git_sha\n": "",
    }
    for old, new in replacements.items():
        if old not in code:
            raise ValueError(f"control benchmark adaptation is stale: {old}")
        code = code.replace(old, new)
    # Only the measurement program needs the current archive identity helper;
    # the immutable control's training package stays untouched.
    code = code.replace("import platform\n", "import platform\nimport re\nimport subprocess\n")
    code = code.replace("def _source_metadata()", inspect.getsource(source_git_sha) + "\n\ndef _source_metadata()")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
