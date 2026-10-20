"""Count only a checkpoint source's train rows, using its identity semantics."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from hal.data.player_identity import PlayerIdentitySidecar
from hal.data.policy_schema import policy_replay_identity
from hal.representation.player_identity import PlayerVocabulary

_SPEC = importlib.util.spec_from_file_location(
    "o65_sampler_builder", Path(__file__).parents[1] / "scripts/build_o65_sampler.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
count_identities = _MODULE.count_identities


def test_counts_use_train_rows_characters_and_the_professional_sidecar(tmp_path: Path) -> None:
    vocabulary = PlayerVocabulary(("PRO#1", "PRO#2"))
    rows = []
    for index, split in enumerate(("train", "train", "val")):
        rows.append(
            {
                "path": f"data/game{index}.slp",
                "slp_version": [3, 15, 0],
                "stage": 32,
                "players": [
                    {
                        "port": port,
                        "character": character,
                        "costume": 0,
                        "player_type": "HUMAN",
                        "code": None,
                        "name": name,
                    }
                    for port, character, name in ((2, 22, "Master Player"), (1, 1, "Diamond Player"))
                ],
                "frame_count": 1000,
                "schema_version": 7,
                "annotation": {
                    "replay_uuid": index,
                    "split": split,
                    "mds_row_idx": index if split == "train" else 0,
                    "frame_count_actual": 1000,
                    "schema_version": 7,
                },
            }
        )
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    sidecar = PlayerIdentitySidecar(
        vocabulary, {policy_replay_identity(str(row["path"])): (5, 6) for row in rows}, {}, "x" * 64
    )
    ranked = count_identities(path, sidecar, professional=False)
    professional = count_identities(path, sidecar, professional=True)
    assert ranked.train_rows == professional.train_rows == 2
    assert ranked.matchups == Counter({(1, 22): 2})
    assert ranked.identities == (
        {"identity": 2, "character": 1, "count": 2},
        {"identity": 3, "character": 22, "count": 2},
    )
    assert professional.identities == (
        {"identity": 5, "character": 1, "count": 2},
        {"identity": 6, "character": 22, "count": 2},
    )


def test_matchup_weights_use_sqrt_count_and_keep_single_replays() -> None:
    rows = _MODULE.matchup_weights(Counter({(1, 1): 1, (1, 22): 4, (22, 22): 9}))
    assert rows == [
        {"characters": [1, 1], "weight": 1.0},
        {"characters": [1, 22], "weight": 2.0},
        {"characters": [22, 22], "weight": 3.0},
    ]
