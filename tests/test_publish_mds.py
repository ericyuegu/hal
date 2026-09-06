import json
from pathlib import Path

from hal.scripts import publish_mds as module


def test_audit_accepts_an_index_referencing_raw_shards(tmp_path: Path, monkeypatch) -> None:
    raw = {"basename": "shard.00000.mds", "bytes": 12, "hashes": {"md5": "abc"}}
    index = {"shards": [{"samples": 1, "raw_data": raw, "zip_data": None, "compression": None}]}
    objects = {
        "manifest.jsonl": {"Size": 1},
        "stats.json": {"Size": 1},
        "projection.json": {"Size": 1},
        "failures.materialize.jsonl": {"Size": 0},
        **{f"{split}/index.json": {"Size": 1} for split in ("train", "val", "test")},
        **{f"{split}/shard.00000.mds": {"Size": 12, "Hashes": {"md5": "abc"}} for split in ("train", "val", "test")},
    }
    monkeypatch.setattr(module, "_objects", lambda _prefix: objects)
    monkeypatch.setattr(
        module,
        "_cat_json",
        lambda path: index if path.endswith("index.json") else {"rows": {"train": 1, "val": 1, "test": 1}},
    )

    def copy_file(source: str, destination: Path) -> None:
        assert source.endswith("/manifest.jsonl")
        rows = [{"annotation": {"split": split, "mds_row_idx": 0}} for split in ("train", "val", "test")]
        destination.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    monkeypatch.setattr(module.r2, "copy_file", copy_file)

    result = module.audit("r2:test/raw")

    assert result["rows"] == {"train": 1, "val": 1, "test": 1}
    assert result["shards"] == 3
    assert result["object_hashes"]["train/shard.00000.mds"] == {"bytes": 12, "md5": "abc"}
