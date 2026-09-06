from pathlib import Path

import pytest

from hal import streams
from hal.scripts import scaleup_policy_world_v8 as module
from hal.scripts.scaleup_policy_world_v8 import PolicyWorldV8ScaleupConfig
from hal.scripts.scaleup_policy_world_v8 import scaleup_policy_world_v8
from hal.scripts.scaleup_policy_world_v8 import verify_and_delete_obsolete_rank_one_prefix


def _source() -> streams.StreamSource:
    return streams.StreamSource(
        name="professional-test-policy-world-v7",
        remote="s3://hal/processed/professional/test/mds-policy-world-v7",
        local=Path("data/processed/professional/test/mds-policy-world-v7"),
    )


def _audit() -> dict[str, object]:
    return {
        "rows": {"train": 8, "val": 1, "test": 1},
        "frames": {"train": 800, "val": 100, "test": 100},
        "source_rows": {"train": 10, "val": 1, "test": 1},
        "rejections": {"train": 2, "val": 0, "test": 0},
        "rejection_reasons": {"four_starting_stocks_per_player": 2},
        "ranks": {"observed": {"PRO": 10}, "imputed": {"MASTER": 10}, "sides": 20},
        "bytes": 1_000,
        "object_hashes": {},
    }


def test_scaleup_stages_a_git_named_attempt_and_publishes_only_after_global_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source()
    monkeypatch.setattr(module.streams, "POLICY_WORLD_V7_SOURCES", (source,))
    monkeypatch.setattr(module, "EXPECTED_V7_ROWS", 12)
    monkeypatch.setattr(module, "_git_sha", lambda: "a" * 40)
    state = {"built": False, "published": False, "global_audits": 0}

    def objects(prefix: str) -> list[str]:
        if prefix.endswith("professional/test/mds-policy-world-v8"):
            return ["_SUCCESS"] if state["published"] else []
        if "attempt-001" in prefix:
            return ["selection.json"] if state["built"] else []
        return []

    def build(source_prefix: str, output: str, **kwargs: object) -> dict[str, object]:
        assert source_prefix == "r2:hal/processed/professional/test/mds-policy-world-v7"
        assert "/" + "a" * 40 + "/professional/test/attempt-001/" in output
        assert kwargs["source_name"] == source.name
        state["built"] = True
        return _audit()

    def publish(staging: str, final: str, *, purge_staging: bool, audit_fn: object) -> None:
        assert staging.endswith("professional/test/attempt-001/mds-policy-world-v8")
        assert final == "r2:hal/processed/professional/test/mds-policy-world-v8"
        assert purge_staging and audit_fn is not None and state["global_audits"] == 1
        state["published"] = True

    def global_audit(_prefixes: dict[str, str], _scratch: Path) -> dict[str, object]:
        state["global_audits"] += 1
        return {"unique_sha1": 10, "rows": {"train": 8, "val": 1, "test": 1}}

    monkeypatch.setattr(module.r2, "list_files", objects)
    monkeypatch.setattr(module, "filter_policy_world_mds", build)
    monkeypatch.setattr(module, "audit_policy_world_v8", lambda *_args, **_kwargs: _audit())
    monkeypatch.setattr(module, "publish_mds", publish)
    monkeypatch.setattr(module, "_audit_manifest_uniqueness", global_audit)
    report_path = tmp_path / "publication.json"

    report = scaleup_policy_world_v8(PolicyWorldV8ScaleupConfig(report=report_path, scratch=tmp_path, publish=True))

    assert state == {"built": True, "published": True, "global_audits": 2}
    row = report["sources"][source.name]
    assert row["remote"] == "s3://hal/processed/professional/test/mds-policy-world-v8"
    assert report_path.is_file()


def test_scaleup_refuses_a_partial_final_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module.streams, "POLICY_WORLD_V7_SOURCES", (_source(),))
    monkeypatch.setattr(module, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(module.r2, "list_files", lambda _prefix: ["train/index.json"])

    with pytest.raises(FileExistsError, match="lacks _SUCCESS"):
        scaleup_policy_world_v8(PolicyWorldV8ScaleupConfig())


def test_obsolete_deletion_is_exact_and_refuses_a_success_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(module.r2, "list_files", lambda _prefix: ["train/index.json"])
    monkeypatch.setattr(module.r2, "run_rclone", lambda *args: calls.append(args) or "")

    removed = verify_and_delete_obsolete_rank_one_prefix()

    assert removed == ["train/index.json"]
    assert calls == [("purge", "r2:hal/processed/ranked-anonymized-1/mds-policy-world-v8")]
    monkeypatch.setattr(module.r2, "list_files", lambda _prefix: ["_SUCCESS"])
    with pytest.raises(ValueError, match="refusing to delete"):
        verify_and_delete_obsolete_rank_one_prefix()


def test_remote_rank_one_metadata_requires_a_pinned_single_object(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = "a" * 64
    remote = f"r2:hal/processed/_staging/policy-world-v8/sha/_metadata/ranked-1-index.{digest}.jsonl"
    monkeypatch.setattr(module.r2, "list_files", lambda _prefix: [Path(remote).name])
    monkeypatch.setattr(module.r2, "run_rclone", lambda *args: '{"count":1,"bytes":120000000}')

    resolved, identity = module._remote_rank_one_metadata(remote)

    assert resolved == remote
    assert identity == {"remote": remote, "sha256": digest, "bytes": 120_000_000}
