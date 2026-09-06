import dataclasses
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from streaming import MDSWriter

import hal.scripts.rematerialize_policy_world_v8 as rematerialize
from hal.data.feature_stats import StatsAccumulator
from hal.data.feature_stats import dump_sufficient_stats
from hal.data.index import PlayerEntry
from hal.data.index import ReplayIndexEntry
from hal.data.index import Split
from hal.data.index import Stage3Annotation
from hal.data.mds import open_shard
from hal.data.mds import read_shard_index
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_schema import policy_replay_identity
from hal.data.policy_world_schema import POLICY_WORLD_FLOAT_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_MDS_COLUMNS
from hal.data.policy_world_schema import POLICY_WORLD_SCHEMA_VERSION
from hal.data.replay_stats import PlayerStats
from hal.data.replay_stats import ReplayStats
from hal.data.schema import SCHEMA_VERSION
from hal.scripts.filter import build_predicates
from hal.scripts.rematerialize_policy_world_v8 import CorpusJob
from hal.scripts.rematerialize_policy_world_v8 import DirectoryObjectStore
from hal.scripts.rematerialize_policy_world_v8 import StoredObject
from hal.scripts.rematerialize_policy_world_v8 import inspect_source
from hal.scripts.rematerialize_policy_world_v8 import publish_dataset
from hal.scripts.rematerialize_policy_world_v8 import rematerialize_corpus
from hal.scripts.rematerialize_policy_world_v8 import select_manifest

_LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "hal_rematerialize_policy_world_v8_modal",
    Path(__file__).parents[1] / "scripts" / "rematerialize_policy_world_v8_modal.py",
)
assert _LAUNCHER_SPEC is not None and _LAUNCHER_SPEC.loader is not None
_LAUNCHER = importlib.util.module_from_spec(_LAUNCHER_SPEC)
sys.modules[_LAUNCHER_SPEC.name] = _LAUNCHER
_LAUNCHER_SPEC.loader.exec_module(_LAUNCHER)
enumerate_corpus_jobs = _LAUNCHER.enumerate_corpus_jobs
WORKER_EPHEMERAL_DISK_MIB = _LAUNCHER.WORKER_EPHEMERAL_DISK_MIB
status_summary = _LAUNCHER._status_summary


def _players(*, human: bool = True) -> list[PlayerEntry]:
    player_type = "HUMAN" if human else "CPU"
    return [
        PlayerEntry(1, 1, 0, player_type, "ONE#1", "One"),
        PlayerEntry(2, 2, 0, "HUMAN", "TWO#2", "Two"),
    ]


def _stats(
    *,
    p1_damage: float = 101.0,
    p2_damage: float = 101.0,
    p1_stocks: int = 1,
    p2_stocks: int = 1,
) -> ReplayStats:
    deaths = (40.0, 50.0, 60.0)
    return ReplayStats(
        (
            PlayerStats(1, p1_damage, p2_damage, p1_stocks, 100, deaths),
            PlayerStats(2, p2_damage, p1_damage, p2_stocks, 100, deaths),
        )
    )


def _entry(
    path: str,
    split: Split,
    row: int,
    *,
    stats: ReplayStats | None = None,
    human: bool = True,
) -> ReplayIndexEntry:
    return ReplayIndexEntry(
        path=path,
        slp_version=(3, 18, 0),
        stage=31,
        players=_players(human=human),
        frame_count=2_000,
        timestamp=None,
        played_on="network",
        outcome=None,
        rank_filename=None,
        sha1=f"sha1-{path}",
        schema_version=SCHEMA_VERSION,
        annotation=Stage3Annotation(
            replay_uuid=row + 100,
            split=split,
            mds_row_idx=row,
            frame_count_actual=3,
            schema_version=SCHEMA_VERSION,
        ),
        stats=_stats() if stats is None else stats,
    )


def _sample(entry: ReplayIndexEntry, value: float) -> dict[str, object]:
    frames = 3
    sample: dict[str, object] = {}
    for name, encoding in POLICY_WORLD_MDS_COLUMNS.items():
        if encoding == "int":
            sample[name] = 1
        elif encoding == "str":
            sample[name] = policy_replay_identity(entry.path)
        else:
            dtype = np.dtype(encoding.removeprefix("ndarray:"))
            sample[name] = np.full(frames, value, dtype=dtype)
    sample.update(
        {
            "policy_schema_version": POLICY_SCHEMA_VERSION,
            "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
            "source_schema_version": SCHEMA_VERSION,
            "num_frames": frames,
            "replay_id": policy_replay_identity(entry.path),
        }
    )
    return sample


def _write_source(root: Path, entries: tuple[ReplayIndexEntry, ...]) -> DirectoryObjectStore:
    store = DirectoryObjectStore(root)
    source = root / "source"
    rows: dict[str, list[ReplayIndexEntry]] = {"train": [], "val": [], "test": []}
    for entry in entries:
        assert entry.annotation is not None
        rows[entry.annotation.split].append(entry)
    for split_entries in rows.values():
        split_entries.sort(key=lambda entry: entry.annotation.mds_row_idx if entry.annotation is not None else -1)

    stats = StatsAccumulator(POLICY_WORLD_FLOAT_COLUMNS)
    for split, split_entries in rows.items():
        writer = MDSWriter(
            out=str(source / split),
            columns=POLICY_WORLD_MDS_COLUMNS,
            compression="zstd",
            hashes=["md5", "sha256"],
            size_limit=5_000,
            exist_ok=False,
        )
        for ordinal, entry in enumerate(split_entries):
            sample = _sample(entry, float(ordinal + 1))
            writer.write(sample)
            if split == "train":
                for name in POLICY_WORLD_FLOAT_COLUMNS:
                    stats.update(name, np.asarray(sample[name]))
        writer.finish()

    manifest = "".join(json.dumps(entry.to_dict()) + "\n" for entry in entries)
    (source / "manifest.jsonl").write_text(manifest)
    counts = {split: len(split_entries) for split, split_entries in rows.items()}
    (source / "projection.json").write_text(
        json.dumps(
            {
                "policy_world_schema_version": POLICY_WORLD_SCHEMA_VERSION,
                "source_schema_version": SCHEMA_VERSION,
                "rows": counts,
                "failures": 0,
                "columns": POLICY_WORLD_MDS_COLUMNS,
            }
        )
    )
    (source / "failures.materialize.jsonl").write_text("")
    dump_sufficient_stats(
        source / "stats.json",
        stats.to_sufficient(),
        split="train",
        mds_schema_version=SCHEMA_VERSION,
    )
    (source / "_SUCCESS").write_text(json.dumps({"rows": counts}))
    return store


def _job(name: str = "test-policy-world-v8") -> CorpusJob:
    return CorpusJob(
        name=name,
        source_prefix="source",
        staging_prefix=f"staging/run/{name}",
        final_prefix=f"final/{name}",
        run_id="run",
        git_sha="a" * 40,
    )


def _source_entries() -> tuple[ReplayIndexEntry, ...]:
    train = [
        _entry("train-keep-0.slp", "train", 0),
        _entry("train-drop.slp", "train", 1, stats=_stats(p1_damage=50.0, p1_stocks=2)),
        _entry("train-keep-2.slp", "train", 2),
    ]
    val = [_entry("val-keep.slp", "val", 0)]
    test = [_entry("test-drop.slp", "test", 0, human=False)]
    return (val[0], train[2], test[0], train[0], train[1])


def test_v8_predicate_boundaries_and_inferred_stocks() -> None:
    predicates = dict(
        build_predicates(
            human_players_only=True,
            min_damage_dealt_per_player_exclusive=50.0,
            starting_stocks=4,
        )
    )
    boundary = _entry("boundary.slp", "train", 0, stats=_stats(p1_damage=50.0))
    above = _entry("above.slp", "train", 0, stats=_stats(p1_damage=np.nextafter(50.0, np.inf)))
    wrong_stocks = _entry("stocks.slp", "train", 0, stats=_stats(p1_stocks=2))

    assert not predicates["damage_dealt_per_player>50.0"](boundary)
    assert predicates["damage_dealt_per_player>50.0"](above)
    assert predicates["starting_stocks=4"](above)
    assert not predicates["starting_stocks=4"](wrong_stocks)
    assert predicates["human_players_only"](above)
    assert not predicates["human_players_only"](_entry("cpu.slp", "train", 0, human=False))


def test_v8_policy_matches_the_published_selection_contract() -> None:
    assert rematerialize._policy_dict() == {
        "characters": [],
        "cheap_death_pct": 10.0,
        "completed_only": False,
        "human_players_only": True,
        "max_cheap_deaths": 2,
        "max_frames": None,
        "min_damage_dealt": 100.0,
        "min_damage_dealt_per_player_exclusive": 50.0,
        "min_damage_taken": 100.0,
        "min_death_count": 3,
        "min_frames": 1_500,
        "min_inputs": None,
        "min_stocks_remaining": None,
        "ranks": [],
        "stages": [
            "FINAL_DESTINATION",
            "BATTLEFIELD",
            "POKEMON_STADIUM",
            "DREAMLAND",
            "FOUNTAIN_OF_DREAMS",
            "YOSHIS_STORY",
        ],
        "starting_stocks": 4,
        "stock_zero_only": False,
    }


def test_statistics_audit_allows_only_reduction_order_roundoff() -> None:
    expected = {"x": {"count": 100, "mean": 1.0, "m2": 1_000.0, "min": -2.0, "max": 3.0}}
    rounded = {"x": {"count": 100, "mean": 1.0 + 1e-7, "m2": 1_000.0 + 5e-6, "min": -2.0, "max": 3.0}}

    rematerialize._validate_recomputed_stats(rounded, expected, "stats")
    rounded["x"]["mean"] = 1.001
    with pytest.raises(ValueError, match=r"x\.mean=.* differs"):
        rematerialize._validate_recomputed_stats(rounded, expected, "stats")
    rounded["x"]["mean"] = 1.0
    rounded["x"]["count"] = 99
    with pytest.raises(ValueError, match=r"x\.count differs"):
        rematerialize._validate_recomputed_stats(rounded, expected, "stats")


def test_selection_records_overlapping_rejection_reasons() -> None:
    entry = _entry("overlap.slp", "train", 0, stats=_stats(p1_damage=50.0, p1_stocks=2), human=False)
    selection = select_manifest((entry,), {"train": 1, "val": 0, "test": 0}, where="manifest")

    assert selection.kept_rows["train"] == ()
    assert selection.rejections[0].reasons == (
        "human_players_only",
        "damage_dealt_per_player>50.0",
        "starting_stocks=4",
    )


def test_selection_rejects_unexpected_v7_policy_drift() -> None:
    entry = _entry("v7-drift.slp", "train", 0, stats=_stats(p1_damage=60.0, p2_damage=60.0))

    with pytest.raises(ValueError, match="source rows fail the immutable v7 policy"):
        select_manifest((entry,), {"train": 1, "val": 0, "test": 0}, where="manifest")


@pytest.mark.parametrize("defect", ["missing-stats", "invalid-stats", "duplicate-row", "schema"])
def test_selection_rejects_malformed_manifests(defect: str) -> None:
    first = _entry("one.slp", "train", 0)
    second = _entry("two.slp", "train", 1)
    if defect == "missing-stats":
        first = dataclasses.replace(first, stats=None)
    elif defect == "invalid-stats":
        first = dataclasses.replace(first, stats=_stats(p1_damage=float("nan")))
    elif defect == "duplicate-row":
        assert second.annotation is not None
        second = dataclasses.replace(second, annotation=dataclasses.replace(second.annotation, mds_row_idx=0))
    else:
        first = dataclasses.replace(first, schema_version=SCHEMA_VERSION - 1)

    with pytest.raises(ValueError):
        select_manifest((first, second), {"train": 2, "val": 0, "test": 0}, where="manifest")


def test_source_materialization_failures_must_match_the_ledger(tmp_path: Path) -> None:
    store = _write_source(tmp_path / "objects", _source_entries())
    job = _job()
    projection_key = f"{job.source_prefix}/projection.json"
    projection = json.loads(store.read_bytes(projection_key))
    projection["failures"] = 1
    store.put_bytes(projection_key, json.dumps(projection).encode())
    store.put_bytes(
        f"{job.source_prefix}/failures.materialize.jsonl",
        b'{"path":"failed.slp","phase":"materialize","error":"bad"}\n',
    )

    assert inspect_source(store, job).materialization_failures == 1
    projection["failures"] = 2
    store.put_bytes(projection_key, json.dumps(projection).encode())
    with pytest.raises(ValueError, match="failure count differs"):
        inspect_source(store, job)


def test_ranked_one_uses_only_sha_pinned_supplemental_statistics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_entries = _source_entries()
    populated = tuple(dataclasses.replace(entry, schema_version=SCHEMA_VERSION - 1) for entry in source_entries)
    missing = tuple(dataclasses.replace(entry, stats=None) for entry in source_entries)
    store = _write_source(tmp_path / "objects", missing)
    metadata = "".join(json.dumps(entry.to_dict()) + "\n" for entry in populated).encode()
    metadata_key = "metadata/ranked-1-index.jsonl"
    store.put_bytes(metadata_key, metadata)
    monkeypatch.setattr(rematerialize, "RANKED_ONE_METADATA_KEY", metadata_key)
    monkeypatch.setattr(rematerialize, "RANKED_ONE_METADATA_SHA256", hashlib.sha256(metadata).hexdigest())
    job = CorpusJob(
        name=rematerialize.RANKED_ONE_CORPUS,
        source_prefix="source",
        staging_prefix=f"staging/run/{rematerialize.RANKED_ONE_CORPUS}",
        final_prefix=f"final/{rematerialize.RANKED_ONE_CORPUS}",
        run_id="run",
        git_sha="a" * 40,
    )

    source = inspect_source(store, job)

    assert all(entry.stats is not None for entry in source.selection.entries)
    assert source.supplemental_metadata is not None
    assert source.supplemental_metadata["sha256"] == hashlib.sha256(metadata).hexdigest()


def test_rematerialization_preserves_rows_reindexes_manifest_and_recomputes_stats(tmp_path: Path) -> None:
    entries = _source_entries()
    store = _write_source(tmp_path / "objects", entries)
    job = _job()

    result = rematerialize_corpus(store, job, tmp_path / "scratch", staging_only=True)

    assert result["state"] == "staged"
    assert result["rows"] == {"train": 2, "val": 1, "test": 0}
    assert result["rejections"] == 2
    manifest = [
        ReplayIndexEntry.from_dict(json.loads(line))
        for line in store.read_bytes(f"{job.staging_prefix}/manifest.jsonl").decode().splitlines()
    ]
    assert [entry.path for entry in manifest] == ["val-keep.slp", "train-keep-2.slp", "train-keep-0.slp"]
    annotations = {entry.path: entry.annotation for entry in manifest}
    first_annotation = annotations["train-keep-0.slp"]
    second_annotation = annotations["train-keep-2.slp"]
    assert first_annotation is not None and first_annotation.mds_row_idx == 0
    assert second_annotation is not None and second_annotation.mds_row_idx == 1

    staging = tmp_path / "objects" / job.staging_prefix
    train_shards = read_shard_index(staging, "train")
    assert len(train_shards) == 1
    decompress = tmp_path / "decompress"
    decompress.mkdir()
    with open_shard(staging, "train", train_shards[0], decompress) as reader:
        output_rows = list(reader)
    assert [row["replay_id"] for row in output_rows] == [
        policy_replay_identity("train-keep-0.slp"),
        policy_replay_identity("train-keep-2.slp"),
    ]
    assert np.array_equal(output_rows[0]["p1_position_x"], np.full(3, 1.0, dtype=np.float32))
    assert np.array_equal(output_rows[1]["p1_position_x"], np.full(3, 3.0, dtype=np.float32))

    stats = json.loads(store.read_bytes(f"{job.staging_prefix}/stats.json"))
    position = stats["sufficient"]["p1_position_x"]
    assert position["count"] == 6
    assert position["mean"] == 2.0


def test_retry_clears_only_exact_unfinished_staging_prefix(tmp_path: Path) -> None:
    store = _write_source(tmp_path / "objects", _source_entries())
    job = _job()
    store.put_bytes(f"{job.staging_prefix}/partial", b"partial")
    sibling = f"staging/old-run/{job.name}/partial"
    store.put_bytes(sibling, b"keep")

    rematerialize_corpus(store, job, tmp_path / "scratch", staging_only=True)

    assert store.head(f"{job.staging_prefix}/partial") is None
    assert store.read_bytes(sibling) == b"keep"
    assert store.head(f"{job.staging_prefix}/_STAGED") is not None


class _RecordingStore(DirectoryObjectStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.events: list[tuple[str, str]] = []

    def put_bytes(self, key: str, data: bytes) -> StoredObject:
        self.events.append(("put", key))
        return super().put_bytes(key, data)

    def copy(self, source: str, destination: str) -> StoredObject:
        self.events.append(("copy", destination))
        return super().copy(source, destination)


def test_publication_resumes_matching_partial_copy_and_writes_success_last(tmp_path: Path) -> None:
    base_store = _write_source(tmp_path / "objects", _source_entries())
    job = _job()
    rematerialize_corpus(base_store, job, tmp_path / "build", staging_only=True)
    store = _RecordingStore(tmp_path / "objects")
    source = inspect_source(store, job)
    staged_names = sorted(
        key.removeprefix(job.staging_prefix + "/")
        for key in store.list(job.staging_prefix + "/")
        if not key.endswith("/_STAGED")
    )
    first = staged_names[0]
    store.copy(f"{job.staging_prefix}/{first}", f"{job.final_prefix}/{first}")
    store.events.clear()

    publish_dataset(store, job, source, tmp_path / "publish")

    assert ("copy", f"{job.final_prefix}/{first}") not in store.events
    assert store.events[-1] == ("put", f"{job.final_prefix}/_SUCCESS")
    assert store.head(f"{job.final_prefix}/_SUCCESS") is not None


def test_publication_rejects_a_different_partial_object(tmp_path: Path) -> None:
    store = _write_source(tmp_path / "objects", _source_entries())
    job = _job()
    rematerialize_corpus(store, job, tmp_path / "build", staging_only=True)
    store.put_bytes(f"{job.final_prefix}/manifest.jsonl", b"different")
    source = inspect_source(store, job)

    with pytest.raises(FileExistsError, match="differs from staging"):
        publish_dataset(store, job, source, tmp_path / "publish")


def test_exact_interrupted_run_metadata_remains_auditable_and_immutable(tmp_path: Path) -> None:
    store = _write_source(tmp_path / "objects", _source_entries())
    job = _job("professional-rapm-policy-world-v8")
    rematerialize_corpus(store, job, tmp_path / "build", staging_only=True)
    source = inspect_source(store, job)
    publish_dataset(store, job, source, tmp_path / "publish")

    input_total = sum(source.selection.input_counts.values())
    retained_total = sum(source.selection.retained_counts.values())
    draft_selection = {
        "schema_version": 1,
        "policy": rematerialize._draft_policy_dict(),
        "corpus": job.name,
        "source": rematerialize._draft_source_identity(source),
        "run_id": rematerialize.DRAFT_PUBLICATION_RUN_ID,
        "git_sha": rematerialize.DRAFT_PUBLICATION_GIT_SHA,
        "counts": {
            "input": source.selection.input_counts,
            "retained": source.selection.retained_counts,
            "rejected": {
                split: source.selection.input_counts[split] - source.selection.retained_counts[split]
                for split in rematerialize.SPLITS
            },
            "input_total": input_total,
            "retained_total": retained_total,
            "rejected_total": input_total - retained_total,
        },
    }
    store.put_bytes(f"{job.final_prefix}/selection.json", json.dumps(draft_selection).encode())

    projection_key = f"{job.final_prefix}/projection.json"
    projection = json.loads(store.read_bytes(projection_key))
    projection["source"] = rematerialize._draft_source_identity(source)
    del projection["source_rows"]
    del projection["rejections"]
    store.put_bytes(projection_key, json.dumps(projection).encode())

    canonical_to_draft = {value: key for key, value in rematerialize._DRAFT_REJECTION_REASONS.items()}
    draft_rejections = [
        {
            "path": rejection.path,
            "replay_id": rejection.replay_id,
            "split": rejection.split,
            "source_mds_row_idx": rejection.source_mds_row_idx,
            "reasons": [canonical_to_draft[reason] for reason in rejection.reasons],
        }
        for rejection in source.selection.rejections
    ]
    store.put_bytes(
        f"{job.final_prefix}/rejections.jsonl",
        "".join(json.dumps(row) + "\n" for row in draft_rejections).encode(),
    )

    success_key = f"{job.final_prefix}/_SUCCESS"
    success = json.loads(store.read_bytes(success_key))
    success["policy"] = rematerialize._draft_policy_dict()
    success["source"] = rematerialize._draft_source_identity(source)
    success["run_id"] = rematerialize.DRAFT_PUBLICATION_RUN_ID
    success["git_sha"] = rematerialize.DRAFT_PUBLICATION_GIT_SHA
    store.put_bytes(success_key, json.dumps(success).encode())

    result = rematerialize_corpus(store, job, tmp_path / "validate")

    assert result["state"] == "validated-existing"
    assert json.loads(store.read_bytes(success_key))["policy"] == rematerialize._draft_policy_dict()


def test_corpus_job_enumeration_is_deterministic() -> None:
    jobs = enumerate_corpus_jobs("run-123", "b" * 40)

    assert len(jobs) == len({job.name for job in jobs}) == 44
    assert jobs[0].name == "ranked-anonymized-1-policy-world-v8"
    assert jobs[-1].name == "professional-zain-policy-world-v8"
    assert jobs[0].source_prefix == "processed/ranked-anonymized-1/mds-policy-world-v7"
    assert jobs[0].final_prefix == "processed/ranked-anonymized-1/mds-policy-world-v8"
    assert jobs[0].staging_prefix == ("processed/_staging/policy-world-v8/run-123/ranked-anonymized-1-policy-world-v8")
    assert jobs == enumerate_corpus_jobs("run-123", "b" * 40)
    assert WORKER_EPHEMERAL_DISK_MIB == 512 * 1024


def test_status_summary_reports_only_completed_audits() -> None:
    audit = {
        "retained": 3,
        "rows": {"train": 2, "val": 1, "test": 0},
        "rejections": 1,
        "train_frames": 20,
    }
    summary = status_summary(
        [
            {"corpus": "done", "published": True, "record": {"state": "published", "audit": audit}},
            {"corpus": "failed", "published": False, "record": {"state": "failed"}},
            {"corpus": "missing", "published": False, "record": None},
        ]
    )

    assert summary == {
        "corpora": 3,
        "published": 1,
        "states": {"failed": 1, "no-record": 1, "published": 1},
        "audited": 1,
        "accepted": False,
        "retained": 3,
        "train_replays": 2,
        "rejections": 1,
        "train_frames": 20,
        "sources": {"done": audit},
    }


def _completed_status_results(retained: int) -> list[dict[str, object]]:
    return [
        {
            "corpus": f"corpus-{index}",
            "published": True,
            "record": {
                "state": "published",
                "audit": {
                    "retained": retained if index == 0 else 0,
                    "rows": {"train": _LAUNCHER.EXPECTED_TRAIN_REPLAYS if index == 0 else 0},
                    "rejections": _LAUNCHER.EXPECTED_REJECTIONS if index == 0 else 0,
                    "train_frames": 1,
                },
            },
        }
        for index in range(_LAUNCHER.EXPECTED_CORPORA)
    ]


def test_status_summary_enforces_complete_corpus_acceptance_totals() -> None:
    results = _completed_status_results(_LAUNCHER.EXPECTED_RETAINED)

    assert status_summary(results)["accepted"] is True
    with pytest.raises(ValueError, match="complete v8 corpus totals differ"):
        status_summary(_completed_status_results(_LAUNCHER.EXPECTED_RETAINED - 1))
