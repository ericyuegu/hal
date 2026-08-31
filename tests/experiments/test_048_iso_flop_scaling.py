"""Protocol contracts for O48 iso-FLOP scaling."""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "experiments" / "048_iso_flop_scaling.py"
    spec = importlib.util.spec_from_file_location("test_exp048", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


def test_grid_has_five_models_per_main_line_and_one_yolo() -> None:
    counts = Counter(endpoint.compute_budget_flops for endpoint in exp.POINTS.values() if not endpoint.yolo)
    assert len(exp.POINTS) == 26
    assert counts == {
        exp._C1E16: 5,
        exp._C3E16: 5,
        exp._C1E17: 5,
        exp._C3E17: 5,
        exp._C1E18: 5,
    }
    yolo = [endpoint for endpoint in exp.POINTS.values() if endpoint.yolo]
    assert yolo == [exp.POINTS["c3e18-d448-yolo"]]


@pytest.mark.parametrize("endpoint", exp.POINTS.values(), ids=lambda endpoint: endpoint.name)
def test_endpoint_is_nearest_update_and_uses_no_periodic_evaluation(endpoint) -> None:
    cfg = exp.endpoint_config(endpoint)
    report = exp.endpoint_report(endpoint)
    quantities = exp.model_quantities(endpoint.d_model)

    assert abs(report["actual_flops"] - endpoint.compute_budget_flops) <= quantities.flops_per_update / 2
    assert report["processed_positions"] == cfg.max_steps * cfg.batch_size * cfg.L_ctx
    assert report["processed_positions"] <= report["raw_corpus_loss_positions"]
    assert cfg.batch_size == 512
    assert cfg.muon_lr == exp._REFERENCE_CFG.muon_lr
    assert cfg.adam_lr == exp._REFERENCE_CFG.adam_lr
    assert cfg.val_every == cfg.eval_every == cfg.final_diag_n_matchups == 0
    assert cfg.final_eval_n_matchups == 96
    assert cfg.exec_horizon == 1
    assert exp.lr_schedule(cfg)(cfg.warmup_steps) == 1.0
    assert exp.lr_schedule(cfg)(cfg.max_steps * 2) == 1.0
    exp._o43.validate_config(cfg)
    assert exp.endpoint_for_config(cfg) == endpoint


def test_current_width_is_exact_o43_architecture() -> None:
    cfg = exp.endpoint_config(exp.POINTS["c1e17-d384"])
    reference = exp._o43.TrainConfig()
    fields = (
        "d_model",
        "n_layers",
        "n_heads",
        "temporal_d_model",
        "temporal_layers",
        "temporal_heads",
        "temporal_ff_dim",
        "group_head_dim",
        "codec_version",
        "head_offsets",
        "next_frame_loss_share",
    )
    assert {name: getattr(cfg, name) for name in fields} == {name: getattr(reference, name) for name in fields}


def test_weight_decay_law_is_anchored_to_o43() -> None:
    assert exp._scaled_weight_decay(
        exp._REFERENCE_CFG.adam_weight_decay,
        exp._REFERENCE_POSITIONS,
        exp._REFERENCE_PARAMETERS,
    ) == pytest.approx(exp._REFERENCE_CFG.adam_weight_decay)


def _write_test_corpus(path: Path) -> list:
    source = exp.streams.POLICY_WORLD_V7_SOURCES[0].name
    entries = [
        exp.CorpusEntry(source, 0, "a.mds", "r0", 6, "s0"),
        exp.CorpusEntry(source, 1, "a.mds", "r1", 11, "s1"),
        exp.CorpusEntry(source, 2, "b.mds", "r2", 16, "s2"),
        exp.CorpusEntry(source, 3, "c.mds", "r3", 21, "s3"),
    ]
    digest = exp.hashlib.sha256()
    for entry in entries:
        digest.update(
            f"{entry.source}\t{entry.row}\t{entry.shard}\t{entry.replay_id}\t{entry.frames}\t{entry.sha1}\n".encode()
        )
    header = {
        "schema_version": exp._CORPUS_SCHEMA,
        "protocol": exp._PROTOCOL,
        "source_names": [item.name for item in exp.streams.POLICY_WORLD_V7_SOURCES],
        "canonical_replays": len(entries),
        "canonical_loss_positions": sum(entry.loss_positions for entry in entries),
        "corpus_hash": digest.hexdigest(),
    }
    exp._write_corpus_index(path, header, entries)
    return entries


def test_dataset_tiers_are_nested_whole_shards(tmp_path: Path) -> None:
    path = tmp_path / "corpus.tsv.gz"
    entries = _write_test_corpus(path)
    low = exp.dataset_audit(path, 1)
    high = exp.dataset_audit(path, sum(entry.loss_positions for entry in entries) - 1)

    assert low.unique_loss_positions >= low.target_loss_positions
    assert high.unique_loss_positions >= high.target_loss_positions
    assert low.replay_ids < high.replay_ids
    for source, shards in low.selected_shards.items():
        assert shards <= high.selected_shards[source]


def test_tier_index_contains_only_selected_physical_shards(tmp_path: Path, monkeypatch) -> None:
    source = exp.StreamSource("source", "s3://bucket/root", tmp_path / "source")
    shards = [
        {"samples": 2, "raw_data": {"basename": "a.mds"}},
        {"samples": 3, "raw_data": {"basename": "b.mds"}},
    ]
    source_index = source.local_root / "train" / "index.json"
    source_index.parent.mkdir(parents=True)
    source_index.write_text(json.dumps({"version": 2, "shards": shards}))
    audit = exp.DatasetAudit(
        replay_ids=frozenset({"r0"}),
        selected_shards={source.name: frozenset({"b.mds"})},
        unique_replays=1,
        unique_loss_positions=10,
        target_loss_positions=10,
        episode_hash="1" * 64,
        corpus_hash="2" * 64,
        corpus_unique_replays=1,
        corpus_unique_loss_positions=10,
        source_replays={source.name: 1},
    )
    monkeypatch.setattr(exp.streams, "POLICY_WORLD_V7_SOURCES", (source,))
    monkeypatch.setattr(exp, "_TIER_CACHE_ROOT", tmp_path / "tiers")

    selected = exp._prepare_tier_sources(audit)
    derived = json.loads((selected[0].local_root / "train" / "index.json").read_text())
    assert [shard["raw_data"]["basename"] for shard in derived["shards"]] == ["b.mds"]
