"""Focused contracts for experiment 034's rank-only loss treatment."""

import gzip
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from dataclasses import fields
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from streaming import MDSWriter
from streaming import StreamingDataset
from torch.utils.data import DataLoader

import hal.training.replay_reservoir as replay_reservoir
from hal.data.policy_schema import POLICY_MDS_COLUMNS
from hal.data.policy_schema import POLICY_SCHEMA_VERSION
from hal.data.policy_schema import policy_replay_identity
from hal.data.schema import Rank
from hal.training.features import A_DIM
from hal.training.features import Context
from hal.training.features import FeatureProjection
from hal.training.features import TrainBatch
from hal.training.replay_reservoir import PolicyReplayPackDataset

_PATH = Path(__file__).resolve().parents[2] / "experiments" / "034_rank_weighted_bc.py"
_SPEC = importlib.util.spec_from_file_location("test_exp034", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


def _cfg(**overrides) -> exp.TrainConfig:
    values = {
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 4,
        "temporal_d_model": 32,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "temporal_ff_dim": 64,
        "group_head_dim": 64,
        "batch_size": 4,
        "grad_accum_steps": 2,
        "reservoir_capacity": 8,
        "warmup_steps": 1,
        "max_steps": 2,
        "compile_trunk": False,
        "compile_temporal": False,
        "num_workers": 0,
        "push_to_r2": False,
        "inference_mode": "eager",
    }
    return exp.TrainConfig(**{**values, **overrides})


def _train_batch(cfg: exp.TrainConfig, pads: list[int], replay_prefix: str = "r") -> TrainBatch:
    batch_size = len(pads)
    context = Context(
        features={"feature": torch.zeros(batch_size, cfg.L_ctx)},
        ctx_pad=torch.tensor(pads, dtype=torch.int64),
    )
    replay_ids = tuple(f"{replay_prefix}{index}" for index in range(batch_size))
    target = torch.zeros(batch_size, cfg.sample_chunk_length, A_DIM)
    return TrainBatch(context=context, target=target, replay_ids=replay_ids)


def _rank_batch(
    cfg: exp.TrainConfig,
    ranks: list[Rank],
    pads: list[int],
    replay_prefix: str = "r",
) -> exp.RankBatch:
    weights = torch.tensor([cfg.rank_weights[int(rank) - 1] for rank in ranks], dtype=torch.float32)
    return exp.RankBatch(
        batch=_train_batch(cfg, pads, replay_prefix),
        rank=torch.tensor([int(rank) for rank in ranks], dtype=torch.uint8),
        rank_weight=weights,
    )


def _write_sidecar(path: Path, rows: list[list[object]]) -> str:
    counts = {
        str(int(rank)): sum(int(value) == int(rank) for row in rows for value in row[1:])
        for rank in (Rank.PLATINUM, Rank.DIAMOND, Rank.MASTER)
    }
    header = {
        "rank_sidecar_schema_version": 1,
        "source_manifest_sha256": exp._RANK_MANIFEST_SHA256,
        "rows": len(rows),
        "player_rank_counts": counts,
    }
    text = "\n".join(json.dumps(row, separators=(",", ":")) for row in (header, *rows)) + "\n"
    payload = gzip.compress(text.encode(), mtime=0)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_defaults_are_the_exact_eligible_026_recipe_plus_rank_metadata() -> None:
    cfg = exp.TrainConfig()
    reference = exp._TrainConfig026()
    expected = asdict(reference)
    expected.update(batch_size=512, cache_limit_gb=160, eval_max_parallel=32)
    base_names = {item.name for item in fields(exp._TrainConfig026)}

    assert {name: getattr(cfg, name) for name in base_names} == expected
    assert cfg.rank_weights == (1.0, 2.0, 4.0)
    assert cfg.max_steps == 2**14
    assert cfg.data_root == "data/processed/ranked-anonymized-1/mds-policy-v7"
    assert exp.model_tag(cfg).endswith("-rank1-2-4")
    exp.validate_config(cfg)


def test_production_and_smoke_reject_scientific_changes_but_smoke_allows_runtime_knobs() -> None:
    with pytest.raises(ValueError, match="production 034 config changed frozen scientific fields"):
        exp.validate_config(exp.TrainConfig(batch_size=256))

    smoke = exp.TrainConfig(max_steps=100, batch_size=512, grad_accum_steps=8)
    exp.validate_config(smoke)
    with pytest.raises(ValueError, match="smoke 034 config changed frozen scientific fields"):
        exp.validate_config(replace(smoke, seed=1))
    with pytest.raises(ValueError, match="smoke 034 config changed frozen scientific fields"):
        exp.validate_config(replace(smoke, rank_weights=(1.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="cannot exceed"):
        exp.validate_config(exp.TrainConfig(max_steps=2**14 + 1))


def test_evaluation_override_allows_runtime_fields_only() -> None:
    checkpoint_cfg = exp.TrainConfig()
    runtime_cfg = replace(checkpoint_cfg, inference_mode="eager", eval_max_parallel=2)
    exp._validate_evaluation_override(runtime_cfg, checkpoint_cfg)

    with pytest.raises(ValueError, match="checkpoint-scientific fields"):
        exp._validate_evaluation_override(replace(runtime_cfg, muon_lr=0.01), checkpoint_cfg)


def test_rank_config_round_trip_and_model_shape_are_026_identical() -> None:
    cfg = _cfg(rank_weights=(1.0, 1.5, 3.0))
    restored = exp.config_from_state(asdict(cfg))
    assert restored.rank_weights == cfg.rank_weights
    assert restored.experiment_id == exp._EXPERIMENT_ID
    shared = {item.name: getattr(cfg, item.name) for item in fields(exp._TrainConfig026)}
    base_cfg = exp._TrainConfig026(**shared)
    treatment_model = exp.GPT(cfg)
    control_model = exp.GPT(base_cfg)
    assert treatment_model.state_dict().keys() == control_model.state_dict().keys()
    assert {name: tuple(value.shape) for name, value in treatment_model.state_dict().items()} == {
        name: tuple(value.shape) for name, value in control_model.state_dict().items()
    }


def test_checkpoint_identity_rejects_026_and_wrong_experiment_configs() -> None:
    with pytest.raises(ValueError, match="not experiment 034"):
        exp.config_from_state(asdict(exp._TrainConfig026()))

    wrong = asdict(exp.TrainConfig())
    wrong["experiment_id"] = "026_temporal_mtp"
    with pytest.raises(ValueError, match="checkpoint experiment_id"):
        exp.config_from_state(wrong)

    assert exp.base.config_from_state is exp.config_from_state
    assert exp.base.eval_checkpoint is exp.eval_checkpoint


def test_rank_sidecar_is_hash_checked_and_selects_the_sampled_ego_port(tmp_path: Path) -> None:
    first = "a" * 32
    second = "b" * 32
    path = tmp_path / "ranks.jsonl.gz"
    digest = _write_sidecar(
        path,
        [
            [first, int(Rank.MASTER), int(Rank.PLATINUM)],
            [second, int(Rank.DIAMOND), int(Rank.MASTER)],
        ],
    )
    cfg = _cfg(
        rank_sidecar_local=str(path),
        rank_sidecar_sha256=digest,
        rank_sidecar_rows=2,
        rank_player_counts=(1, 1, 2),
    )
    lookup = exp.load_rank_lookup(cfg)
    p1_window: dict[str, np.ndarray] = {}
    p2_window: dict[str, np.ndarray] = {}
    lookup(first, "p1", p1_window)
    lookup(first, "p2", p2_window)
    assert int(p1_window["ego_rank"]) == int(Rank.MASTER)
    assert int(p2_window["ego_rank"]) == int(Rank.PLATINUM)

    bad = _cfg(
        rank_sidecar_local=str(path),
        rank_sidecar_sha256="0" * 64,
        rank_sidecar_rows=2,
        rank_player_counts=(1, 1, 2),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        exp.load_rank_lookup(bad)


def test_rank_sidecar_rejects_source_manifest_or_count_provenance_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "ranks.jsonl.gz"
    digest = _write_sidecar(
        path,
        [["a" * 32, int(Rank.PLATINUM), int(Rank.DIAMOND)]],
    )
    common = {
        "rank_sidecar_local": str(path),
        "rank_sidecar_sha256": digest,
        "rank_sidecar_rows": 1,
    }
    with pytest.raises(ValueError, match="source manifest SHA-256"):
        exp.load_rank_lookup(
            _cfg(
                **common,
                rank_manifest_sha256="0" * 64,
                rank_player_counts=(1, 1, 0),
            )
        )
    with pytest.raises(ValueError, match="player counts"):
        exp.load_rank_lookup(_cfg(**common, rank_player_counts=(1, 0, 1)))


def test_missing_local_sidecar_is_derived_read_only_from_frozen_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {
            "path": f"archive://ranked/{name}.slp",
            "players": [
                {"port": 1, "name": first},
                {"port": 2, "name": second},
            ],
            "annotation": {"split": "train", "mds_row_idx": index},
        }
        for index, (name, first, second) in enumerate(
            (
                ("one", "Master Player", "Platinum Player"),
                ("two", "Diamond Player", "Master Player"),
            )
        )
    ]
    payload = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    manifest.write_bytes(payload)
    cfg = _cfg(
        rank_sidecar_local=str(tmp_path / "missing.jsonl.gz"),
        rank_sidecar_rows=2,
        rank_manifest_local=str(manifest),
        rank_manifest_remote="unused",
        rank_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        rank_player_counts=(1, 1, 2),
    )

    lookup = exp.load_rank_lookup(cfg)

    replay_id = policy_replay_identity(rows[0]["path"])
    window: dict[str, np.ndarray] = {}
    lookup(replay_id, "p1", window)
    assert int(window["ego_rank"]) == int(Rank.MASTER)


def test_rank_metadata_does_not_change_window_sampling_or_policy_columns(monkeypatch) -> None:
    frames = 40
    replay_id = "c" * 32
    compact = {"replay_id": replay_id, "source_schema_version": 7, "num_frames": frames}

    def decode_slices(source, ranges):
        del source
        outputs = []
        for start, stop in ranges:
            frame = np.arange(start, stop, dtype=np.int32)
            outputs.append(
                {
                    "schema_version": 7,
                    "frame": frame,
                    "p1_value": np.full(len(frame), 1, dtype=np.float32),
                    "p2_value": np.full(len(frame), 2, dtype=np.float32),
                }
            )
        return tuple(outputs)

    monkeypatch.setattr(replay_reservoir, "decode_policy_replay_slices", decode_slices)
    projection = FeatureProjection(frozenset({"ego_value"}), derive_spatial=False)
    common = {
        "L_ctx": 8,
        "L_chunk": 4,
        "seed": 13,
        "windows_per_replay": 3,
        "schema_version": 7,
        "projection": projection,
    }
    control = next(iter(PolicyReplayPackDataset([compact], **common)))
    lookup = exp.RankLookup({replay_id: (int(Rank.MASTER), int(Rank.PLATINUM))})
    treatment = next(iter(PolicyReplayPackDataset([compact], window_transform=lookup, **common)))

    assert treatment.replay_id == control.replay_id
    assert len(treatment.windows) == len(control.windows)
    for plain, ranked in zip(control.windows, treatment.windows, strict=True):
        assert ranked.keys() == plain.keys() | {"ego_rank"}
        for name in plain:
            np.testing.assert_array_equal(ranked[name], plain[name])
        expected = Rank.MASTER if float(ranked["ego_value"][-1]) == 1 else Rank.PLATINUM
        assert int(ranked["ego_rank"]) == int(expected)


def test_rank_window_transform_survives_two_dataloader_workers_and_prefetch(tmp_path: Path) -> None:
    frames = 12
    replay_ids = [f"{index:032x}" for index in range(4)]
    with MDSWriter(out=str(tmp_path), columns=POLICY_MDS_COLUMNS, compression="zstd") as writer:
        for replay_id in replay_ids:
            sample: dict[str, object] = {
                "policy_schema_version": POLICY_SCHEMA_VERSION,
                "source_schema_version": 7,
                "replay_id": replay_id,
                "num_frames": frames,
                "stage": 2,
                "p1_character": 1,
                "p2_character": 22,
                "p1_nana_present": 0,
                "p2_nana_present": 0,
            }
            for name, encoding in POLICY_MDS_COLUMNS.items():
                if name in sample:
                    continue
                dtype = np.dtype(encoding.removeprefix("ndarray:"))
                sample[name] = np.zeros(1 if "nana" in name else frames, dtype=dtype)
            writer.write(sample)

    source = StreamingDataset(local=str(tmp_path), batch_size=1, shuffle=False)
    lookup = exp.RankLookup({replay_id: (int(Rank.PLATINUM), int(Rank.MASTER)) for replay_id in replay_ids})
    packs = PolicyReplayPackDataset(
        source,
        L_ctx=4,
        L_chunk=2,
        seed=17,
        windows_per_replay=1,
        schema_version=7,
        projection=None,
        window_transform=lookup,
    )
    loader = DataLoader(
        packs,
        batch_size=None,
        num_workers=2,
        collate_fn=replay_reservoir._identity,
        prefetch_factor=2,
    )
    observed = list(loader)

    assert {pack.replay_id for pack in observed} == set(replay_ids)
    assert len(observed) == len(replay_ids)
    for pack in observed:
        assert len(pack.windows) == 1
        assert int(pack.windows[0]["ego_rank"]) in (int(Rank.PLATINUM), int(Rank.MASTER))


def test_collate_maps_tiers_and_rejects_unknown_or_professional_rows() -> None:
    cfg = _cfg(batch_size=3, grad_accum_steps=1, reservoir_capacity=6)
    batch = _train_batch(cfg, [0, 1, 2])
    windows = [
        {"ego_rank": np.asarray(int(rank), dtype=np.uint8)} for rank in (Rank.PLATINUM, Rank.DIAMOND, Rank.MASTER)
    ]
    ranked = exp.collate_rank_batch(windows, batch, rank_weights=cfg.rank_weights)
    assert ranked.rank.tolist() == [1, 2, 3]
    torch.testing.assert_close(ranked.rank_weight, torch.tensor([1.0, 2.0, 4.0]))
    assert not [name for name in ranked.context.features if "rank" in name]

    for invalid in (Rank.UNKNOWN, Rank.PRO):
        with pytest.raises(ValueError, match="unsupported ego rank"):
            exp.collate_rank_batch(
                [{"ego_rank": np.asarray(int(invalid), dtype=np.uint8)}],
                _train_batch(_cfg(batch_size=1, grad_accum_steps=1), [0]),
                rank_weights=cfg.rank_weights,
            )


def test_rank_weights_align_with_the_same_valid_prefix_mask_as_nll() -> None:
    cfg = _cfg()
    batch = _rank_batch(
        cfg,
        [Rank.PLATINUM, Rank.MASTER, Rank.DIAMOND, Rank.MASTER],
        [0, 1, 2, 3],
    )
    valid = torch.arange(cfg.L_ctx)[None, :] >= batch.context.ctx_pad[:, None]
    weights = batch.valid_rank_weights(valid)
    torch.testing.assert_close(weights, torch.tensor([1, 1, 1, 1, 4, 4, 4, 2, 2, 4], dtype=torch.float32))


def test_whole_step_normalizer_counts_valid_prefixes_and_passes_ess_gate() -> None:
    cfg = _cfg()
    first = _rank_batch(cfg, [Rank.PLATINUM, Rank.MASTER], [0, 1], "a")
    second = _rank_batch(cfg, [Rank.DIAMOND, Rank.MASTER], [2, 3], "b")
    summary = exp.summarize_rank_step([first, second], cfg)

    assert summary.valid_prefixes == 10
    assert summary.weight_sum == 24
    assert summary.metrics["rank/ess_fraction"] == pytest.approx(576 / 760)
    assert summary.metrics["rank/example_fraction_master"] == 0.5
    assert summary.metrics["rank/gradient_mass_fraction_master"] == pytest.approx(16 / 24)


def test_uniform_rank_objective_and_gradient_match_026() -> None:
    values = torch.randn(7, 10, exp.N_GROUPS, generator=torch.Generator().manual_seed(5))
    plain_nll = values.clone().requires_grad_()
    ranked_nll = values.clone().requires_grad_()
    parts = exp.ActionLoss(nll=plain_nll, targets=torch.empty(0))
    plain = exp.objective(parts, aux_loss_weight=0.7)
    ranked = exp.rank_weighted_objective(
        ranked_nll,
        torch.ones(7),
        weight_sum=7,
        aux_loss_weight=0.7,
    )
    plain.backward()
    ranked.backward()

    torch.testing.assert_close(ranked, plain, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ranked_nll.grad, plain_nll.grad, rtol=1e-6, atol=1e-7)


def test_accumulated_weighted_objective_matches_one_concatenated_batch() -> None:
    values = torch.randn(9, 10, exp.N_GROUPS, generator=torch.Generator().manual_seed(9))
    weights = torch.tensor([1, 4, 2, 1, 4, 4, 2, 1, 2], dtype=torch.float32)
    weight_sum = float(weights.sum())
    full_nll = values.clone().requires_grad_()
    full = exp.rank_weighted_objective(
        full_nll,
        weights,
        weight_sum=weight_sum,
        aux_loss_weight=1.0,
    )
    full.backward()

    pieces = [
        values[:2].clone().requires_grad_(),
        values[2:6].clone().requires_grad_(),
        values[6:].clone().requires_grad_(),
    ]
    split = sum(
        exp.rank_weighted_objective(
            piece,
            piece_weights,
            weight_sum=weight_sum,
            aux_loss_weight=1.0,
        )
        for piece, piece_weights in zip(pieces, (weights[:2], weights[2:6], weights[6:]), strict=True)
    )
    split.backward()

    torch.testing.assert_close(split, full, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(torch.cat([piece.grad for piece in pieces]), full_nll.grad, rtol=1e-6, atol=1e-7)


def test_rank_batch_cpu_transfer_preserves_metadata_and_replay_ids() -> None:
    cfg = _cfg(batch_size=2, grad_accum_steps=1)
    batch = _rank_batch(cfg, [Rank.MASTER, Rank.PLATINUM], [0, 1])
    moved = batch.to("cpu")
    assert moved.replay_ids == batch.replay_ids
    torch.testing.assert_close(moved.rank, batch.rank)
    torch.testing.assert_close(moved.rank_weight, batch.rank_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for asynchronous staging")
def test_async_cuda_staging_preserves_two_prefetched_rank_batches_under_allocator_pressure() -> None:
    cfg = _cfg(batch_size=4, grad_accum_steps=2)
    batches = [
        _rank_batch(cfg, [Rank.PLATINUM, Rank.MASTER], [0, 1], "first"),
        _rank_batch(cfg, [Rank.DIAMOND, Rank.MASTER], [2, 3], "second"),
    ]
    for index, batch in enumerate(batches, start=1):
        batch.context.features["feature"].fill_(index * 10)
        batch.batch.target.fill_(index * 100)
    expected = [
        {
            "feature": batch.context.features["feature"].clone(),
            "pad": batch.context.ctx_pad.clone(),
            "target": batch.target.clone(),
            "rank": batch.rank.clone(),
            "weight": batch.rank_weight.clone(),
        }
        for batch in batches
    ]
    pinned = [batch.pin_memory() for batch in batches]
    copy_stream = torch.cuda.Stream()
    observed = []
    for batch in exp.device_rank_batches(pinned, "cuda", copy_stream):
        observed.append(
            {
                "feature": batch.context.features["feature"].clone(),
                "pad": batch.context.ctx_pad.clone(),
                "target": batch.target.clone(),
                "rank": batch.rank.clone(),
                "weight": batch.rank_weight.clone(),
            }
        )
        pressure = [torch.empty_like(batch.target) for _ in range(8)]
        del pressure
    torch.cuda.synchronize()

    for actual, wanted in zip(observed, expected, strict=True):
        for name in wanted:
            torch.testing.assert_close(actual[name].cpu(), wanted[name])
