"""Return alignment, decoder isolation, and exact continuation contracts for O58."""

import copy
import importlib.util
import random
import sys
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch


def _load(number: str, filename: str) -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(f"test_exp{number}_return", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load("058", "058_return_conditioning.py")
control = _load("055", "055_history_cross_attention.py")


def tiny_cfg():
    arch = replace(
        exp.Architecture(),
        d_model=32,
        n_layers=1,
        n_heads=4,
        L_ctx=8,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=32,
        value_hidden_dim=16,
        item_hidden_dim=8,
        item_dim=5,
    )
    return exp.TrainConfig(
        arch=arch,
        batch_size=2,
        compile_trunk=False,
        compile_temporal=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
    )


def replay(frames: int) -> dict[str, np.ndarray]:
    return {
        f"{port}_{field}": np.full(frames, value, dtype=dtype)
        for port in ("p1", "p2")
        for field, value, dtype in (("stock", 4, np.int32), ("percent", 0, np.float32))
    }


def inputs(cfg):
    hidden = torch.randn(cfg.batch_size, cfg.arch.L_ctx, cfg.arch.d_model)
    pad = torch.zeros(cfg.batch_size, dtype=torch.long)
    observed = torch.zeros(cfg.batch_size, 1, exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    targets = torch.zeros(cfg.batch_size, 1, len(cfg.arch.head_offsets), exp.CONTROLLER_GROUP_COUNT, dtype=torch.long)
    return hidden, pad, observed, targets


def batch(cfg):
    value = exp.synthetic_awr_batch(cfg, torch.device("cpu"))
    return replace(value, batch=replace(value.batch, replay_ids=("a", "b")), future_return=torch.tensor([30.0, -40.0]))


def test_exact_horizon_alignment_and_port_sign() -> None:
    sample = replay(130)
    sample["p2_percent"][1:] += 1
    sample["p2_percent"][60:] += 2
    sample["p2_percent"][61:] += 4
    labels = exp.future_return_labels(sample)
    assert labels["p1_return60"][0] == pytest.approx(1 + 2 * 0.99855**59)
    assert labels["p1_return60"][1] == pytest.approx(2 * 0.99855**58 + 4 * 0.99855**59)
    np.testing.assert_array_equal(labels["p1_return60"], -labels["p2_return60"])
    assert labels["p1_return60_valid"][69]
    assert not labels["p1_return60_valid"][70]
    assert np.isnan(labels["p1_return60"][70:]).all()


def test_terminal_padding_includes_event_and_ignores_later_rewards() -> None:
    sample = replay(65)
    sample["p2_stock"][:] = 1
    sample["p2_stock"][5:] = 0
    sample["p2_percent"][5:] = 10
    sample["p2_percent"][6:] = 100
    labels = exp.future_return_labels(sample)
    assert labels["p1_return60"][4] == 180
    assert labels["p1_return60"][0] == pytest.approx(180 * 0.99855**4)
    assert (labels["p1_return60"][5:] == 0).all()
    assert labels["p1_return60_valid"].all()


def test_known_terminal_and_truncated_short_replay() -> None:
    sample = replay(20)
    labels = exp.future_return_labels(sample)
    assert not labels["p1_return60_valid"].any()
    sample["mc_terminated"] = np.asarray(True)
    labels = exp.future_return_labels(sample)
    assert labels["p1_return60_valid"].all()
    assert (labels["p1_return60"] == 0).all()


def test_shared_initialization_rng_and_unavailable_decoder_match_control() -> None:
    cfg = tiny_cfg()
    control_cfg = control.TrainConfig(
        **{
            **asdict(cfg),
            "arch": control.Architecture(**asdict(cfg.arch)),
            "awr": control.AWRCalibration(**asdict(cfg.awr)),
        }
    )
    torch.manual_seed(89)
    original = control.GPT(control_cfg)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(89)
    model = exp.GPT(cfg)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for name, value in original.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    args = inputs(cfg)
    expected = original.temporal.teacher_forced_logits_by_group(*args)
    actual = model.temporal.teacher_forced_logits_by_group(
        *args, torch.full((2,), float("nan")), torch.zeros(2, dtype=torch.bool)
    )
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_conditioning_changes_decoder_only_and_matches_stepwise() -> None:
    cfg = tiny_cfg()
    model = exp.GPT(cfg)
    value = batch(cfg)
    hidden = model.forward_dense(value.context.features, value.context.ctx_pad, None)
    critic = model.value_head(hidden[:, -1])
    observed, targets, _ = exp.prepared_targets(model, value)
    zero = torch.zeros(2)
    valid = torch.ones(2, dtype=torch.bool)
    before = model.temporal.teacher_forced_logits_by_group(
        hidden, value.context.ctx_pad, observed, targets, zero, valid
    )
    after = model.temporal.teacher_forced_logits_by_group(
        hidden, value.context.ctx_pad, observed, targets, value.future_return, valid
    )
    assert not torch.equal(before["main_stick"], after["main_stick"])
    torch.testing.assert_close(
        hidden, model.forward_dense(value.context.features, value.context.ctx_pad, None), rtol=0, atol=0
    )
    torch.testing.assert_close(critic, model.value_head(hidden[:, -1]), rtol=0, atol=0)
    stepwise = model.temporal.forced_stepwise_logits(
        hidden, value.context.ctx_pad, observed[:, 0], targets[:, 0], value.future_return, valid
    )
    for depth, frame in enumerate(stepwise):
        for name in exp.CONTROLLER_GROUP_NAMES:
            torch.testing.assert_close(after[name][:, 0, depth], frame[name], rtol=2e-5, atol=2e-5)
    sum(logit[torch.isfinite(logit)].sum() for logit in after.values()).backward()
    assert model.temporal.return_projection.weight.grad is not None
    assert model.temporal.return_projection.weight.grad.abs().sum() > 0


def test_compiled_decoder_accepts_explicit_conditions() -> None:
    cfg = tiny_cfg()
    model = exp.GPT(cfg)
    args = inputs(cfg)
    fn = torch.compile(model.temporal.teacher_forced_nll, backend="eager", fullgraph=True)
    for available in (torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)):
        desired = torch.tensor([40.0, -20.0])
        torch.testing.assert_close(
            fn(*args, desired, available), model.temporal.teacher_forced_nll(*args, desired, available)
        )
    hidden, ctx_pad, observed, _targets = args
    offsets = model.head_offsets[: cfg.prediction_frames]

    def decode(hidden, ctx_pad, observed, desired, available):
        return model.temporal.sample_indices(
            hidden,
            ctx_pad,
            observed[:, 0],
            offsets,
            desired,
            available,
            argmax=True,
        )

    compiled_decode = torch.compile(decode, backend="eager", fullgraph=True)
    desired = torch.tensor([40.0, -20.0])
    available = torch.ones(2, dtype=torch.bool)
    torch.testing.assert_close(
        compiled_decode(hidden, ctx_pad, observed, desired, available),
        decode(hidden, ctx_pad, observed, desired, available),
    )
    inference = exp.BF16Inference(model, cfg, desired_return=30.0, compiled=False)
    context = exp.synthetic_context(cfg, 2, torch.device("cpu"))
    a = inference.decode(context, cfg.prediction_frames, argmax=True)
    b = inference.decode(context, cfg.prediction_frames, argmax=True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert inference.desired_return == 30


def test_collation_and_transfer_do_not_leak_labels() -> None:
    cfg = tiny_cfg()
    value = batch(cfg)
    windows = [
        {
            "ego_awr_return": np.arange(28, dtype=np.float32),
            "ego_awr_return_valid": np.ones(28, dtype=bool),
            "ego_return60": np.arange(28, dtype=np.float32) * 2,
            "ego_return60_valid": np.ones(28, dtype=bool),
        }
        for _ in range(2)
    ]
    result = exp.collate_awr_batch(windows, value.batch, L_ctx=8).to("cpu")
    assert result.future_return.tolist() == [14, 14]
    assert result.returns[:, -1].tolist() == [8, 8]
    assert result.available.all()
    assert not any("return" in name for name in result.context.features)
    assert result.slice(1).future_return.shape == (1,)
    assert result.slice(1).batch.replay_ids == ("a",)


def test_calibration_uses_first_windows_and_strictly_positive_valid_returns() -> None:
    cfg = tiny_cfg()
    value = batch(cfg)
    n = exp.CALIBRATION_WINDOWS
    labels = torch.arange(n + 2, dtype=torch.float32) - 100
    available = torch.arange(n + 2) % 2 == 0
    calibration = exp.ReturnCalibration()
    calibration.observe(
        replace(
            value,
            batch=replace(value.batch, replay_ids=tuple(map(str, range(n + 2)))),
            future_return=labels,
            available=available,
        )
    )
    positive = labels[:n][available[:n] & (labels[:n] > 0)].numpy()
    assert calibration.targets() == (0.0, *np.quantile(positive, [0.5, 0.9]))
    state = copy.deepcopy(calibration.state_dict())
    clone = exp.ReturnCalibration()
    clone.load_state_dict(state)
    assert clone.targets() == calibration.targets()
    state["targets"] = (0.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="identity or targets"):
        clone.load_state_dict(state)


def test_boundary_checkpoint_restores_next_dropout_batch_and_optimizer_update(tmp_path: Path) -> None:
    cfg = tiny_cfg()
    torch.manual_seed(cfg.seed)
    model = exp.GPT(cfg)
    opt = exp.make_optimizer(model, cfg)
    sched = exp.LambdaLR(opt, exp.lr_schedule(cfg))
    masker = exp.IdentityMasker(cfg.seed ^ 0x0551D, 0.5)
    value = batch(cfg)

    def step(model, opt, sched, value, index):
        return exp.train_step(
            model,
            value,
            cfg,
            step=index,
            update=index + 1,
            valid_prefixes=2,
            trunk_fn=model.forward_dense,
            temporal_fn=model.temporal.teacher_forced_nll_with_diagnostics,
            optimizer=opt,
            scheduler=sched,
        )

    step(model, opt, sched, masker(value), 0)
    model.calibration.observe(value)
    path = exp.save_boundary_checkpoint(
        tmp_path,
        update=1,
        model=model,
        optimizer=opt,
        scheduler=sched,
        cfg=cfg,
        uploader=None,
        milestone=False,
        wandb_id=None,
        actual_supervised_prefixes=2,
        loader_state={"next": 1},
        identity_masker_state=masker.state_dict(),
    )
    expected_rng = (random.random(), np.random.random(), torch.rand(1))
    expected_batch = masker(value)
    step(model, opt, sched, expected_batch, 1)
    state = torch.load(path, weights_only=False)
    exp.validate_conditioning_state(state)
    restored = exp.GPT(exp.config_from_state(state["cfg"]))
    restored.load_state_dict(state["model"])
    other_opt = exp.make_optimizer(restored, cfg)
    other_sched = exp.LambdaLR(other_opt, exp.lr_schedule(cfg))
    other_opt.load_state_dict(state["opt"])
    other_sched.load_state_dict(state["sched"])
    other_mask = exp.IdentityMasker(999, 0.5)
    other_mask.load_state_dict(state["identity_masker"])
    restored.calibration.load_state_dict(state["return_calibration"])
    exp.restore_rng(state["rng"])
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    torch.testing.assert_close(torch.rand(1), expected_rng[2], rtol=0, atol=0)
    next_batch = other_mask(value)
    for name, tensor in expected_batch.context.features.items():
        torch.testing.assert_close(next_batch.context.features[name], tensor, rtol=0, atol=0)
    step(restored, other_opt, other_sched, next_batch, 1)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    state["conditioning_protocol"]["horizon"] = 59
    with pytest.raises(ValueError, match="incompatible"):
        exp.validate_conditioning_state(state)


def test_saved_control_rows_have_the_registered_checkpoint_and_protocol(tmp_path: Path) -> None:
    cfg = exp.proxy_config()
    model = exp.GPT(cfg)
    protocol = exp._eval_protocol(
        cfg,
        model,
        n_matchups=96,
        checkpoint_sha256=exp.CONTROL_CHECKPOINT_SHA256,
        desired_return=None,
        inference_compile_mode="default",
    )
    path = Path(__file__).resolve().parents[2] / "experiments/o58/control-selection-rows.json"
    rows = exp.validated_eval_rows(path, protocol, historical=True)
    assert len({row.boot_index for row in rows}) == 96
    altered = tmp_path / "wrong.json"
    import json

    payload = json.loads(path.read_text())
    payload["protocol"]["delay_frames"] = 3
    altered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="protocol mismatch"):
        exp.validated_eval_rows(altered, protocol, historical=True)
    payload["protocol"]["delay_frames"] = 2
    payload["rows"] = [row for row in payload["rows"] if row["boot_index"] != 0]
    altered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="96 boots"):
        exp.validated_eval_rows(altered, protocol, historical=True)


def test_target_selection_uses_only_gameplay_and_stable_ties() -> None:
    rows = [{"treatment_net_stock_per_min": x, "offline_nll": y} for x, y in ((-1, 0), (2, 100), (1, 1))]
    assert exp.select_return_target(rows) == 1
    assert exp.select_return_target([rows[1]] * 3) == 0
    with pytest.raises(ValueError, match="all three"):
        exp.select_return_target(rows[:2])


def test_unavailable_windows_keep_the_original_awr_loss() -> None:
    cfg = tiny_cfg()
    model = exp.GPT(cfg)
    value = replace(
        batch(cfg), future_return=torch.full((2,), float("nan")), available=torch.zeros(2, dtype=torch.bool)
    )
    actual, _, _ = exp.microbatch_loss(
        model,
        value,
        cfg,
        step=5000,
        valid_prefixes=2,
        trunk_fn=model.forward_dense,
        temporal_fn=model.temporal.teacher_forced_nll_with_diagnostics,
    )
    zeros = replace(value, future_return=torch.zeros(2), available=torch.ones(2, dtype=torch.bool))
    expected, _, _ = exp.microbatch_loss(
        model,
        zeros,
        cfg,
        step=5000,
        valid_prefixes=2,
        trunk_fn=model.forward_dense,
        temporal_fn=model.temporal.teacher_forced_nll_with_diagnostics,
    )
    assert torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_validation_shuffle_does_not_change_rng_or_masks() -> None:
    cfg = tiny_cfg()
    model = exp.GPT(cfg)
    values = [
        batch(cfg),
        replace(batch(cfg), future_return=torch.tensor([float("nan"), 120.0]), available=torch.tensor([False, True])),
    ]
    before = torch.get_rng_state()
    result = exp.conditioning_diagnostic(model, values, cfg)
    assert torch.equal(torch.get_rng_state(), before)
    assert result["return60_available_fraction"] == 0.75
    assert all(np.isfinite(value) for value in result.values())


def test_compact_labels_preserve_full_match_awr_and_are_pickleable() -> None:
    import pickle

    from hal.data.policy_schema import pack_player_state

    sample = replay(130)
    sample["p2_stock"][100:] = 0
    sample["p2_percent"][30:] = 12
    compact = {"num_frames": 130, "replay_id": "example", "p1_rank": 1, "p2_rank": 2}
    for port in ("p1", "p2"):
        zeros = np.zeros(130, dtype=np.int32)
        compact[f"{port}_state"] = pack_player_state(
            {
                "stock": sample[f"{port}_stock"],
                "action": zeros,
                "jumps_used": zeros,
                "hurtbox_state": zeros,
                "airborne": zeros,
                "direction": zeros.astype(np.float32),
            }
        )
        compact[f"{port}_percent"] = sample[f"{port}_percent"]
    original = exp.returns_lib.PolicyReturnLabels(exp.ReplayPlayerLookup({}), 0.99855, 1.0, 50.0, 120.0, "awr_return")
    callback = pickle.loads(pickle.dumps(exp.ReturnLabels(original)))
    expected = original(compact)
    actual = callback(compact)
    for name, value in expected.items():
        if value.shape == ():
            assert actual[name].shape == (130,)
            np.testing.assert_array_equal(actual[name], np.full(130, value.item(), dtype=value.dtype))
        else:
            np.testing.assert_array_equal(actual[name], value)
    assert actual["p1_return60"][0] == pytest.approx(12 * 0.99855**29)
    assert actual["p1_awr_return"][1] > actual["p1_return60"][0]


loader_tests = _load("physical", "../tests/test_physical_shard_loader.py")


class ReturnAdapter(loader_tests._FakeAdapter):
    def __init__(self, cfg):
        super().__init__(rows=30, length=cfg.arch.L_ctx + cfg.arch.sample_chunk_length)
        context = exp.synthetic_context(cfg, 1, torch.device("cpu"))
        self.columns = {name: np.repeat(value[0, :1].numpy(), self.length) for name, value in context.features.items()}

    def _generation(self, task, row, epoch, windows=4):
        replay_id, samples = super()._generation(task, row, epoch, windows)
        output = []
        for index, sample in enumerate(samples):
            columns = {**sample, **{name: value.copy() for name, value in self.columns.items()}}
            value = np.float32(row + epoch + index)
            columns["ego_percent"][:] = value
            columns["ego_awr_return"] = np.full(self.length, value * 3)
            columns["ego_awr_return_valid"] = np.ones(self.length, dtype=bool)
            columns["ego_return60"] = np.full(self.length, value)
            columns["ego_return60_valid"] = np.full(self.length, row % 3 != 0)
            output.append(columns)
        return replay_id, tuple(output)


def physical_loader(cfg):
    import functools

    from hal.training.ego_stats import consolidate_key
    from hal.training.features import feature_kind

    adapter = ReturnAdapter(cfg)
    stats = {
        consolidate_key(name): exp.FeatureStats(0, 1, 0, 100)
        for name in adapter.columns
        if feature_kind(name, exp.ITEM_PLAYER_COLUMNS) == "float"
    }
    projection = exp.FeatureProjection(
        exp.ITEM_PLAYER_PROJECTION.columns
        | {"ego_awr_return", "ego_awr_return_valid", "ego_return60", "ego_return60_valid"},
        derive_spatial=False,
    )
    selection = exp.PhysicalShardSelection.from_sources((exp.SourceRowSelection("source", 30),))
    return exp.PhysicalShardReplayLoader(
        selection=selection,
        adapter=adapter,
        tasks=exp.build_shard_plan(selection, adapter.manifests),
        data_protocol=cfg.data_protocol,
        source_manifest_sha256={"source": "c" * 64},
        labels=loader_tests._no_labels,
        projection=projection,
        batch_transform=functools.partial(
            exp._collate_o58_batch,
            stats=stats,
            projection=projection,
            context_length=cfg.arch.L_ctx,
            return_column="ego_awr_return",
            return_valid_column="ego_awr_return_valid",
        ),
        batch_size=2,
        replay_slots=20,
        seed=cfg.seed,
        num_workers=0,
        context_length=cfg.arch.L_ctx,
        chunk_length=cfg.arch.sample_chunk_length,
        windows_per_generation=2,
        replay_phase_block_batches=2,
        schema_version=7,
        reserved_disk_bytes=0,
        pin_memory=False,
        materialization_threads=0,
    )


def test_physical_loader_prefetch_resume_reproduces_next_batch_and_update(tmp_path: Path) -> None:
    cfg = tiny_cfg()
    torch.manual_seed(11)
    model = exp.GPT(cfg)
    opt = exp.make_optimizer(model, cfg)
    sched = exp.LambdaLR(opt, exp.lr_schedule(cfg))
    mask = exp.IdentityMasker(91, 0.5)
    with physical_loader(cfg) as loader:
        prefetch = exp.DeviceBatchPrefetcher(loader, cfg, "cpu", mask)
        try:
            first, count = prefetch.next()
            exp.train_step(
                model,
                first,
                cfg,
                step=0,
                update=1,
                valid_prefixes=count,
                trunk_fn=model.forward_dense,
                temporal_fn=model.temporal.teacher_forced_nll_with_diagnostics,
                optimizer=opt,
                scheduler=sched,
            )
            model.calibration.observe(first)
            assert prefetch.drained
            path = exp.save_boundary_checkpoint(
                tmp_path,
                update=1,
                model=model,
                optimizer=opt,
                scheduler=sched,
                cfg=cfg,
                uploader=None,
                milestone=False,
                wandb_id=None,
                actual_supervised_prefixes=2,
                loader_state=loader.state_dict(),
                identity_masker_state=mask.state_dict(),
            )
            prefetch.fill_lookahead(1)
            prefetch.stage_next()
            expected, count = prefetch.next()
            exp.train_step(
                model,
                expected,
                cfg,
                step=1,
                update=2,
                valid_prefixes=count,
                trunk_fn=model.forward_dense,
                temporal_fn=model.temporal.teacher_forced_nll_with_diagnostics,
                optimizer=opt,
                scheduler=sched,
            )
        finally:
            prefetch.close()
    state = torch.load(path, weights_only=False)
    clone = exp.GPT(cfg)
    clone.load_state_dict(state["model"])
    clone_opt = exp.make_optimizer(clone, cfg)
    clone_sched = exp.LambdaLR(clone_opt, exp.lr_schedule(cfg))
    clone_opt.load_state_dict(state["opt"])
    clone_sched.load_state_dict(state["sched"])
    clone_mask = exp.IdentityMasker(91, 0.5)
    clone_mask.load_state_dict(state["identity_masker"])
    with physical_loader(cfg) as loader:
        loader.load_state_dict(state["loader"])
        iterator = iter(loader)
        exp.restore_rng(state["rng"])
        prefetch = exp.DeviceBatchPrefetcher(loader, cfg, "cpu", clone_mask, iterator=iterator)
        try:
            actual, count = prefetch.next()
            assert actual.batch.replay_ids == expected.batch.replay_ids
            for name in ("future_return", "available", "returns", "eligible", "target"):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
            for name in actual.context.features:
                torch.testing.assert_close(
                    actual.context.features[name], expected.context.features[name], rtol=0, atol=0
                )
            exp.train_step(
                clone,
                actual,
                cfg,
                step=1,
                update=2,
                valid_prefixes=count,
                trunk_fn=clone.forward_dense,
                temporal_fn=clone.temporal.teacher_forced_nll_with_diagnostics,
                optimizer=clone_opt,
                scheduler=clone_sched,
            )
        finally:
            prefetch.close()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(clone.state_dict()[name], tensor, rtol=0, atol=0)
