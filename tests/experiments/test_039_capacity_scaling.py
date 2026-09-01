"""Contracts for experiment 039 capacity/exposure/delay scaling."""

import importlib.util
import math
import sys
from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import melee
import numpy as np
import pytest
import torch

from hal.sim.inputs import ControllerInputs
from hal.sim.rollout import ObservationRow
from hal.sim.session import Matchup
from hal.sim.session import PlayerSetup
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch
from hal.sim.vec import drive_vec
from hal.training.features import A_DIM
from hal.training.features import Context


def _load(name: str, filename: str):
    path = Path(__file__).resolve().parents[2] / "experiments" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp = _load("test_exp039", "039_capacity_scaling.py")
exp026 = _load("test_exp026_for_039", "026_temporal_mtp.py")


def _tiny_scaled(**overrides):
    values = asdict(exp.scaled_config(5))
    values |= {
        "batch_size": 8,
        "grad_accum_steps": 1,
        "reservoir_capacity": 16,
        "compile_trunk": False,
        "compile_temporal": False,
        "push_to_r2": False,
    }
    values |= overrides
    return exp.TrainConfig(**values)


def test_scaled_family_and_distinct_026_baseline_geometry() -> None:
    for level in exp.CAPACITY_LEVELS:
        cfg = exp.scaled_config(level)
        assert cfg.d_model == 64 * level
        assert cfg.n_heads == cfg.n_layers == level
        assert cfg.d_model // cfg.n_heads == 64
        assert cfg.head_offsets == tuple(range(1, 37))
        assert cfg.temporal_d_model == max(128, 64 * math.ceil(cfg.d_model / 256))
        assert cfg.temporal_layers == max(2, math.ceil(level / 4))
        assert cfg.temporal_heads == cfg.temporal_d_model // 32
        assert cfg.temporal_ff_dim == cfg.group_head_dim == 2 * cfg.temporal_d_model
        assert cfg.grad_accum_steps == exp.GRAD_ACCUM_BY_LEVEL[level]
        assert cfg.batch_size // cfg.grad_accum_steps in (4, 8, 16)
        exp.validate_config(cfg)

    baseline = exp.baseline_026_config()
    scaled_l5 = exp.scaled_config(5)
    assert (baseline.d_model, baseline.n_layers, baseline.n_heads) == (256, 8, 4)
    assert baseline.head_offsets == tuple(range(1, 37))
    assert baseline.grad_accum_steps == exp.BASELINE_026_GRAD_ACCUM
    assert (baseline.d_model, baseline.n_layers) != (scaled_l5.d_model, scaled_l5.n_layers)
    exp.validate_config(baseline)


@pytest.mark.parametrize(
    ("level", "target", "total_parameters", "effective_parameters", "decay_endpoint"),
    [
        (3, 701_963_168, 2_064_111, 23_742_936, 350_981_584),
        (4, 624_290_248, 3_936_239, 26_696_984, 312_145_120),
        (5, 1_621_761_184, 6_988_015, 30_830_680, 2**30),
        (7, 1_141_118_368, 17_810_159, 43_816_664, 2**30),
        (7, 3_803_727_888, 17_810_159, 43_816_664, 2**30),
        (10, 1_493_689_000, 51_154_223, 111_580_568, 2**30),
    ],
)
def test_exact_isoflop_configs_have_expected_geometry_and_endpoint(
    level: int,
    target: int,
    total_parameters: int,
    effective_parameters: int,
    decay_endpoint: int,
) -> None:
    cfg = exp.requested_config(exp.Args(model_l=level, phase="prefix", target_positions=target))
    counts = exp.parameter_counts_for_config(cfg)

    assert counts["total"] == total_parameters
    assert counts["trunk"] + 36 * counts["decoder"] == effective_parameters
    expected_flops = 1e17 if level in (3, 4) else 3e17 if level < 10 and target < 2e9 else 1e18
    assert 6 * effective_parameters * target == pytest.approx(expected_flops, rel=1e-8)
    assert cfg.adam_weight_decay_endpoint == decay_endpoint
    expected_targets = tuple(value for value in exp.EXACT_ISOFLOP_ENDPOINTS_BY_LEVEL[level] if value <= target)
    assert exp._prefix_branch_targets(cfg) == expected_targets
    assert exp._prefix_branch_positions(cfg) == tuple(exp.branch_position(value, 0.125) for value in expected_targets)
    assert exp.branch_checkpoint_name(target) == f"branch_D{target}.pt"
    assert f"D{target}" in exp.run_name_for(exp.replace(cfg, phase="cooldown"), counts["total"])
    exp.validate_config(cfg)


def test_exact_isoflop_endpoint_is_bound_to_its_capacity() -> None:
    target = exp.EXACT_ISOFLOP_ENDPOINTS_BY_LEVEL[3][0]
    cfg = exp.scaled_config(
        4,
        exp.replace(
            exp.TrainConfig(),
            target_processed_positions=target,
            adam_weight_decay_endpoint=target,
        ),
    )
    with pytest.raises(ValueError, match="not a study endpoint"):
        exp.validate_config(cfg)


def test_long_l7_prefix_includes_both_exact_branches() -> None:
    exact_target = exp.EXACT_ISOFLOP_ENDPOINTS_BY_LEVEL[7][-1]
    cfg = exp.requested_config(exp.Args(model_l=7, phase="prefix", target_positions=exact_target))

    assert exp._prefix_branch_targets(cfg) == exp.EXACT_ISOFLOP_ENDPOINTS_BY_LEVEL[7]
    assert exp._training_stop(cfg) == 3_328_261_902


def test_addition_cooldown_boundaries_match_the_production_matrix() -> None:
    expected = {
        701_963_168: 614_217_772,
        624_290_248: 546_253_967,
        1_621_761_184: 1_419_041_036,
        1_141_118_368: 998_478_572,
        3_803_727_888: 3_328_261_902,
        1_493_689_000: 1_306_977_875,
    }

    assert {target: exp.branch_position(target, 0.125) for target in expected} == expected


def test_standard_prefix_name_remains_canonical_d2p30() -> None:
    cfg = exp.scaled_config(5, exp.replace(exp.TrainConfig(), target_processed_positions=2**26))
    counts = exp.parameter_counts_for_config(cfg)

    assert exp._training_stop(cfg) == exp.branch_position(2**30, cfg.cooldown_fraction)
    assert exp.run_name_for(cfg, counts["total"]) == "cap-L5-d320-7M-U1-prefix-D2p30-tauPL"


def test_l7_prefix_fork_requires_compatible_saved_source_branch() -> None:
    source = exp.scaled_config(7)
    target = exp.requested_config(
        exp.Args(
            model_l=7,
            phase="prefix",
            target_positions=exp.EXACT_ISOFLOP_ENDPOINTS_BY_LEVEL[7][-1],
        )
    )
    source_scale = exp.adam_scale(source, exp.parameter_counts_for_config(source)["total"])
    target_scale = exp.adam_scale(target, exp.parameter_counts_for_config(target)["total"])
    source = exp.replace(source, adam_weight_decay=source_scale.weight_decay)
    target = exp.replace(target, adam_weight_decay=target_scale.weight_decay)
    source_position = exp.branch_position(2**30, source.cooldown_fraction)

    exp._configs_match_for_prefix_fork(source, target, source_position)
    assert source_position == 939_524_096
    assert exp.branch_position(target.target_processed_positions, target.cooldown_fraction) == 3_328_261_902
    with pytest.raises(ValueError, match="not a saved branch boundary"):
        exp._configs_match_for_prefix_fork(
            source,
            target,
            exp.branch_position(target.target_processed_positions, target.cooldown_fraction),
        )
    with pytest.raises(ValueError, match="prefix fork changed muon_lr"):
        exp._configs_match_for_prefix_fork(source, exp.replace(target, muon_lr=0.02), source_position)


def test_prefix_fork_requires_cuda_rng_state() -> None:
    source = exp.scaled_config(7)
    counts = exp.parameter_counts_for_config(source)
    scale = exp.adam_scale(source, counts["total"])
    source = exp.replace(source, adam_weight_decay=scale.weight_decay)
    target = exp.requested_config(
        exp.Args(
            model_l=7,
            phase="prefix",
            target_positions=exp.EXACT_ISOFLOP_ENDPOINTS_BY_LEVEL[7][-1],
        )
    )
    state = {
        "cfg": asdict(source),
        "checkpoint_schema": 2,
        "data_state": {},
        "model": {},
        "numpy_rng_state": None,
        "opt": {},
        "pending_batches": [],
        "processed_positions": exp.branch_position(2**30, source.cooldown_fraction),
        "torch_rng_state": None,
        "update": 1,
    }

    with pytest.raises(ValueError, match="cuda_rng_state"):
        exp._validate_prefix_fork_state(state, target, counts["total"])


def test_restore_rng_restores_cuda_state_when_available(monkeypatch) -> None:
    events = []
    state = {
        "numpy_rng_state": object(),
        "torch_rng_state": object(),
        "cuda_rng_state": object(),
    }
    monkeypatch.setattr(exp.np.random, "set_state", lambda value: events.append(("numpy", value)))
    monkeypatch.setattr(exp.torch, "set_rng_state", lambda value: events.append(("torch", value)))
    monkeypatch.setattr(exp.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        exp.torch.cuda,
        "set_rng_state_all",
        lambda value: events.append(("cuda", value)),
    )

    exp._restore_rng(state)

    assert events == [
        ("numpy", state["numpy_rng_state"]),
        ("torch", state["torch_rng_state"]),
        ("cuda", state["cuda_rng_state"]),
    ]


def test_main_routes_exact_prefix_fork_without_resuming_wandb(tmp_path, monkeypatch) -> None:
    source_name = "cap-L7-d448-18M-U1-prefix-D2p30-tauPL"
    checkpoint_name = "branch_D2p30.pt"
    source = exp.scaled_config(7, exp.replace(exp.TrainConfig(), push_to_r2=False))
    source_scale = exp.adam_scale(source, exp.parameter_counts_for_config(source)["total"])
    source = exp.replace(source, adam_weight_decay=source_scale.weight_decay)
    source_state = {
        "cfg": asdict(source),
        "checkpoint_schema": 2,
        "cuda_rng_state": None,
        "data_state": {},
        "model": {},
        "numpy_rng_state": None,
        "opt": {},
        "pending_batches": [],
        "processed_positions": exp.branch_position(2**30, source.cooldown_fraction),
        "torch_rng_state": None,
        "update": 1,
        "wandb_id": "source-wandb-id",
    }
    loads = []
    trains = []

    def fake_load(run_name, run_dir, *, device, name):
        loads.append((run_name, run_dir, device, name))
        return source_state

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(exp, "load_for_resume", fake_load)
    monkeypatch.setattr(exp, "load_consolidated_stats", lambda _: {})
    monkeypatch.setattr(
        exp,
        "dataset_audit",
        lambda _: SimpleNamespace(unique_replays=1, episode_hash="episode", unique_loss_positions=1),
    )
    monkeypatch.setattr(exp, "train", lambda *args, **kwargs: trains.append((args, kwargs)))

    exp.main(
        exp.Args(
            cfg=exp.replace(exp.TrainConfig(), push_to_r2=False),
            model_l=7,
            phase="prefix",
            target_positions=3_803_727_888,
            prefix_fork_from_run=source_name,
            prefix_fork_checkpoint=checkpoint_name,
        )
    )

    assert loads == [(source_name, Path("runs") / source_name, "cpu", checkpoint_name)]
    assert len(trains) == 1
    _, kwargs = trains[0]
    assert kwargs["requested_run_name"] == "cap-L7-d448-18M-U1-prefix-D3803727888-tauPL"
    assert kwargs["resume_state"] is None
    assert kwargs["branch_state"] is None
    assert kwargs["prefix_fork_state"] is source_state
    assert kwargs["loader_workers"] == 8
    assert kwargs["loader_prefetch_updates"] == 1


def test_026_observation_trunk_codec_and_optimizer_partition_are_frozen() -> None:
    cfg = exp.baseline_026_config()
    shared = {field.name: getattr(cfg, field.name) for field in fields(exp026.TrainConfig) if hasattr(cfg, field.name)}
    shared |= {
        "sample_chunk_length": 20,
        "head_offsets": (1, 2, 3, 4, 5, 6, 9, 12, 16, 20),
        "temporal_d_model": 128,
        "temporal_layers": 2,
        "temporal_heads": 4,
        "temporal_ff_dim": 256,
        "group_head_dim": 256,
    }
    control_cfg = exp026.TrainConfig(**shared)
    torch.manual_seed(17)
    control = exp026.GPT(control_cfg)
    torch.manual_seed(17)
    treatment = exp.GPT(cfg)

    frozen_prefixes = (
        "codec.",
        "cat_embeds.",
        "v6_cat_embeds.",
        "char_emb.",
        "stage_emb.",
        "ctx_proj.",
        "trunk.",
        "hitstun_action",
    )
    ours = {name: value for name, value in treatment.state_dict().items() if name.startswith(frozen_prefixes)}
    theirs = {name: value for name, value in control.state_dict().items() if name.startswith(frozen_prefixes)}
    assert ours.keys() == theirs.keys()
    for name, value in ours.items():
        torch.testing.assert_close(value, theirs[name], rtol=0, atol=0)
    assert treatment.group_order == control.group_order == exp.GROUP_ORDER

    def assignments(model, optimizer):
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        return {
            names[id(parameter)]: (index, bool(group["use_muon"]), float(group["weight_decay"]))
            for index, group in enumerate(optimizer.param_groups)
            for parameter in group["params"]
        }

    control_groups = assignments(control, exp026.make_optimizer(control, control_cfg))
    treatment_groups = assignments(treatment, exp.make_optimizer(treatment, cfg))
    for name in control_groups.keys() & treatment_groups.keys():
        assert treatment_groups[name][0:2] == control_groups[name][0:2]


def test_dense_loss_weighting_is_exact() -> None:
    nll = torch.arange(2 * 36 * exp.N_GROUPS, dtype=torch.float32).reshape(2, 36, exp.N_GROUPS) / 100
    parts = exp.ActionLoss(nll=nll, targets=torch.zeros(2, 36, exp.N_GROUPS, dtype=torch.long))
    joint = nll.sum(dim=-1)
    expected = joint[:, :16].mean() + 0.5 * joint[:, 16:].mean()
    torch.testing.assert_close(exp.objective(parts), expected, rtol=0, atol=0)


def test_parse_eval_delays_defaults_and_validates_subset() -> None:
    assert exp.parse_eval_delays("") == exp.DELAY_BUCKETS
    assert exp.parse_eval_delays("4,6,16") == (4, 6, 16)
    with pytest.raises(ValueError, match="unique members"):
        exp.parse_eval_delays("4,4")
    with pytest.raises(ValueError, match="unique members"):
        exp.parse_eval_delays("3")


@pytest.mark.parametrize("value", ["", ".", "..", "a/b", "a\\b"])
def test_run_component_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ValueError, match="one nonempty run-name component"):
        exp.validate_run_component(value, flag="--test")


def test_repair_evaluations_preserves_checkpoint_and_commits_evidence_first(tmp_path, monkeypatch) -> None:
    run_name = "cap-test"
    run_dir = tmp_path / run_name
    replay_dir = run_dir / "replays"
    run_dir.mkdir()
    final_path = run_dir / "final.pt"
    original = b"immutable terminal checkpoint"
    final_path.write_bytes(original)
    cfg = _tiny_scaled(phase="cooldown", target_processed_positions=2**26, push_to_r2=True)
    state = {"wandb_id": "abc123", "processed_positions": 2**26}
    events = []

    class FakeUploader:
        def __init__(self, selected_run):
            assert selected_run == run_name
            events.append("uploader")

        def upload_tree(self, root, *, base, pattern="*"):
            assert base == run_dir
            assert root == replay_dir / "final_d4"
            events.append("evidence")
            return 1

        def upload(self, path, *, key=None):
            assert path == run_dir / "eval_progress.json"
            events.append("progress")
            return True

        def close(self):
            events.append("close")

    monkeypatch.setattr(exp, "setup_run_dir", lambda _: (run_dir, replay_dir))
    monkeypatch.setattr(exp, "load_for_resume", lambda *args, **kwargs: state)
    monkeypatch.setattr(exp, "load_checkpoint", lambda _: (object(), cfg, {}, state))
    monkeypatch.setattr(exp, "dataset_audit", lambda _: SimpleNamespace(episode_hash="episode"))
    monkeypatch.setattr(exp, "_load_eval_progress", lambda *args: {1})
    monkeypatch.setattr(exp, "BackgroundUploader", FakeUploader)
    monkeypatch.setattr(exp, "BF16Inference", lambda *args: object())

    def fake_eval(*args, replay_dir, **kwargs):
        replay_dir.mkdir(parents=True)
        (replay_dir / "metrics.json").write_text("{}")
        return {"metric": 1.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", fake_eval)
    monkeypatch.setattr(exp, "require_complete_eval", lambda *args: None)
    monkeypatch.setattr(exp.wandb, "init", lambda **kwargs: events.append(("wandb", kwargs["resume"])))
    monkeypatch.setattr(exp.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(exp.wandb, "finish", lambda: None)
    monkeypatch.setattr(exp.wandb, "run", SimpleNamespace(summary={}))

    exp.repair_endpoint_evaluations(run_name, (1, 4))

    assert final_path.read_bytes() == original
    assert events == [("wandb", "must"), "uploader", "evidence", "close", "uploader", "progress", "close"]


def test_repair_evaluations_reuploads_completed_progress_before_return(tmp_path, monkeypatch) -> None:
    run_name = "cap-test"
    run_dir = tmp_path / run_name
    replay_dir = run_dir / "replays"
    run_dir.mkdir()
    (run_dir / "final.pt").write_bytes(b"checkpoint")
    progress_path = run_dir / "eval_progress.json"
    progress_path.write_text("{}")
    cfg = _tiny_scaled(phase="cooldown", target_processed_positions=2**26, push_to_r2=True)
    state = {"wandb_id": "abc123", "processed_positions": 2**26}
    events = []

    class FakeUploader:
        def __init__(self, selected_run):
            assert selected_run == run_name

        def upload(self, path, *, key=None):
            assert path == progress_path
            events.append("progress")
            return True

        def close(self):
            events.append("close")

    monkeypatch.setattr(exp, "setup_run_dir", lambda _: (run_dir, replay_dir))
    monkeypatch.setattr(exp, "load_for_resume", lambda *args, **kwargs: state)
    monkeypatch.setattr(exp, "load_checkpoint", lambda _: (object(), cfg, {}, state))
    monkeypatch.setattr(exp, "dataset_audit", lambda _: SimpleNamespace(episode_hash="episode"))
    monkeypatch.setattr(exp, "_load_eval_progress", lambda *args: set(exp.DELAY_BUCKETS))
    monkeypatch.setattr(exp, "BackgroundUploader", FakeUploader)

    exp.repair_endpoint_evaluations(run_name, (1, 4))

    assert events == ["progress", "close"]


def test_explicit_eager_checkpoint_eval_disables_only_the_official_compile_guard(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    cfg = _tiny_scaled(inference_mode="compiled")
    state = {"step": 7}
    calls = []

    monkeypatch.setattr(exp, "load_checkpoint", lambda _: (object(), cfg, {}, state))

    def fake_eval(*args, **kwargs):
        calls.append(kwargs)
        return {"scheduled_boots": 2.0, "completed_boots": 2.0, "boots": 2.0}

    monkeypatch.setattr(exp, "eval_vs_cpu", fake_eval)

    exp.eval_checkpoint(str(checkpoint), delay=1, n_matchups=2, eager=True)

    assert calls[0]["require_compiled_cuda"] is False


def test_eval_prewarms_before_starting_dolphin(tmp_path, monkeypatch) -> None:
    cfg = _tiny_scaled(inference_mode="eager", eval_max_parallel=4)
    model = exp.GPT(cfg)
    inference = exp.BF16Inference(model, cfg, compiled=False)
    events = []

    def prewarm(rows, horizon):
        events.append(("prewarm", rows, horizon))
        return 1.25

    def sweep(*args, **kwargs):
        events.append(("sweep",))
        assert not model.training
        return [], []

    monkeypatch.setattr(inference, "prewarm", prewarm)
    monkeypatch.setattr(exp, "sweep_vs_cpu_prior_with_rows", sweep)
    monkeypatch.setattr(exp, "vs_cpu_metrics", lambda *args, **kwargs: {})
    monkeypatch.setattr(exp, "default_session_cfg", lambda *args, **kwargs: object())

    metrics = exp.eval_vs_cpu(
        model,
        {},
        cfg,
        n_matchups=4,
        replay_dir=tmp_path,
        inference=inference,
    )

    assert events == [("prewarm", 4, 1), ("sweep",)]
    assert metrics["inference_compile_seconds"] == pytest.approx(1.25)
    assert model.training


def test_remote_run_exists_checks_any_prefixed_object(monkeypatch) -> None:
    calls = []

    class FakeClient:
        def list_objects_v2(self, **kwargs):
            calls.append(kwargs)
            return {"KeyCount": 1, "Contents": [{"Key": "runs/new/branch_D2p29.pt"}]}

    monkeypatch.setattr(exp.r2, "client", lambda: FakeClient())
    monkeypatch.setattr(exp.r2, "bucket", lambda: "bucket")

    assert exp.remote_run_exists("new")
    assert calls == [{"Bucket": "bucket", "Prefix": "runs/new/", "MaxKeys": 1}]


def test_powerlines_decay_and_terminal_schedule_use_exact_position_formula() -> None:
    cfg = _tiny_scaled()
    parameters = 6_988_015
    scale = exp.adam_scale(cfg, parameters)
    batch_positions = cfg.batch_size * cfg.L_ctx
    tau_ref = batch_positions / (cfg.adam_lr * cfg.adam_reference_weight_decay * cfg.adam_reference_positions)
    expected_tau = (
        tau_ref
        * (
            (cfg.adam_weight_decay_endpoint / parameters)
            / (cfg.adam_reference_positions / cfg.adam_reference_parameters)
        )
        ** -0.52
    )
    expected_decay = batch_positions / (cfg.adam_lr * expected_tau * cfg.adam_weight_decay_endpoint)
    assert scale.tau == pytest.approx(expected_tau)
    assert scale.weight_decay == pytest.approx(expected_decay)

    endpoint = 2**26
    cooldown = exp.cooldown_positions(endpoint, cfg.cooldown_fraction)
    branch = endpoint - cooldown
    terminal = exp.replace(cfg, phase="cooldown", target_processed_positions=endpoint)
    assert exp.lr_multiplier(terminal, branch) == pytest.approx(1.0)
    assert exp.lr_multiplier(terminal, endpoint) == pytest.approx(cfg.lr_floor_ratio)


def test_position_boundary_retains_every_unused_loss_position() -> None:
    cfg = _tiny_scaled(L_ctx=4)
    context = Context(features={}, ctx_pad=torch.tensor([0, 2]))
    batch = exp.TrainBatch(context=context, target=torch.zeros(2, 36, A_DIM))
    available = exp._valid_position_mask(batch, cfg)
    selected, pending, count = exp._select_position_work([(batch, available)], 3)
    assert count == 3
    selected_mask = selected[0][1]
    pending_mask = pending[0][1]
    assert not (selected_mask & pending_mask).any()
    assert torch.equal(selected_mask | pending_mask, available)


class _FakeContextBuilder:
    def _ingest(self, live, obs) -> None:
        del live, obs

    def _context(self, due) -> Context:
        return Context(
            features={},
            ctx_pad=torch.zeros(len(due), dtype=torch.long),
            slot_ids=torch.arange(len(due)),
            reset=torch.zeros(len(due), dtype=torch.bool),
        )

    def _ingest_row(self, slot, row) -> None:
        del slot, row

    def _push_ego(self, slot, action) -> None:
        del slot, action


class _RecordingSession:
    def __init__(self) -> None:
        self.frame = 0
        self.main_x: list[float] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        del exc
        return False

    @staticmethod
    def _frame(frame: int) -> dict:
        value = exp._benchmark_observation(frame)
        value["start"] = {"random_seed": frame}
        return value

    def start_match(self, matchup: Matchup) -> dict:
        del matchup
        return self._frame(0)

    def step(self, inputs: Mapping[int, ControllerInputs]) -> tuple[dict, bool]:
        self.main_x.append(inputs[1].main_x)
        self.frame += 1
        return self._frame(self.frame), True


@pytest.mark.parametrize("delay", [1, 2, 4, 6])
def test_absolute_frame_alignment_executes_exact_dense_offsets(delay: int) -> None:
    cfg = exp.scaled_config(5)
    horizon = exp.decode_horizon(delay, delay)

    def predict(ctx: Context) -> np.ndarray:
        actions = np.zeros((ctx.ctx_pad.shape[0], horizon, A_DIM), dtype=np.float32)
        actions[:, :, 0] = np.arange(1, horizon + 1, dtype=np.float32) / 100
        return actions

    policy = exp.AbsoluteDelayPolicy(
        predict,
        {},
        cfg,
        delay=delay,
        replan_interval=delay,
        telemetry=None,
        device="cpu",
        float_dtype=torch.float32,
    )
    policy.context = _FakeContextBuilder()
    session = _RecordingSession()
    matchup = Matchup(
        stage=melee.Stage.FINAL_DESTINATION,
        players=(
            PlayerSetup(port=1, character=melee.Character.FOX),
            PlayerSetup(port=2, character=melee.Character.FOX, cpu_level=9),
        ),
    )
    drive_vec(
        [session],
        [VecMatch(matchup=matchup, model_ports=(1,))],
        policy,
        max_frames=3 * delay + 1,
        progress_every=0,
    )

    expected = []
    for observation_frame in range(3 * delay):
        if observation_frame < delay - 1:
            expected.append(0.0)
        else:
            offset = delay + (observation_frame - (delay - 1)) % delay
            expected.append(offset / 100)
    assert session.main_x == pytest.approx(expected)


def test_horizon_covers_every_execution_frame_at_max_delay() -> None:
    assert exp.decode_horizon(16, 16) == 31
    assert set(range(16, 32)).issubset(exp.HEAD_OFFSETS)


def test_process_worker_contract_holds_and_releases_by_absolute_frame() -> None:
    delay = 4
    horizon = exp.decode_horizon(delay, delay)

    def predict(ctx: Context) -> np.ndarray:
        actions = np.zeros((ctx.ctx_pad.shape[0], horizon, A_DIM), dtype=np.float32)
        actions[:, :, 0] = np.arange(1, horizon + 1, dtype=np.float32) / 100
        return actions

    policy = exp.AbsoluteDelayPolicy(
        predict,
        {},
        exp.scaled_config(5),
        delay=delay,
        replan_interval=delay,
        telemetry=None,
        device="cpu",
        float_dtype=torch.float32,
    )
    policy.context = _FakeContextBuilder()
    assert policy.runtime_spec.prediction_frames == policy.runtime_spec.execution_stride == 1
    slot = Slot(0, 1)
    executed = []
    for frame in range(10):
        plan = policy.plan_rows(
            {
                slot: [
                    ObservationRow(
                        frame_id=frame,
                        flat={},
                        action=np.zeros(A_DIM, dtype=np.float32),
                        reset=frame == 0,
                    )
                ]
            }
        )
        executed.append(float(plan[slot][0, 0]))
    assert executed == pytest.approx([0.0, 0.0, 0.0, 0.04, 0.05, 0.06, 0.07, 0.04, 0.05, 0.06])
