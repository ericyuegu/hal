from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from hal.training.features import AWRBatch
from hal.training.features import Context
from hal.training.features import TrainBatch
from hal.training.muon import SingleDeviceMuonWithAuxAdam


def _load_notebook() -> Any:
    path = Path(__file__).parents[1] / "notebooks" / "o50_checkpoint_forensics.py"
    spec = importlib.util.spec_from_file_location("o50_checkpoint_forensics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


NOTEBOOK = _load_notebook()


def test_probe_replay_labels_expand_player_ids_to_frame_length() -> None:
    class Returns:
        def __call__(self, _compact: Any) -> dict[str, Any]:
            return {"return": torch.arange(3).numpy(), "p1_player_id": torch.tensor(7).numpy()}

    class Players:
        def __call__(self, _compact: Any) -> dict[str, Any]:
            return {
                "p1_player_id": torch.full((3,), 7).numpy(),
                "p2_player_id": torch.full((3,), 8).numpy(),
            }

    labels = NOTEBOOK.ProbeReplayLabels(Returns(), Players())({})

    assert labels["return"].shape == (3,)
    assert labels["p1_player_id"].shape == (3,)
    assert labels["p2_player_id"].shape == (3,)


def test_batch_identity_does_not_require_replay_ids() -> None:
    batch = AWRBatch(
        TrainBatch(
            Context(features={"feature": torch.arange(6).reshape(2, 3)}, ctx_pad=torch.zeros(2, dtype=torch.long)),
            target=torch.arange(8).reshape(2, 2, 2),
        ),
        returns=torch.arange(6).reshape(2, 3),
        eligible=torch.ones(2, 3, dtype=torch.bool),
    )

    first = NOTEBOOK._batch_identity(batch)
    second = NOTEBOOK._batch_identity(batch)

    assert first == second
    assert len(first) == 32


def test_checkpoint_updates_requires_complete_milestones() -> None:
    assert NOTEBOOK.checkpoint_updates(65_536) == (24_576, 32_768, 40_960, 49_152, 57_344, 65_536)
    with pytest.raises(ValueError, match="not an O50 milestone"):
        NOTEBOOK.checkpoint_updates(65_535)


def test_turning_updates_use_gameplay_lcb() -> None:
    rows = [
        {"update": 24_576, "net_stock_lcb": -0.1},
        {"update": 32_768, "net_stock_lcb": -0.4},
        {"update": 40_960, "net_stock_lcb": -0.2},
        {"update": 73_728, "net_stock_lcb": -1.0},
    ]

    assert NOTEBOOK.select_turning_updates(rows, through_update=65_536) == (24_576, 32_768, 65_536)


def test_fork_validation_allows_only_half_muon_learning_rate(tmp_path: Path) -> None:
    parent_path = tmp_path / "parent.pt"
    child_path = tmp_path / "child.pt"
    optimizer_state = {0: {"momentum_buffer": torch.arange(4, dtype=torch.float32).reshape(2, 2)}}
    parent_group = {
        "params": [0],
        "lr": 0.014,
        "initial_lr": 0.014,
        "use_muon": True,
        "momentum": 0.95,
        "weight_decay": 1e-4,
    }
    common = {
        "step": 24_575,
        "model": {"weight": torch.ones(2, 2)},
        "opt": {"state": optimizer_state},
    }
    torch.save({**common, "opt": {"state": optimizer_state, "param_groups": [parent_group]}}, parent_path)
    torch.save(
        {
            **common,
            "opt": {
                "state": optimizer_state,
                "param_groups": [{**parent_group, "lr": 0.007, "initial_lr": 0.007}],
            },
        },
        child_path,
    )
    parent = NOTEBOOK.CheckpointRef("reference", "parent", 24_576, parent_path, "parent-sha")
    child = NOTEBOOK.CheckpointRef("half_muon", "child", 24_576, child_path, "child-sha")

    result = NOTEBOOK.validate_fork(parent, child)

    assert result["model_equal"] is True
    assert result["optimizer_state_equal"] is True
    assert result["muon_lr_ratios"] == [0.5]


def _step_and_compare(
    parameter: torch.nn.Parameter,
    gradient: torch.Tensor,
    group: dict[str, Any],
    state: dict[str, Any],
) -> None:
    optimizer = SingleDeviceMuonWithAuxAdam([{**group, "params": [parameter]}])
    optimizer.state[parameter].update(
        {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in state.items()}
    )
    before = parameter.detach().clone()
    state_versions = {
        name: value._version for name, value in optimizer.state[parameter].items() if isinstance(value, torch.Tensor)
    }
    result = NOTEBOOK.hypothetical_update(
        parameter.detach(), gradient, optimizer.state[parameter], optimizer.param_groups[0], reset_state=False
    )
    assert {
        name: value._version for name, value in optimizer.state[parameter].items() if isinstance(value, torch.Tensor)
    } == state_versions

    parameter.grad = gradient.clone()
    optimizer.step()

    torch.testing.assert_close(parameter, before + result.delta)
    torch.testing.assert_close(
        optimizer.state[parameter]["exp_avg"]
        if not group["use_muon"]
        else optimizer.state[parameter]["momentum_buffer"],
        result.next_first_moment,
    )
    if result.next_second_moment is not None:
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], result.next_second_moment)


def test_hypothetical_adam_update_matches_optimizer_and_preserves_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([[0.4, -0.2], [0.7, 0.3]], dtype=torch.float32))
    gradient = torch.tensor([[0.1, -0.3], [0.2, 0.4]])
    group = {
        "lr": 3e-4,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 1e-4,
        "update_clip_threshold": None,
        "use_muon": False,
    }
    state = {
        "exp_avg": torch.tensor([[0.03, -0.04], [0.01, 0.02]]),
        "exp_avg_sq": torch.tensor([[0.2, 0.3], [0.4, 0.5]]),
        "step": 17,
    }

    _step_and_compare(parameter, gradient, group, state)


def test_hypothetical_muon_update_matches_optimizer_and_preserves_state() -> None:
    parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)
    gradient = torch.linspace(-0.3, 0.4, 12).reshape(3, 4)
    group = {
        "lr": 0.007,
        "momentum": 0.95,
        "weight_decay": 1e-4,
        "use_muon": True,
        "muon_scale_clamp_min_one": False,
        "logical_splits": 1,
    }
    state = {"momentum_buffer": torch.linspace(0.2, -0.1, 12).reshape(3, 4)}

    _step_and_compare(parameter, gradient, group, state)


def test_lazy_probe_loader_uses_bounded_validation_stream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    marker = object()

    class FakeExperiment:
        FeatureProjection = NOTEBOOK.ITEM_PLAYER_PROJECTION.__class__

        @staticmethod
        def load_identity_sidecar(_cfg: Any) -> Any:
            return SimpleNamespace(by_replay={})

    cfg = SimpleNamespace(
        player_sidecar_local=str(tmp_path / "sidecar.gz"),
        awr=SimpleNamespace(
            gamma=0.9,
            damage_shaping=1.0,
            win_reward=50.0,
            stock_value=120.0,
            return_suffix="awr_return",
            ego_return_column="ego_awr_return",
            ego_return_valid_column="ego_awr_return_valid",
        ),
        arch=SimpleNamespace(L_ctx=128, sample_chunk_length=20),
        val_split="val",
        seed=0,
        source_names=("fake",),
        mds_schema_version=7,
    )
    source = NOTEBOOK.streams.StreamSource("fake", "s3://hal/fake", Path("data/fake"))
    monkeypatch.setattr(NOTEBOOK.streams, "BY_NAME", {"fake": source})
    monkeypatch.setattr(NOTEBOOK, "replace", lambda value, **_changes: value)

    def fake_make_loader(*_args: Any, **kwargs: Any) -> object:
        calls.append(kwargs)
        return marker

    monkeypatch.setattr(NOTEBOOK, "make_loader", fake_make_loader)
    args = NOTEBOOK.Args(analysis_id="test", probe_batch_size=32, cache_limit="64gb")

    assert NOTEBOOK.make_lazy_probe_loader(FakeExperiment, cfg, {}, args) is marker
    assert calls[0]["split"] == "val"
    assert tuple(value.name for value in calls[0]["sources"]) == ("fake",)
    assert tuple(value.remote for value in calls[0]["sources"]) == (source.remote,)
    assert all(str(value.local).startswith(str(Path(args.cache_dir).resolve())) for value in calls[0]["sources"])
    assert calls[0]["cache_limit"] == "64gb"
    assert calls[0]["num_workers"] == 0
    assert calls[0]["replay_format"] == "policy-world"
    assert calls[0]["require_full_context"] is True


def test_manifest_excludes_completion_markers(tmp_path: Path) -> None:
    (tmp_path / "table.parquet").write_bytes(b"table")
    (tmp_path / "manifest.json").write_text("old")
    (tmp_path / "complete.json").write_text("old")

    manifest = NOTEBOOK._manifest(tmp_path)

    assert [value["path"] for value in manifest["files"]] == ["table.parquet"]
    assert manifest["files"][0]["sha256"] == "0d4fc4a78d3706edccafb665a8b2fdd9309e82c78625bb0f2b8e7bb9e1c4d21c"
