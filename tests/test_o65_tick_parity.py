"""Compare fixed tick inputs with a0ae12df, which must be in Git history."""

import importlib.util
import io
import math
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.conditioning import ConditioningRanges
from hal.models.self_play import SelfPlayTransformer
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import feature_kind
from hal.representation.player_identity import PlayerVocabulary

CONTROL = "a0ae12df"
ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def tick_outputs(root: Path = ROOT, device: str = "cpu") -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("o65_tick_parity_runtime", root / "experiments/065_selfplay_ppo.py")
    assert spec is not None and spec.loader is not None
    exp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = exp
    spec.loader.exec_module(exp)
    torch.set_num_threads(2)
    torch.manual_seed(171)
    vocabulary = PlayerVocabulary(())
    cfg = ActionSequenceConfig(
        d_model=32,
        n_layers=2,
        n_heads=2,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=2,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
        return_embed_dim=16,
        head_offsets=tuple(range(1, 29)),
        player_vocab_size=vocabulary.size,
        player_vocab_sha256=vocabulary.sha256,
    )
    model = (
        SelfPlayTransformer(
            cfg,
            ConditioningRanges((math.log(0.0001), math.log(0.1)), (16, 21), (2, 4)),
            vocabulary,
            max_opponent_queue=24,
            critic_hidden_dim=16,
        )
        .to(device)
        .eval()
    )
    # Exercise the decoder contribution to values, which starts at zero in O64.
    model.critic_residual.residual.down.weight.normal_(std=0.02)
    stats = {
        consolidate_key(name): FeatureStats(0, 1, -1, 1)
        for name in BASE_ITEMS_PROJECTION.columns
        if feature_kind(name, ITEM_COLUMNS) == "float" and not name.startswith("ego_button_")
    }
    policy = exp.TransformerPolicy(model, exp.production_config(cfg.head_offsets), stats)
    store = exp.FrameStore(140, 6, torch.device(device))
    store.actions.uniform_()
    table = exp.DenseRollout(64, store)
    table.allocate_evaluation(model)
    ids = torch.tensor([3, 1, 5, 0, 4, 2], device=device)
    streams = (7, 2, 9, 0, 5, 10)
    lanes = (3, 1, 4, 0, 2, 5)
    layout = exp.TickLayout(streams, lanes)
    meta = torch.zeros(6, 18, dtype=torch.long, device=device)
    meta[:, exp.STREAM] = torch.tensor(streams, device=device)
    meta[:, exp.LANE] = torch.tensor(lanes, device=device)
    meta[:, exp.SIDE] = torch.tensor(streams, device=device) % 2
    meta[:, exp.LENGTH] = torch.tensor([129, 1, 2, 65, 80, 129], device=device)
    meta[:, exp.DELAY] = torch.tensor([16, 17, 18, 19, 20, 21], device=device)
    meta[:, exp.REPLAN] = torch.tensor([4, 2, 4, 2, 4, 2], device=device)
    meta[:, exp.OPP_DELAY] = meta[:, exp.DELAY].flip(0)
    meta[:, exp.OPP_REPLAN] = meta[:, exp.REPLAN].flip(0)
    meta[:, exp.OPP_LENGTH] = meta[:, exp.OPP_DELAY]
    table.conditions[ids] = torch.tensor([-5.0, -6.0, 0.3, 0.7], device=device)
    table.prefix[ids] = model.codec.quantize(torch.rand(6, 24, exp.ACTION_DIM, device=device))
    table.opponent[ids] = model.codec.quantize(torch.rand(6, 24, exp.ACTION_DIM, device=device))
    table.active[ids] = torch.tensor([True, False, True, True, False, True], device=device)
    ring = exp.ActionRing(6, 32, torch.device(device))
    ring.length_device[meta[:, exp.LANE], meta[:, exp.SIDE]] = meta[:, exp.DELAY]
    states = exp.TickInputs(table, ids)

    class EndpointRng(exp.CounterRng):
        def tensor_uniform(
            self, domain: int, stream: torch.Tensor, episode: torch.Tensor, frame: torch.Tensor
        ) -> torch.Tensor:
            # Endpoint uniforms keep discrete draws stable under BF16 rounding.
            return torch.where(
                stream % 2 == 0,
                torch.zeros_like(stream, dtype=torch.float64),
                torch.full_like(stream, 1 - 2**-24, dtype=torch.float64),
            )

    rng = EndpointRng(65) if device == "cuda" else exp.CounterRng(65)
    inputs = [value.clone() for value in model.parameters()]
    inputs.extend([store.actions.clone(), table.prefix[ids], table.opponent[ids], meta.clone()])
    outputs = []
    for clock in (128, 130, 132, 134):
        meta[:, exp.CLOCK] = clock
        table.integers[ids] = meta
        uniforms = policy._tensor_action_uniforms(meta, rng)
        policy.sample_tick(states, rng, layout, ring, clock=clock)
        due = table.active[ids] & (clock % meta[:, exp.REPLAN] == 0)
        rows = ids[due]
        cache = policy.tick_caches[ring, (layout.buffer, len(ids))]
        outputs.append(
            {
                "actions": table.actions[rows].clone(),
                "log_probs": table.behavior[rows].clone(),
                "values": table.values[rows].clone(),
                "distributions": tuple(value[rows].clone() for value in table.evaluation.distributions),
                "decoder": table.evaluation.features.decoder[rows].clone(),
                "ring": ring.values.clone(),
                "controllers": ring.controllers.clone(),
                "lengths": ring.length_device.clone(),
                "cache": tuple(value.clone() for value in cache.cache.buffers()),
                "uniforms": uniforms,
            }
        )
        meta[:, exp.LENGTH] += 2
    return {"inputs": inputs, "ticks": outputs}


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tick_matches_control_actions_log_probs_values_and_ring(tmp_path: Path, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    control = tmp_path / "control"
    control.mkdir()
    archive = subprocess.check_output(["git", "archive", CONTROL, "hal", "experiments/065_selfplay_ppo.py"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(control, filter="data")
    output = tmp_path / "control.pt"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(output), device],
        cwd=control,
        env={**os.environ, "PYTHONPATH": str(control)},
        check=True,
        timeout=300,
    )
    expected = torch.load(output, weights_only=False)
    actual = tick_outputs(device=device)
    for value, reference in zip(actual["inputs"], expected["inputs"], strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
    for tick, control_tick in zip(actual["ticks"], expected["ticks"], strict=True):
        for name in ("actions", "ring", "controllers", "lengths", "uniforms"):
            torch.testing.assert_close(tick[name], control_tick[name], atol=0, rtol=0)
        for name in ("log_probs", "values", "decoder"):
            torch.testing.assert_close(tick[name], control_tick[name], atol=1.6e-2, rtol=1.6e-2)
        for value, reference in zip(tick["distributions"], control_tick["distributions"], strict=True):
            torch.testing.assert_close(value, reference, atol=2e-4, rtol=1.6e-2)
        for value, reference in zip(tick["cache"], control_tick["cache"], strict=True):
            torch.testing.assert_close(
                value, reference, atol=1.6e-2 if device == "cuda" else 2e-6, rtol=1.6e-2 if device == "cuda" else 2e-5
            )


if __name__ == "__main__":
    torch.save(tick_outputs(Path.cwd(), sys.argv[2]), sys.argv[1])
