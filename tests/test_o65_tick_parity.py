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
def tick_outputs(root: Path = ROOT, device: str = "cpu", frozen: bool = False) -> dict[str, Any]:
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
    model.requires_grad_(not frozen)
    stats = {
        consolidate_key(name): FeatureStats(0, 1, -1, 1)
        for name in BASE_ITEMS_PROJECTION.columns
        if feature_kind(name, ITEM_COLUMNS) == "float" and not name.startswith("ego_button_")
    }

    class RecordingPolicy(exp.TransformerPolicy):
        def __init__(self, model: SelfPlayTransformer, schedule: Any, statistics: dict[str, FeatureStats]) -> None:
            super().__init__(model, schedule, statistics)
            self.sampled_streams: list[int] = []

        def _tensor_action_uniforms(self, meta: torch.Tensor, rng: Any) -> torch.Tensor:
            if meta.device.type == "cpu":
                self.sampled_streams.extend(meta[:, exp.STREAM].tolist())
            return super()._tensor_action_uniforms(meta, rng)

    policy = RecordingPolicy(model, exp.production_config(cfg.head_offsets), stats)
    store = exp.FrameStore(140, 10, torch.device(device))
    store.actions.uniform_()
    table = exp.DenseRollout(64, store)
    table.allocate_evaluation(model)
    ids = torch.tensor([3, 1, 5, 0, 4, 2, 9, 6, 8, 7], device=device)
    streams = (7, 2, 9, 0, 5, 10, 13, 14, 17, 18)
    lanes = (3, 1, 4, 0, 2, 5, 6, 7, 8, 9)
    layout = exp.TickLayout(streams, lanes)
    meta = torch.zeros(10, 18, dtype=torch.long, device=device)
    meta[:, exp.STREAM] = torch.tensor(streams, device=device)
    meta[:, exp.LANE] = torch.tensor(lanes, device=device)
    meta[:, exp.SIDE] = torch.tensor(streams, device=device) % 2
    meta[:, exp.LENGTH] = torch.tensor([129, 1, 2, 65, 80, 129, 7, 100, 63, 127], device=device)
    meta[:, exp.DELAY] = torch.tensor([16, 17, 18, 19, 20, 21, 16, 17, 18, 19], device=device)
    meta[:, exp.REPLAN] = torch.tensor([4, 2, 4, 2, 4, 2, 4, 2, 4, 2], device=device)
    meta[:, exp.OPP_DELAY] = meta[:, exp.DELAY].flip(0)
    meta[:, exp.OPP_REPLAN] = meta[:, exp.REPLAN].flip(0)
    meta[:, exp.OPP_LENGTH] = meta[:, exp.OPP_DELAY]
    table.conditions[ids] = torch.tensor([-5.0, -6.0, 0.3, 0.7], device=device)
    table.prefix[ids] = model.codec.quantize(torch.rand(10, 24, exp.ACTION_DIM, device=device))
    table.opponent[ids] = model.codec.quantize(torch.rand(10, 24, exp.ACTION_DIM, device=device))
    table.active[ids] = torch.tensor([True, False, True, True, False, True, False, True, False, False], device=device)
    ring = exp.ActionRing(10, 32, torch.device(device))
    ring.length_device[meta[:, exp.LANE], meta[:, exp.SIDE]] = meta[:, exp.DELAY]
    states = exp.TickInputs(table, ids)
    ownership = torch.ones(10, dtype=torch.bool, device=device)

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
    for clock in (128, 130, 132, 134, 136):
        if clock == 132:
            # Replays read current table rows and observations, including a
            # different row assignment for the same stream layout.
            active = table.active[ids].clone()
            ids = ids.roll(1)
            table.active[ids] = active
            table.conditions[ids, 0] += 0.125
            table.prefix[ids] = table.prefix[ids].flip(1)
            table.opponent[ids] = table.opponent[ids].flip(1)
            store.actions[129:] *= 0.5
            ownership[2] = False
        if clock == 134:
            # R2 grows from bucket 4 to bucket 6. Bucket 6 was captured with
            # inactive synthetic rows; production must replay it on live rows.
            table.active[ids[[1, 9]]] = True
        if clock == 136:
            table.active[ids] = False
        meta[:, exp.CLOCK] = clock
        table.integers[ids] = meta
        states = exp.TickInputs(table, ids, enabled=ownership)
        if device == "cuda" and "decoder_rows" in exp.TickInputs.__dataclass_fields__:
            due = table.active[ids] & ownership & (clock % meta[:, exp.REPLAN] == 0)
            decoder_rows = tuple(torch.nonzero(due).flatten().cpu().tolist())
            states = exp.TickInputs(table, ids, enabled=ownership, decoder_rows=decoder_rows)
        uniforms = policy._tensor_action_uniforms(meta, rng)
        policy.sampled_streams.clear()
        if device == "cuda" and clock == 134 and hasattr(policy, "tick_graphs"):
            graphs = len(policy.tick_graphs)
            cache_key = ring, (layout.buffer, len(ids))
            if "decoder_rows" in exp.TickInputs.__dataclass_fields__:
                assert (cache_key, 2, 2, 6) in policy.tick_graphs
        policy.sample_tick(states, rng, layout, ring, clock=clock)
        if device == "cuda" and clock == 134 and hasattr(policy, "tick_graphs"):
            assert len(policy.tick_graphs) == graphs
        due = table.active[ids] & ownership & (clock % meta[:, exp.REPLAN] == 0)
        if device == "cpu" and "decoder_rows" in exp.TickInputs.__dataclass_fields__:
            assert policy.sampled_streams == meta[due, exp.STREAM].tolist()
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
@pytest.mark.parametrize("frozen", [False, True])
def test_tick_matches_control_actions_log_probs_values_and_ring(tmp_path: Path, device: str, frozen: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    control = tmp_path / "control"
    control.mkdir()
    archive = subprocess.check_output(["git", "archive", CONTROL, "hal", "experiments/065_selfplay_ppo.py"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(control, filter="data")
    output = tmp_path / "control.pt"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(output), device, str(int(frozen))],
        cwd=control,
        env={**os.environ, "PYTHONPATH": str(control)},
        check=True,
        timeout=300,
    )
    expected = torch.load(output, weights_only=False)
    actual = tick_outputs(device=device, frozen=frozen)
    for value, reference in zip(actual["inputs"], expected["inputs"], strict=True):
        torch.testing.assert_close(value, reference, atol=0, rtol=0)
    for tick, control_tick in zip(actual["ticks"], expected["ticks"], strict=True):
        for name in ("actions", "ring", "controllers", "lengths", "uniforms"):
            torch.testing.assert_close(tick[name], control_tick[name], atol=0, rtol=0)
        for name in ("log_probs", "values", "decoder"):
            torch.testing.assert_close(tick[name], control_tick[name], atol=1.6e-2, rtol=1.6e-2)
        for value, reference in zip(tick["distributions"], control_tick["distributions"], strict=True):
            # Compare every categorical log-probability under the same BF16
            # contract as the sampled factors, including exact zero support.
            positive = reference > 0
            assert torch.equal(value > 0, positive)
            torch.testing.assert_close(value[positive].log(), reference[positive].log(), atol=1.6e-2, rtol=1.6e-2)
        for value, reference in zip(tick["cache"], control_tick["cache"], strict=True):
            torch.testing.assert_close(
                value, reference, atol=1.6e-2 if device == "cuda" else 2e-6, rtol=1.6e-2 if device == "cuda" else 2e-5
            )


@pytest.mark.parametrize(
    "clock,decoder_rows,message",
    [(0, (0,), "omit a due row"), (2, (0,), "not due"), (0, (0, 0), "distinct"), (0, (2,), "row indices")],
)
def test_tick_rejects_incorrect_decoder_rows(clock: int, decoder_rows: tuple[int, ...], message: str) -> None:
    from test_065_loop import exp
    from test_065_loop import transformer_trainer

    run = transformer_trainer(development=True)
    engine = run.engine
    table = exp.DenseRollout(64, engine.stores[0])
    table.allocate_evaluation(run.policy.model)
    ids = engine._stage_tick(clock, 0, table, torch.full((2,), -1, dtype=torch.long))
    table.integers[ids, exp.REPLAN] = 4
    meta = table.integers[ids]
    layout = exp.TickLayout(tuple(meta[:, exp.STREAM].tolist()), tuple(meta[:, exp.LANE].tolist()))
    states = exp.TickInputs(table, ids, decoder_rows=decoder_rows)
    with pytest.raises((ValueError, RuntimeError), match=message):
        run.policy.sample_tick(states, engine.sampler.rng, layout, engine.rings[0], clock=clock)


def test_staged_decoder_rows_match_device_metadata_and_ownership() -> None:
    from test_065_loop import exp
    from test_o65_fixed_layout import fixed_trainer

    engine = fixed_trainer().engine
    table = exp.DenseRollout(64, engine.stores[0])
    pending = torch.full((table.players,), -1, dtype=torch.long)
    for clock in (0, 2):
        if clock == 2:
            engine.active[0][0] = False
            engine.live_host[0][0] = False
        ids = engine._stage_tick(clock, 0, table, pending)
        for key in engine.tick_layouts:
            slots = engine.group_slots[0][key]
            selected = ids[slots]
            owner = -1 if key[1] is None else key[1]
            metadata = table.integers[selected]
            due = table.active[selected] & (engine.owners[0][slots] == owner)
            due &= metadata[:, exp.CLOCK] % metadata[:, exp.REPLAN] == 0
            assert engine.tick_decoder_rows[key] == tuple(torch.nonzero(due).flatten().tolist())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_head_start_preserves_captured_store_addresses(device: str) -> None:
    from test_065_loop import exp
    from test_065_loop import trainer

    from hal.inference.cuda_graph import CapturedCall

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    engine = trainer(device, groups=1).engine
    count = engine.sampler.schedule.rollout_frames
    previous = engine.stores[0]
    store = exp.FrameStore(129 + count + count // 128 + 1, len(engine.group_matches[0]), engine.policy.device)
    for name in ("records", "actions", "rewards", "flags"):
        getattr(store, name)[: len(previous.records)].copy_(getattr(previous, name))
    engine.stores[0] = store
    table = exp.DenseRollout(count // 2, store)
    engine.tables[0] = table
    captured = (
        CapturedCall(lambda: store.records.clone(), (store.records,), device=engine.policy.device)
        if device == "cuda"
        else None
    )
    engine.stagger_starts(64)
    assert engine.stores[0] is store
    engine.collect(-1)
    assert engine.stores[0] is table.store is store
    assert torch.count_nonzero(store.records[1:])
    if captured:
        torch.testing.assert_close(captured((store.records,)), table.store.records, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tick_linear_casts_refresh_without_changing_master_weights_or_gradients(device: str) -> None:
    from test_065_loop import exp

    from hal.inference.cuda_graph import CapturedCall

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(91)
    source = torch.nn.Linear(64, 96, device=device)
    layer = exp.TickLinear(source)
    assert layer.weight is source.weight and layer.bias is source.bias
    assert layer.state_dict().keys() == source.state_dict().keys()
    inputs = torch.randn(8, 64, device=device, requires_grad=True)
    parameters = inputs, source.weight, source.bias
    expected = source(inputs)
    actual = layer(inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for gradient, reference in zip(
        torch.autograd.grad(actual.sum(), parameters), torch.autograd.grad(expected.sum(), parameters), strict=True
    ):
        torch.testing.assert_close(gradient, reference, atol=0, rtol=0)
    layer.tick = True
    # The critic explicitly disables autocast and must keep its FP32 path.
    torch.testing.assert_close(layer(inputs), source(inputs), atol=0, rtol=0)
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, cache_enabled=False):
        static = inputs.detach().clone()
        captured = (
            CapturedCall(
                torch.compile(lambda: layer(static), fullgraph=True, options={"triton.cudagraphs": False}),
                (static,),
                device=torch.device(device),
            )
            if device == "cuda"
            else None
        )
        for _ in range(2):
            source.weight.add_(0.125)
            source.bias.mul_(0.75)
            layer.refresh()
            actual = layer(static) if captured is None else captured((inputs.detach(),))
            torch.testing.assert_close(actual, source(inputs), atol=0, rtol=0)


if __name__ == "__main__":
    torch.save(tick_outputs(Path.cwd(), sys.argv[2], bool(int(sys.argv[3]))), sys.argv[1])
