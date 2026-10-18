"""Every observation and attention path must stop at an episode boundary."""

import pytest
import torch

from hal.data.feature_stats import FeatureStats
from hal.data.feature_stats import consolidate_key
from hal.inference.column_context import ColumnContextBuilder
from hal.models.action_sequence import ActionSequenceConfig
from hal.models.action_sequence import ActionSequenceTransformer
from hal.models.attention import Trunk
from hal.models.attention import TrunkConfig
from hal.models.attention import segment_cu_seqlens
from hal.models.attention import segment_starts
from hal.representation.features import BASE_ITEMS_PROJECTION
from hal.representation.features import ITEM_COLUMNS
from hal.representation.features import feature_kind
from hal.representation.features import preprocess
from hal.wire import ACTION_CHANNELS
from hal.wire import ACTION_DIM
from hal.wire import MASK_INT32


def _columns(
    port: int, device: str = "cpu"
) -> tuple[ColumnContextBuilder, dict[str, torch.Tensor], dict[str, FeatureStats]]:
    stats = {
        consolidate_key(name): FeatureStats(1.0, 3.0, -2.0, 5.0)
        for name in BASE_ITEMS_PROJECTION.columns
        if feature_kind(name, ITEM_COLUMNS) == "float"
    }
    builder = ColumnContextBuilder(stats, port)
    columns = {}
    for source, _, kind, _, _, _ in builder.routes:
        value = torch.randint(0, 2, (2, 13), device=device) if kind == "cat" else torch.randn(2, 13, device=device)
        columns[source] = value
    return builder, columns, stats


@pytest.mark.parametrize("port", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_columns_match_python_preprocessing(port: int, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    builder, columns, stats = _columns(port, device)
    # Include missing float and categorical fields, plus an empty item slot.
    columns[f"p{port}_position_x"][0, 3] = float("nan")
    columns[f"p{port}_stock"][0, 4] = MASK_INT32
    columns["item0_pos_x"][0, :] = float("nan")
    applied = torch.rand(2, 13, 2, ACTION_DIM, device=device)
    segments = torch.zeros(2, 13, dtype=torch.long, device=device)
    segments[:, 6:] = 3
    resets = torch.zeros_like(segments, dtype=torch.bool)
    resets[:, [0, 6]] = True
    ctx = builder(columns, applied, segments, resets, torch.tensor([3, 4], device=device))
    raw = {name: columns[source].cpu().numpy() for source, name, *_ in builder.routes}
    for name in ACTION_CHANNELS:
        raw[f"ego_{name}"] = ctx.features[f"ego_{name}"].cpu().numpy()
    expected = preprocess(raw, stats, extra=ITEM_COLUMNS, projection=BASE_ITEMS_PROJECTION)
    for name, value in expected.items():
        torch.testing.assert_close(ctx.features[name].cpu(), value, rtol=0, atol=0)
    assert ctx.segment_ids is segments
    assert ctx.to("cpu").segment_ids is not None
    assert all(ctx.features[f"ego_{name}"][:, 6].eq(0).all() for name in ACTION_CHANNELS)


def test_column_context_preserves_each_episodes_identity() -> None:
    builder, columns, _ = _columns(1)
    actions = torch.zeros(2, 13, 2, ACTION_DIM)
    segments = torch.zeros(2, 13, dtype=torch.long)
    segments[:, 6:] = 1
    starts = torch.zeros_like(segments, dtype=torch.bool)
    starts[:, [0, 6]] = True
    identities = torch.tensor([[3] * 6 + [4] * 7, [5] * 6 + [6] * 7])
    context = builder(columns, actions, segments, starts, identities)
    torch.testing.assert_close(context.features["ego_player_id"], identities, rtol=0, atol=0)
    constant = torch.tensor([3, 5])
    rollout = builder(columns, actions, segments, starts, constant)
    window = builder(columns, actions, segments, starts, constant[:, None].expand_as(identities))
    torch.testing.assert_close(rollout.features["ego_player_id"], window.features["ego_player_id"], rtol=0, atol=0)
    with pytest.raises(ValueError, match="player ids"):
        builder(columns, actions, segments, starts, identities.float())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@torch.inference_mode()
def test_overwriting_every_previous_episode_byte_cannot_change_outputs(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(17)
    builder, columns, _ = _columns(1, device)
    applied = torch.rand(2, 13, 2, ACTION_DIM, device=device)
    segments = torch.tensor([[0] * 5 + [1] * 4 + [2] * 4, [0] * 7 + [1] * 6], device=device)
    resets = torch.ones_like(segments, dtype=torch.bool)
    resets[:, 1:] = segments[:, 1:] != segments[:, :-1]
    cfg = ActionSequenceConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        temporal_d_model=32,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=64,
        group_head_dim=16,
        value_hidden_dim=16,
    )
    model = ActionSequenceTransformer(cfg).to(device).eval()
    positions = torch.tensor([[9, 12], [7, 12]], device=device)
    observed = torch.zeros(2, 2, 4, dtype=torch.long, device=device)
    target = torch.zeros(2, 2, len(cfg.head_offsets), 4, dtype=torch.long, device=device)
    values = torch.full((2, 2), 108.0, device=device)
    present = torch.ones(2, 2, dtype=torch.bool, device=device)

    def outputs():
        ctx = builder(columns, applied, segments, resets, torch.tensor([3, 3], device=device))
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            hidden = model(ctx.features, ctx.ctx_pad, segment_ids=ctx.segment_ids)
            logits = model.temporal.teacher_forced_logits_by_group(
                hidden, segment_starts(segments), positions, observed, target, values, present
            )
        return hidden, logits

    before, logits_before = outputs()
    for row, stop in enumerate((9, 7)):
        for column in columns.values():
            column[row, :stop].view(torch.uint8).fill_(0x41)
        applied[row, :stop].view(torch.uint8).fill_(0x41)
    after, logits_after = outputs()
    for row, start in enumerate((9, 7)):
        torch.testing.assert_close(after[row, start:], before[row, start:], rtol=0, atol=0)
    for name, value in logits_before.items():
        torch.testing.assert_close(logits_after[name], value, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_varlen_general_segments_match_dense_values_and_gradients() -> None:
    cfg = TrunkConfig(d_model=32, n_layers=2, n_heads=4, L_ctx=13)
    trunk = Trunk(cfg).cuda().bfloat16()
    x = torch.randn(2, 13, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    # Labels may recur; only contiguous runs define a segment.
    ids = torch.tensor([[1] * 3 + [4] * 5 + [1] * 5, [7] * 4 + [8] * 9], device="cuda")
    pads = torch.tensor([1, 0], device="cuda")
    cu = segment_cu_seqlens(ids, pads)
    assert cu[-1].item() == 26
    flash = trunk(x, pads, ids)
    dense = trunk.forward_dense(y, pads, ids)
    torch.testing.assert_close(flash, dense, atol=2e-2, rtol=2e-2)
    flash[:, 1:].float().square().mean().backward()
    dense[:, 1:].float().square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=5e-3, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_segmented_trunk_keeps_boundaries_after_device_change() -> None:
    trunk = Trunk(TrunkConfig(d_model=32, n_layers=2, n_heads=4, L_ctx=13))
    values = torch.randn(1, 13, 32)
    pads = torch.zeros(1, dtype=torch.long)
    segments = torch.tensor([[0] * 5 + [1] * 8])
    trunk(values, pads, segments)
    trunk = trunk.cuda().bfloat16()
    values, pads, segments = values.cuda().bfloat16(), pads.cuda(), segments.cuda()
    before = trunk(values, pads, segments)
    values[:, :5].fill_(10)
    after = trunk(values, pads, segments)
    torch.testing.assert_close(after[:, 5:], before[:, 5:], rtol=0, atol=0)
    assert trunk.attn_path == "varlen_flash"
    trunk = trunk.cpu().float()
    actual = trunk(values.cpu().float(), pads.cpu(), segments.cpu())
    expected = trunk.forward_dense(values.cpu().float(), pads.cpu(), segments.cpu())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert trunk.attn_path == "dense"


def test_native_context_rejects_changed_column_dtypes() -> None:
    builder, columns, _ = _columns(1)
    actions = torch.zeros(2, 13, 2, ACTION_DIM)
    segments = torch.zeros(2, 13, dtype=torch.long)
    starts = torch.zeros_like(segments, dtype=torch.bool)
    players = torch.tensor([3, 3])
    columns["p1_stock"] = columns["p1_stock"].float()
    with pytest.raises(ValueError, match="categorical column"):
        builder(columns, actions, segments, starts, players)
    columns["p1_stock"] = columns["p1_stock"].long()
    columns["p1_percent"] = columns["p1_percent"].double()
    with pytest.raises(ValueError, match="float column"):
        builder(columns, actions, segments, starts, players)
    with pytest.raises(ValueError, match="episode starts and segments"):
        builder(columns, actions, segments.float(), starts, players)
