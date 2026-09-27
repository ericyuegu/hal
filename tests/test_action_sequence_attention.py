"""The maintained 059 trunk keeps dense and varlen attention semantics."""

import pytest
import torch

from hal.models.attention import Trunk
from hal.models.attention import TrunkConfig
from hal.models.attention import apply_rotary_emb
from hal.models.attention import dense_mask

_GEOMETRY = dict(d_model=64, n_layers=2, n_heads=4, L_ctx=48)


def _trunk(*, backend: str = "dense_sdpa", window: int = 0) -> Trunk:
    torch.manual_seed(17)
    return Trunk(TrunkConfig(**_GEOMETRY, attention_backend=backend, attn_window=window))


@pytest.mark.parametrize("window", [0, 1, 8, 128])
def test_dense_mask_obeys_causal_window_and_left_padding(window: int) -> None:
    length = _GEOMETRY["L_ctx"]
    padding = torch.tensor([0, 3, 20, length])
    mask = dense_mask(padding, length, window)
    expected = torch.zeros(len(padding), 1, length, length, dtype=torch.bool)
    for batch, pad in enumerate(padding.tolist()):
        for query in range(length):
            for key in range(length):
                in_window = window == 0 or query - key < window
                expected[batch, 0, query, key] = (key <= query and in_window and key >= pad) or key == query
    assert torch.equal(mask, expected)
    assert mask.any(-1).all()


def test_dense_and_unpadded_paths_match_value_and_gradient() -> None:
    trunk = _trunk()
    dense_input = torch.randn(3, _GEOMETRY["L_ctx"], _GEOMETRY["d_model"], requires_grad=True)
    unpadded_input = dense_input.detach().clone().requires_grad_(True)
    dense = trunk.forward_dense(dense_input, torch.zeros(3, dtype=torch.long))
    unpadded = trunk.forward_unpadded(unpadded_input)
    dense.square().mean().backward()
    unpadded.square().mean().backward()
    torch.testing.assert_close(unpadded, dense)
    torch.testing.assert_close(unpadded_input.grad, dense_input.grad)


def test_rotary_table_survives_module_half_cast() -> None:
    rotary = _trunk().blocks[0].attn.rotary
    reference, _ = rotary.at(1024, torch.device("cpu"))
    rounded, _ = rotary.half().at(1024, torch.device("cpu"))
    assert rounded.dtype == torch.float16
    assert (rounded.float() - reference).abs().max() < 1e-3


def test_rotary_preserves_activation_dtype() -> None:
    rotary = _trunk().blocks[0].attn.rotary
    values = torch.zeros(2, 8, 4, rotary.dim, dtype=torch.bfloat16)
    cosine, sine = rotary(values)
    assert cosine.dtype == sine.dtype == values.dtype
    assert apply_rotary_emb(values, cosine, sine).dtype == values.dtype


def test_attention_rejects_invalid_geometry_and_backend() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TrunkConfig(**{**_GEOMETRY, "n_heads": 5})
    with pytest.raises(ValueError, match="head_dim must be even"):
        TrunkConfig(**{**_GEOMETRY, "d_model": 6, "n_heads": 2})
    with pytest.raises(ValueError, match="attn_window"):
        TrunkConfig(**_GEOMETRY, attn_window=-1)
    with pytest.raises(ValueError, match="attention_backend"):
        TrunkConfig(**_GEOMETRY, attention_backend="auto_flex")


def test_invalid_context_padding_cannot_broadcast_across_rows() -> None:
    trunk = _trunk()
    values = torch.randn(4, _GEOMETRY["L_ctx"], _GEOMETRY["d_model"])
    with pytest.raises(ValueError, match="ctx_pad"):
        trunk(values, torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match="ctx_pad"):
        trunk(values, torch.zeros(4, 1, dtype=torch.long))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native varlen FlashAttention requires CUDA")
@pytest.mark.parametrize("window", [0, 8])
def test_varlen_flash_matches_dense_on_valid_tokens_and_gradients(window: int) -> None:
    config = TrunkConfig(**_GEOMETRY, attention_backend="varlen_flash", attn_window=window)
    flash = Trunk(config).cuda().bfloat16()
    dense = Trunk(TrunkConfig(**_GEOMETRY, attention_backend="dense_sdpa", attn_window=window)).cuda().bfloat16()
    dense.load_state_dict(flash.state_dict())
    dense_input = torch.randn(4, config.L_ctx, config.d_model, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    flash_input = dense_input.detach().clone().requires_grad_(True)
    padding = torch.tensor([0, 1, 17, config.L_ctx - 1], device="cuda")
    valid = torch.arange(config.L_ctx, device="cuda")[None] >= padding[:, None]
    dense_output = dense(dense_input, padding)
    flash_output = flash(flash_input, padding)
    torch.testing.assert_close(flash_output[valid], dense_output[valid], rtol=2e-2, atol=2e-2)
    dense_output[valid].float().square().mean().backward()
    flash_output[valid].float().square().mean().backward()
    torch.testing.assert_close(flash_input.grad[valid], dense_input.grad[valid], rtol=5e-2, atol=5e-3)
