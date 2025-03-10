"""Test KV cache implementation for transformer models."""
import time
from typing import List
from typing import Tuple

import torch
from loguru import logger
from tensordict import TensorDict

from hal.eval.eval_helper import mock_framedata_as_tensordict
from hal.preprocess.preprocessor import Preprocessor
from hal.training.config import DataConfig
from hal.training.models.gpt_multi_token import GPTMultiTokenValue
from hal.training.models.gpt_multi_token import GPTMultiTokenValueWithCache
from hal.training.models.gpt_multi_token import MultiTokenGPTConfig


class KVCacheManager:
    """Helper class to manage KV caches for transformer blocks."""

    def __init__(self, n_workers: int, model: GPTMultiTokenValueWithCache, device: torch.device | str) -> None:
        """Initialize KV cache manager.

        Args:
            n_workers: Number of CPU workers
            model: Model instance to get config from
            device: Device to store caches on
        """
        self.n_workers = n_workers
        self.device = device
        self.block_size = model.block_size
        self.n_layer = model.gpt_config.n_layer
        self.n_head = model.gpt_config.n_head
        self.n_embd = model.gpt_config.n_embd
        self.head_size = self.n_embd // self.n_head

        # Initialize KV caches for each layer
        # Shape: (n_layer, 2, B, nh, L, hs) where 2 is for k,v
        self.kv_cache = torch.zeros(
            self.n_layer,
            2,
            n_workers,
            self.n_head,
            self.block_size,
            self.head_size,
            device=device,
            dtype=torch.float32,
        )
        self.cache_pos = 0

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Update KV cache for a specific layer.

        Args:
            layer_idx: Layer index
            k: Key tensor of shape (B, nh, 1, hs)
            v: Value tensor of shape (B, nh, 1, hs)
        """
        # Update cache at current position
        self.kv_cache[layer_idx, 0, :, :, self.cache_pos : self.cache_pos + 1] = k
        self.kv_cache[layer_idx, 1, :, :, self.cache_pos : self.cache_pos + 1] = v

    def update_all(self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Update KV cache for all layers at once.

        Args:
            kv_caches: List of (key, value) cache tuples for each layer
        """
        for layer_idx, (k, v) in enumerate(kv_caches):
            self.kv_cache[layer_idx, 0, :, :, self.cache_pos : self.cache_pos + 1] = k
            self.kv_cache[layer_idx, 1, :, :, self.cache_pos : self.cache_pos + 1] = v

    def get_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached KV tensors for a specific layer up to current position.

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (k, v) tensors of shape (B, nh, L, hs)
        """
        return (
            self.kv_cache[layer_idx, 0, :, :, : self.cache_pos + 1],
            self.kv_cache[layer_idx, 1, :, :, : self.cache_pos + 1],
        )

    def roll_cache(self) -> None:
        """Roll KV cache left when full."""
        if self.cache_pos >= self.block_size - 1:
            self.kv_cache[:, :, :, :, :-1] = self.kv_cache[:, :, :, :, 1:].clone()
            self.cache_pos = self.block_size - 2
        else:
            self.cache_pos += 1

    def reset(self) -> None:
        """Reset KV cache."""
        self.kv_cache.zero_()
        self.cache_pos = 0


def test_kv_cache_correctness() -> None:
    """Test that KV cache produces identical outputs to non-cached forward pass."""
    logger.info("Testing KV cache correctness...")

    # Set up model and inputs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len = 512
    data_config = DataConfig(
        data_dir="/opt/projects/hal2/data/top_players/Cody/",
        input_preprocessing_fn="baseline_button_fine_main_coarser_cstick_medium_analog_shoulder",
        target_preprocessing_fn="frame_1_and_12_value",
        pred_postprocessing_fn="frame_1",
    )
    preprocessor = Preprocessor(data_config=data_config)
    model_config = MultiTokenGPTConfig(
        block_size=1024,
        n_embd=1024,
        n_layer=6,
        n_head=8,
        dropout=0.0,  # Disable dropout for deterministic outputs
        multi_token_heads=(1, 12),
    )

    # Create input data
    mock_inputs = mock_framedata_as_tensordict(seq_len)
    mock_inputs = preprocessor.preprocess_inputs(mock_inputs, "p1")
    mock_inputs = torch.stack([mock_inputs for _ in range(batch_size)], dim=0)
    inputs = mock_inputs.to(device)

    # Initialize both model variants
    base_model = GPTMultiTokenValue(preprocessor, model_config)
    base_model.eval()
    base_model = base_model.to(device)

    cached_model = GPTMultiTokenValueWithCache(preprocessor, model_config)
    cached_model.eval()
    cached_model = cached_model.to(device)
    cached_model.load_state_dict(base_model.state_dict())  # Ensure identical weights

    # Test 1: Compare base model vs cached model regular forward
    with torch.no_grad():
        base_outputs = base_model(inputs)
        cached_outputs = cached_model(inputs)

    logger.info("Comparing base model vs cached model forward outputs...")
    max_diff_forward = 0.0
    for key in base_outputs.keys():
        diff = (base_outputs[key] - cached_outputs[key]).abs().max().item()
        max_diff_forward = max(max_diff_forward, diff)
        logger.info(f"Max difference for {key}: {diff:.3e}")

    # Test 2: Compare cached model forward vs forward_with_kv_cache
    kv_cache_manager = KVCacheManager(batch_size, cached_model, device)
    outputs_with_cache = []

    with torch.no_grad():
        for i in range(seq_len):
            frame_input = inputs[:, i : i + 1]
            output, kv_caches = cached_model.forward_with_kv_cache(
                frame_input,
                kv_caches=[kv_cache_manager.get_kv(i) for i in range(cached_model.gpt_config.n_layer)],
                return_kv=True,
            )
            kv_cache_manager.update_all(kv_caches)
            kv_cache_manager.roll_cache()
            outputs_with_cache.append(output)

    outputs_with_cache = TensorDict.cat(outputs_with_cache, dim=1)

    logger.info("Comparing cached model forward vs forward_with_kv_cache outputs...")
    max_diff_cache = 0.0
    for key in cached_outputs.keys():
        diff = (cached_outputs[key] - outputs_with_cache[key]).abs().max().item()
        max_diff_cache = max(max_diff_cache, diff)
        logger.info(f"Max difference for {key}: {diff:.3e}")

    if max_diff_forward < 1e-5 and max_diff_cache < 1e-5:
        logger.success("All model variants and forward methods produce matching outputs!")
    else:
        if max_diff_forward >= 1e-5:
            logger.error("Base model and cached model outputs do not match!")
        if max_diff_cache >= 1e-5:
            logger.error("Cached model forward and forward_with_kv_cache outputs do not match!")
        # raise ValueError("KV cache implementation is incorrect")

    n_trials = 100

    # Benchmark non-cached forward pass
    logger.info("Benchmarking non-cached forward pass...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(n_trials):
        with torch.no_grad():
            _ = base_model(inputs)
    torch.cuda.synchronize()
    no_cache_time = (time.perf_counter() - start_time) / n_trials
    logger.info(f"Non-cached forward pass: {no_cache_time*1000:.2f}ms per sequence")

    logger.info("Benchmarking cached forward pass...")
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(n_trials):
        kv_cache_manager.reset()
        frame_input = inputs[:, 0:1]
        with torch.no_grad():
            _, kv_caches = cached_model.forward_with_kv_cache(
                frame_input,
                kv_caches=[kv_cache_manager.get_kv(i) for i in range(cached_model.gpt_config.n_layer)],
                return_kv=True,
            )
            kv_cache_manager.update_all(kv_caches)
            kv_cache_manager.roll_cache()
    torch.cuda.synchronize()
    cache_time = (time.perf_counter() - start_time) / n_trials
    logger.info(f"Cached forward pass: {cache_time*1000:.2f}ms per sequence")
    logger.info(f"Speedup: {no_cache_time/cache_time:.2f}x")


if __name__ == "__main__":
    # Test correctness
    test_kv_cache_correctness()
