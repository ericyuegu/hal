import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from hal.inference.o50_model import BASE_PLAYER_PREFIXES
from hal.inference.o50_model import CAT_FEATURES
from hal.inference.o50_model import CONTROLLER_GROUP_COUNT
from hal.inference.o50_model import FLOAT_FEATURES
from hal.inference.o50_model import ITEM_FLOATS
from hal.inference.o50_model import ITEM_SLOTS
from hal.inference.o50_model import O50Config
from hal.inference.o50_model import O50Model
from hal.inference.o50_model import item_column


def _experiment():
    path = Path(__file__).parents[1] / "experiments" / "050_scaled_temporal_awr.py"
    spec = importlib.util.spec_from_file_location("o50_parity_experiment", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_portable_model_matches_frozen_o50_trunk_and_decoder() -> None:
    experiment = _experiment()
    architecture = {
        **asdict(experiment.Architecture()),
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 4,
        "L_ctx": 8,
        "sample_chunk_length": 4,
        "head_offsets": (1, 2, 3, 4),
        "temporal_d_model": 32,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "temporal_ff_dim": 64,
        "group_head_dim": 32,
        "value_hidden_dim": 16,
        "item_hidden_dim": 8,
        "item_dim": 5,
    }
    vocabulary = experiment.PlayerVocabulary(("IBDW#0",))
    training_config = experiment.TrainConfig(
        arch=experiment.Architecture(**architecture),
        player_vocab_size=vocabulary.size,
        player_vocab_sha256=vocabulary.sha256,
        amp_dtype="float32",
        compile_trunk=False,
        compile_temporal=False,
        inference_mode="eager",
        num_workers=0,
        push_to_r2=False,
    )
    torch.manual_seed(11)
    frozen = experiment.GPT(training_config, vocabulary)
    checkpoint_config = asdict(training_config)
    checkpoint_config["architecture"] = checkpoint_config.pop("arch")
    checkpoint_config["experiment_id"] = "050_scaled_temporal_awr_v6"
    portable_config = O50Config.from_checkpoint(
        checkpoint_config,
        player_code_bytes=frozen.player_code_bytes.numel(),
        stats_sha256="0" * 64,
    )
    portable = O50Model(portable_config)
    portable.load_state_dict(frozen.state_dict(), strict=True)

    batch, length = 2, portable_config.architecture.L_ctx
    features: dict[str, torch.Tensor] = {}
    for prefix in BASE_PLAYER_PREFIXES:
        for name in FLOAT_FEATURES:
            features[f"{prefix}_{name}"] = torch.randn(batch, length)
            features[f"{prefix}_{name}_mask"] = torch.zeros(batch, length)
        for name, (vocabulary_size, _width) in CAT_FEATURES.items():
            features[f"{prefix}_{name}"] = torch.randint(vocabulary_size, (batch, length))
    for name in experiment.ACTION_CHANNELS:
        values = torch.rand(batch, length)
        if name.startswith("button_"):
            values = values.round()
        elif "stick" in name:
            values = 2 * values - 1
        features[f"ego_{name}"] = values
    features["ego_character"] = torch.randint(32, (batch, length))
    features["opp_character"] = torch.randint(32, (batch, length))
    features["stage"] = torch.randint(32, (batch, length))
    features["ego_player_id"] = torch.randint(vocabulary.size, (batch, length))
    for slot in range(ITEM_SLOTS):
        features[item_column(slot, "type")] = torch.zeros(batch, length, dtype=torch.long)
        features[item_column(slot, "state")] = torch.zeros(batch, length, dtype=torch.long)
        for name in ITEM_FLOATS:
            column = item_column(slot, name)
            features[column] = torch.zeros(batch, length)
            features[f"{column}_mask"] = torch.ones(batch, length)

    context_padding = torch.tensor([0, 3])
    frozen_observed = frozen.codec.quantize(experiment.stack_actions(features))
    portable_observed = portable.codec.quantize(experiment.stack_actions(features))
    frozen_hidden = frozen.forward_dense(features, context_padding, frozen_observed)
    portable_hidden = portable.forward_dense(features, context_padding, portable_observed)
    assert torch.equal(frozen_hidden, portable_hidden)

    uniforms = torch.rand(4, CONTROLLER_GROUP_COUNT, batch)
    frozen_indices = frozen.temporal.sample_indices(
        frozen_hidden,
        frozen_observed[:, -1],
        (1, 2, 3, 4),
        argmax=False,
        uniforms=uniforms,
    )
    portable_indices = portable.temporal.sample_conditioned(
        portable_hidden,
        portable_observed[:, -1],
        torch.empty(batch, 0, CONTROLLER_GROUP_COUNT, dtype=torch.long),
        uniforms,
    )
    assert torch.equal(frozen_indices, portable_indices)
