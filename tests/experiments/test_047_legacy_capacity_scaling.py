"""Contracts for O47 legacy-codec capacity scaling."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "experiments" / "047_legacy_capacity_scaling.py"
    spec = importlib.util.spec_from_file_location("test_exp047", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exp = _load()


EXPECTED = {
    "c5e16-xs": (2_506_223, 8_078_474, 15_740, 1_031_536_640, 49_999_451_557_724_160),
    "c5e16-s": (4_722_415, 10_572_874, 12_027, 788_201_472, 50_001_329_100_423_168),
    "c5e16-m": (9_445_359, 15_574_026, 8_165, 535_101_440, 50_002_102_435_184_640),
    "cref-s": (4_722_415, 10_572_874, 33_255, 2_179_399_680, 138_255_109_273_681_920),
    "cref-m": (9_445_359, 15_574_026, 22_576, 1_479_540_736, 138_254_435_343_138_816),
    "cref-reference": (15_053_039, 21_459_914, 16_384, 1_073_741_824, 138_254_443_207_458_816),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_endpoint_has_exact_model_data_and_compute_quantities(name: str) -> None:
    report = exp.endpoint_report(exp.POINTS[name])
    actual = (
        report["total_parameters"],
        report["effective_parameters"],
        report["max_steps"],
        report["processed_positions"],
        report["actual_flops"],
    )
    assert actual == EXPECTED[name]
    assert report["processed_positions"] < report["corpus_loss_positions"]


@pytest.mark.parametrize("name", exp.TRAIN_ENDPOINTS)
def test_endpoint_preserves_complete_o43_treatment(name: str) -> None:
    cfg = exp.endpoint_config(exp.POINTS[name])
    reference = exp._o43.TrainConfig()

    assert cfg.codec_version == reference.codec_version == 2
    assert cfg.head_offsets == reference.head_offsets
    assert cfg.next_frame_loss_share == reference.next_frame_loss_share == 0.5
    assert cfg.group_order == reference.group_order
    assert cfg.data_root == reference.data_root
    assert cfg.batch_size == reference.batch_size == 512
    assert cfg.L_ctx == reference.L_ctx == 128
    assert cfg.seed == reference.seed == 0
    assert cfg.eval_every == cfg.final_diag_n_matchups == 0
    assert cfg.final_eval_n_matchups == 96
    exp._o43.validate_config(cfg)
    assert exp.endpoint_for_config(cfg) == exp.POINTS[name]


def test_reference_point_is_not_launchable() -> None:
    assert "cref-reference" not in exp.TRAIN_ENDPOINTS
    assert exp.POINTS["cref-reference"].reference_run_id == "1imfy8v3"
    with pytest.raises(SystemExit, match="--endpoint must be one of"):
        exp._selected_endpoint("cref-reference")
