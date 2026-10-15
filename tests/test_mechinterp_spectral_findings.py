import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "research" / "mechinterp_20260911" / "spectral_findings.py"
_SPEC = importlib.util.spec_from_file_location("mechinterp_spectral_findings", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
findings = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = findings
_SPEC.loader.exec_module(findings)


def test_spectral_grouping_does_not_count_fused_qkv_twice() -> None:
    assert findings.matrix_group("trunk.blocks.0.attn.c_attn.weight") is None
    assert findings.matrix_group("trunk.blocks.0.attn.c_attn.weight/q") == "Trunk Q"
    assert findings.matrix_group("trunk.blocks.0.attn.c_attn.weight/v") == "Trunk V"
    assert findings.matrix_group("temporal.outputs.main_stick.down.weight") == "Action readout"


def test_spectral_summary_quantiles_describe_matrices_not_bootstraps() -> None:
    matrices = (
        findings.Matrix("trunk.blocks.0.attn.c_attn.weight/q", 2, 3, 1, None, None),
        findings.Matrix("trunk.blocks.1.attn.c_attn.weight/q", 4, 5, 2, None, None),
    )
    result = findings.group_summary(findings.Checkpoint("muon", 10, matrices))["Trunk Q"]
    assert result["matrices"] == 2
    assert result["frobenius"] == {"q25": 2.5, "median": 3.0, "q75": 3.5}
    assert result["angle_degrees"] is None


@pytest.mark.parametrize("step", [2.5, True, -1])
def test_spectral_checkpoint_step_is_strict(tmp_path: Path, step: object) -> None:
    path = tmp_path / "spectra.json"
    path.write_text(
        json.dumps({"muon": {"checkpoints": [{"metadata": {"checkpoint_step": step}}]}, "adamw": {"checkpoints": []}})
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        findings.load_checkpoints(path)
