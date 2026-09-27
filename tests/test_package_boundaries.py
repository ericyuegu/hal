import subprocess
import sys

import pytest


def test_package_root_is_inert_and_submodules_remain_importable() -> None:
    code = """
import sys
import hal
assert 'melee' not in sys.modules
from hal import r2, streams
assert r2 is not None and streams is not None
assert 'melee' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("module", "forbidden"),
    (
        ("hal.models.action_sequence", "melee"),
        ("hal.sim.process_vec", "torch"),
        ("hal.sim.session", "torch"),
        ("hal.representation.observations", "torch"),
    ),
)
def test_model_and_simulator_import_graphs_stay_separate(module: str, forbidden: str) -> None:
    code = f"import sys\nimport {module}\nassert {forbidden!r} not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)
