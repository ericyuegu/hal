import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "hal_launch_vast", Path(__file__).parents[1] / "scripts" / "launch_vast.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Args = _MODULE.Args
build_query = _MODULE.build_query


def test_build_query_can_select_one_compute_capability() -> None:
    query = build_query(
        max_price=1.0,
        disk=500,
        min_vram=24,
        min_ram=128,
        min_dlperf=35.0,
        min_compute_cap=1200,
        max_compute_cap=1200,
    )

    terms = set(query.split())
    assert "compute_cap>=1200" in terms
    assert "compute_cap<=1200" in terms


def test_compute_capability_bounds_are_disabled_by_default() -> None:
    cfg = Args()
    query = build_query(
        max_price=cfg.max_price,
        disk=cfg.disk,
        min_vram=cfg.min_vram,
        min_ram=cfg.min_ram,
        min_dlperf=cfg.min_dlperf,
        min_compute_cap=cfg.min_compute_cap,
        max_compute_cap=cfg.max_compute_cap,
    )

    assert "compute_cap" not in query
