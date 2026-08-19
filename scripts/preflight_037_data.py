"""Verify actual train-window and validation order across experiment-037 cells."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from hal.training.ego_stats import load_consolidated_stats

_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENT = _ROOT / "experiments" / "037_width_decoder_factorial.py"
_SPEC = importlib.util.spec_from_file_location("hal_exp037_data_preflight", _EXPERIMENT)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


def main() -> None:
    results: dict[str, dict[str, str]] = {}
    for cell in exp._CELL_GEOMETRY:
        cfg = exp._production_config(cell)
        stats = load_consolidated_stats(Path(cfg.data_root) / "stats.json")
        train_loader, validation = exp.base._make_loaders(cfg, stats)
        iterator = iter(train_loader)
        first_batches = [next(iterator) for _ in range(exp._TRAIN_ORDER_PREFLIGHT_BATCHES)]
        results[cell] = {
            "config_sha256": exp.config_sha256(cfg),
            "sampling_contract_sha256": exp.sampling_contract_sha256(cfg),
            "train_order_first_two_batches_sha256": exp.train_order_sha256(first_batches),
            "validation_cache_sha256": exp.validation_cache_sha256(validation),
        }
        del iterator
        del train_loader
    train_hashes = {values["train_order_first_two_batches_sha256"] for values in results.values()}
    validation_hashes = {values["validation_cache_sha256"] for values in results.values()}
    sampling_hashes = {values["sampling_contract_sha256"] for values in results.values()}
    if len(train_hashes) != 1 or len(validation_hashes) != 1 or len(sampling_hashes) != 1:
        raise RuntimeError(f"factorial data-order mismatch: {results}")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
