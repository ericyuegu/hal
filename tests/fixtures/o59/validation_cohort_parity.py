"""Capture or compare the 059 validation cohort before collation loses identity.

Use ``--mode control`` with the recorded Git checkout to recreate
``validation_cohort.json``. Use ``--mode candidate`` against the maintained
checkout to compare all 2,048 identities, window digests, and batch-row digests.
Both modes need the published local corpus, player sidecar, and R2 access.
"""

import argparse
import configparser
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import CodeType
from types import FrameType
from types import ModuleType
from typing import Any
from typing import cast

import numpy as np
import torch


def _r2_environment(config_path: Path) -> None:
    if all(os.environ.get(name) for name in ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")):
        return
    config = configparser.ConfigParser()
    if not config.read(config_path) or "r2" not in config:
        raise ValueError(f"R2 credentials are absent from {config_path}")
    r2 = config["r2"]
    os.environ["AWS_ENDPOINT_URL"] = r2["endpoint"]
    os.environ["AWS_ACCESS_KEY_ID"] = r2["access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = r2["secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = "auto"


def _experiment(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load experiment {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _window_digest(window: Mapping[str, np.ndarray | np.integer]) -> str:
    digest = hashlib.sha256()
    for name in sorted(window):
        value = np.asarray(window[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _tensor_row_digest(batch: Any, index: int) -> str:
    values = {
        "ctx_pad": batch.context.ctx_pad,
        "target": batch.target,
        "returns": batch.returns,
        "eligible": batch.eligible,
        "future_return": batch.future_return,
        "available": batch.available,
        "condition_present": batch.condition_present,
    }
    values.update({f"feature/{name}": value for name, value in batch.context.features.items()})
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        row = tensor[index].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(row.dtype).encode())
        digest.update(str(tuple(row.shape)).encode())
        digest.update(row.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class _ControlWindowCapture:
    """Read the old loader's replay identity at the make_window call boundary."""

    def __init__(self, make_window_code: CodeType, context_length: int) -> None:
        self.make_window_code = make_window_code
        self.context_length = context_length
        self.records: list[dict[str, object]] = []

    def __call__(self, frame: FrameType, event: str, value: object) -> None:
        if event != "return" or frame.f_code is not self.make_window_code:
            return
        parent = frame.f_back
        if parent is None or not isinstance(value, dict):
            raise ValueError("control window capture lost the loader call frame")
        replay_id = parent.f_locals.get("replay_id")
        if not isinstance(replay_id, str):
            raise ValueError("control validation window has no replay identity")
        pad = int(parent.f_locals["pad"])
        window = dict(cast(dict[str, np.ndarray | np.integer], value))
        window["ctx_pad"] = np.int64(min(pad, self.context_length))
        self.records.append(
            {
                "replay_id": replay_id,
                "start": int(parent.f_locals["start"]),
                "ego_prefix": parent.f_locals["ego_prefix"],
                "window_sha256": _window_digest(window),
            }
        )


def _source_manifest(repo: Path, sources: tuple[Any, ...]) -> list[dict[str, object]]:
    manifest = []
    for source in sources:
        index_path = repo / source.local / "val/index.json"
        index_bytes = index_path.read_bytes()
        index = json.loads(index_bytes)
        manifest.append(
            {
                "name": source.name,
                "remote": source.remote,
                "val_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
                "val_rows": sum(int(shard["samples"]) for shard in index["shards"]),
                "stats_sha256": hashlib.sha256((repo / source.local / "stats.json").read_bytes()).hexdigest(),
            }
        )
    return manifest


def _control(repo: Path, checkout: Path, output: Path) -> None:
    from hal import streams
    from hal.training import dataloader
    from hal.training import returns as returns_lib
    from hal.training.features import FeatureProjection
    from hal.training.player_identity import ReplayPlayerLookup

    streams.REPO_DIR = str(repo)
    experiment = _experiment(checkout / "experiments/059_muon_history_decoder.py", "o59_control")
    cfg = experiment.TrainConfig(
        player_sidecar_local=str(repo / "data/processed/player-identity-v1/professional-code-v1.jsonl.gz")
    )
    stats = experiment.load_stats(cfg)
    sidecar = experiment.load_identity_sidecar(cfg)
    projection = FeatureProjection(
        columns=experiment.ITEM_PLAYER_PROJECTION.columns
        | {
            cfg.awr.ego_return_column,
            cfg.awr.ego_return_valid_column,
            "ego_return60",
            "ego_return60_valid",
        },
        derive_spatial=experiment.ITEM_PLAYER_PROJECTION.derive_spatial,
    )
    labels = experiment.ReturnLabels(
        returns_lib.PolicyReturnLabels(
            ReplayPlayerLookup(sidecar.by_replay),
            cfg.awr.gamma,
            cfg.awr.damage_shaping,
            cfg.awr.win_reward,
            cfg.awr.stock_value,
            cfg.awr.return_suffix,
        )
    )
    loader = dataloader.make_loader(
        data_root=None,
        split=cfg.val_split,
        stats=stats,
        L_ctx=cfg.arch.L_ctx,
        L_chunk=cfg.arch.sample_chunk_length,
        batch_size=cfg.val_batch_size,
        seed=cfg.seed,
        sources=tuple(streams.BY_NAME[name] for name in cfg.source_names),
        cache_limit="1792gb",
        shuffle_block_size=8192,
        shuffle_seed=cfg.seed,
        num_workers=0,
        schema_version=cfg.mds_schema_version,
        extra=experiment.ITEM_PLAYER_COLUMNS,
        projection=projection,
        batch_transform=functools.partial(experiment.collate_awr_batch, L_ctx=cfg.arch.L_ctx),
        replay_format="policy-world",
        replay_labels=labels,
        require_full_context=True,
        shuffle=True,
    )

    capture = _ControlWindowCapture(dataloader.make_window.__code__, cfg.arch.L_ctx)
    previous_profile = sys.getprofile()
    transformed: list[str] = []
    started = time.monotonic()
    try:
        sys.setprofile(capture)
        for batch in loader:
            transformed.extend(_tensor_row_digest(batch, index) for index in range(len(batch.target)))
            if len(transformed) >= cfg.val_n_samples:
                break
    finally:
        sys.setprofile(previous_profile)
    if len(transformed) != cfg.val_n_samples or len(capture.records) != cfg.val_n_samples:
        raise ValueError("control validation capture did not produce exactly 2,048 windows")
    for row, digest in zip(capture.records, transformed, strict=True):
        row["tensor_sha256"] = digest
    result = {
        "schema": 1,
        "control_git_sha": subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "player_sidecar_sha256": cfg.player_sidecar_sha256,
        "player_vocab_sha256": cfg.player_vocab_sha256,
        "seed": cfg.seed,
        "shuffle_algo": "py1e",
        "shuffle_block_size": 8192,
        "context_length": cfg.arch.L_ctx,
        "chunk_length": cfg.arch.sample_chunk_length,
        "batch_size": cfg.val_batch_size,
        "samples": cfg.val_n_samples,
        "sources": _source_manifest(repo, streams.POLICY_WORLD_V8_SOURCES),
        "windows": capture.records,
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": len(capture.records),
                "elapsed_s": round(time.monotonic() - started, 2),
                "source_count": len(result["sources"]),
                "cohort_sha256": hashlib.sha256(
                    json.dumps(capture.records, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            },
            sort_keys=True,
        )
    )


def _candidate(repo: Path, fixture: Path, output: Path | None) -> None:
    from hal import streams
    from hal.training import returns as returns_lib
    from hal.training.validation_replay_loader import make_validation_replay_loader

    streams.REPO_DIR = str(repo)
    experiment = _experiment(repo / "experiments/059_muon_action_sequence.py", "o59_candidate")
    cfg = experiment.TrainConfig(
        player_sidecar_local=str(repo / "data/processed/player-identity-v1/professional-code-v1.jsonl.gz")
    )
    stats = experiment.load_stats(cfg)
    sidecar = experiment.load_identity_sidecar(cfg)
    loader = make_validation_replay_loader(
        sources=tuple(streams.BY_NAME[name] for name in cfg.source_names),
        stats=stats,
        context_length=cfg.arch.L_ctx,
        chunk_length=cfg.arch.sample_chunk_length,
        batch_size=cfg.val_batch_size,
        seed=cfg.seed,
        cache_limit="1792gb",
        schema_version=cfg.mds_schema_version,
        extra=experiment.ITEM_PLAYER_COLUMNS,
        projection=experiment.FeatureProjection(
            columns=experiment.ITEM_PLAYER_PROJECTION.columns
            | {
                cfg.awr.ego_return_column,
                cfg.awr.ego_return_valid_column,
                "ego_return60",
                "ego_return60_valid",
            }
        ),
        batch_transform=functools.partial(experiment.collate_awr_batch, L_ctx=cfg.arch.L_ctx),
        replay_labels=experiment.ReturnLabels(
            returns_lib.PolicyReturnLabels(
                experiment.ReplayPlayerLookup(sidecar.by_replay),
                cfg.awr.gamma,
                cfg.awr.damage_shaping,
                cfg.awr.win_reward,
                cfg.awr.stock_value,
                cfg.awr.return_suffix,
            )
        ),
    )
    transformed: list[str] = []
    started = time.monotonic()
    for batch in loader:
        transformed.extend(_tensor_row_digest(batch, index) for index in range(len(batch.target)))
        if len(transformed) >= cfg.val_n_samples:
            break
    records = [dataclasses.asdict(identity) for identity in loader.dataset.identities]
    if len(records) != len(transformed) or len(records) != cfg.val_n_samples:
        raise ValueError("candidate validation did not produce exactly 2,048 windows")
    for record, digest in zip(records, transformed, strict=True):
        record["tensor_sha256"] = digest
    expected = json.loads(fixture.read_text())["windows"]
    mismatches = [
        (index, old, new) for index, (old, new) in enumerate(zip(expected, records, strict=True)) if old != new
    ]
    report = {
        "rows": len(records),
        "mismatches": len(mismatches),
        "elapsed_s": round(time.monotonic() - started, 2),
        "cohort_sha256": hashlib.sha256(
            json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    if output is not None:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))
    for index, old, new in mismatches[:5]:
        print(json.dumps({"index": index, "control": old, "candidate": new}, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("control", "candidate"), required=True)
    parser.add_argument("--repo", type=Path, required=True, help="checkout with local published corpus and sidecar")
    parser.add_argument("--control-checkout", type=Path)
    parser.add_argument("--fixture", type=Path, default=Path(__file__).with_name("validation_cohort.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rclone-config", type=Path, default=Path.home() / ".config/rclone/rclone.conf")
    args = parser.parse_args()
    repo = args.repo.resolve()
    source = args.control_checkout.resolve() if args.mode == "control" and args.control_checkout else repo
    if args.mode == "control" and (args.control_checkout is None or args.output is None):
        parser.error("control mode requires --control-checkout and --output")
    sys.path.insert(0, str(source))
    _r2_environment(args.rclone_config)
    import hal

    if not Path(hal.__file__).resolve().is_relative_to(source):
        raise RuntimeError(f"imported HAL source does not come from {source}")
    if args.mode == "control":
        _control(repo, source, args.output)
    else:
        _candidate(repo, args.fixture, args.output)


if __name__ == "__main__":
    main()
