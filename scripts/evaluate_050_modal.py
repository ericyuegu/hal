"""Run periodic O50 evaluations on one Modal RTX Pro 6000.

Deploy this production service after its source commit is on ``origin/main``::

    uv run modal deploy scripts/evaluate_050_modal.py

The CPU poller discovers immutable 8,192-update checkpoints in R2. It queues
one 96-boot evaluation for each checkpoint. The GPU function is limited to one
container, so evaluations run in checkpoint order without GPU contention.
"""

import json
import os
import re
import subprocess
import time
from typing import Any
from typing import Final

import boto3
import launch_modal
import modal
from botocore.exceptions import ClientError

_RUN_NAME: Final[str] = (
    "260902-101548_050_scaled_temporal_awr_scaled050-d1024-L16-h16-Lc256-t512x4-"
    "o1-2-3-4-5-6-9-12-16-20-d2r2-linear-head-no-skip-projectiles-mix-r1-cosine-"
    "awr-v-near-b199.5-g0.99618-wu4096_"
)
_APP_NAME: Final[str] = "hal-050-eval96-rtx-pro-6000"
_CHECKPOINT_EVERY: Final[int] = 8192
_FINAL_UPDATE: Final[int] = 131_072
_N_MATCHUPS: Final[int] = 96
_MAX_PARALLEL: Final[int] = 32
_DISK_GIB: Final[int] = 512
_STALE_REQUEST_SECONDS: Final[int] = 8 * 60 * 60
_EXPERIMENT: Final[str] = "experiments/050_scaled_temporal_awr.py"
_CHECKPOINT_PREFIX: Final[str] = f"runs/{_RUN_NAME}/checkpoints/"
_CHECKPOINT_PATTERN: Final[re.Pattern[str]] = re.compile(re.escape(_CHECKPOINT_PREFIX) + r"step-(\d{7})\.pt")

_secret = modal.Secret.from_name("hal", required_keys=list(launch_modal.REQUIRED_SECRET_KEYS))
_cache_volume = modal.Volume.from_name("hal-o50-rtx-pro-6000-eval-cache", create_if_missing=True)
_image = launch_modal._image(launch_modal.IMAGE, ("uv", "run", _EXPERIMENT), _secret)
app = modal.App(_APP_NAME, image=_image)


def _r2_client() -> Any:
    """Create an R2 S3 client from the Modal secret."""
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def _checkpoint_step(key: str) -> int | None:
    """Return the periodic update encoded in an immutable checkpoint key."""
    match = _CHECKPOINT_PATTERN.fullmatch(key)
    if match is None:
        return None
    update = int(match.group(1))
    if update <= 0 or update > _FINAL_UPDATE or update % _CHECKPOINT_EVERY:
        return None
    return update


def _result_prefix(update: int) -> str:
    return f"{_CHECKPOINT_PREFIX}eval96_step_{update:07d}"


def _request_key(update: int) -> str:
    return f"runs/{_RUN_NAME}/eval96_requests/step-{update:07d}.json"


def _read_json(client: Any, key: str) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=os.environ["AWS_BUCKET"], Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    payload = json.loads(response["Body"].read())
    if not isinstance(payload, dict):
        raise ValueError(f"R2 JSON object is not a mapping: {key}")
    return payload


def _write_request(client: Any, update: int, status: str, **extra: Any) -> None:
    payload = {"schema": 1, "update": update, "status": status, "time": time.time(), **extra}
    client.put_object(
        Bucket=os.environ["AWS_BUCKET"],
        Key=_request_key(update),
        Body=json.dumps(payload, sort_keys=True).encode(),
        ContentType="application/json",
    )


def _result_complete(client: Any, update: int) -> bool:
    metrics = _read_json(client, f"{_result_prefix(update)}/metrics.json")
    if metrics is None:
        return False
    return all(int(metrics.get(name, 0)) == _N_MATCHUPS for name in ("scheduled_boots", "completed_boots", "boots"))


def _available_checkpoints(client: Any) -> list[tuple[int, str]]:
    paginator = client.get_paginator("list_objects_v2")
    checkpoints: list[tuple[int, str]] = []
    for page in paginator.paginate(Bucket=os.environ["AWS_BUCKET"], Prefix=_CHECKPOINT_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            update = _checkpoint_step(key)
            if update is not None:
                checkpoints.append((update, key))
    return sorted(checkpoints)


@app.function(
    secrets=[_secret],
    schedule=modal.Period(minutes=5),
    cpu=1.0,
    memory=2048,
    timeout=300,
    max_containers=1,
    serialized=True,
    include_source=False,
)
def poll() -> list[int]:
    """Queue each available checkpoint that has no complete 96-boot result."""
    client = _r2_client()
    queued: list[int] = []
    now = time.time()
    for update, checkpoint_key in _available_checkpoints(client):
        if _result_complete(client, update):
            continue
        request = _read_json(client, _request_key(update))
        if request is not None:
            status = request.get("status")
            age = now - float(request.get("time", 0.0))
            if status in {"complete", "failed"}:
                continue
            if status in {"queued", "running"} and age < _STALE_REQUEST_SECONDS:
                continue
        _write_request(client, update, "queued", checkpoint_key=checkpoint_key)
        evaluate.spawn(update, checkpoint_key)
        queued.append(update)
    print(f"[eval-poll] queued={queued}", flush=True)
    return queued


@app.function(
    secrets=[_secret],
    volumes={str(launch_modal.LOCAL_CACHE_ROOT): _cache_volume},
    gpu="RTX-PRO-6000",
    cpu=32.0,
    memory=64 * 1024,
    ephemeral_disk=_DISK_GIB * 1024,
    timeout=2 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=1.0, backoff_coefficient=2.0),
    max_containers=1,
    serialized=True,
    include_source=False,
)
def evaluate(update: int, checkpoint_key: str) -> None:
    """Evaluate one checkpoint and publish a complete companion W&B row."""
    expected_key = f"{_CHECKPOINT_PREFIX}step-{update:07d}.pt"
    if checkpoint_key != expected_key or _checkpoint_step(checkpoint_key) != update:
        raise ValueError(f"checkpoint does not match update {update}: {checkpoint_key}")

    client = _r2_client()
    _write_request(client, update, "running", checkpoint_key=checkpoint_key)
    try:
        env = launch_modal._prepare_remote(skip_sm120_probe=False)
        output_name = f"eval96_step_{update:07d}"
        command = [
            "uv",
            "run",
            _EXPERIMENT,
            "eval",
            "--checkpoint",
            f"checkpoints/step-{update:07d}.pt",
            "--run",
            _RUN_NAME,
            "--n-matchups",
            str(_N_MATCHUPS),
            "--max-parallel",
            str(_MAX_PARALLEL),
            "--output-name",
            output_name,
            "--companion-wandb",
        ]
        subprocess.run(command, cwd=launch_modal.REMOTE_ROOT, env=env, check=True)
        if not _result_complete(client, update):
            raise RuntimeError(f"evaluation did not publish a complete result for update {update}")
        _cache_volume.commit()
    except Exception as error:
        _write_request(client, update, "failed", checkpoint_key=checkpoint_key, error=repr(error)[:2000])
        raise
    _write_request(client, update, "complete", checkpoint_key=checkpoint_key)


@app.local_entrypoint()
def main() -> None:
    """Poll immediately when the file is run directly."""
    print(poll.remote())
