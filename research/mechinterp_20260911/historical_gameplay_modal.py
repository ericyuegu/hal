"""Modal H100 wrapper for the bounded historical O50 gameplay runner.

This module defines the job but does not launch it. A launch uploads the listed
HAL and research source files, so it requires explicit approval.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import modal
import torch

_IMAGE_TAG = "ghcr.io/ericyuegu/hal:cuda13"
_VOLUME_NAME = "hal-mechinterp-20260911"
_APP_NAME = "hal-historical-gameplay-20260911"

app = modal.App(_APP_NAME)
image = modal.Image.from_registry(_IMAGE_TAG).workdir("/workspace")
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    research_root = Path(__file__).parent
    image = (
        image.add_local_dir(root / "hal", "/workspace/hal")
        .add_local_file(
            research_root / "checkpoint_interventions.py",
            "/workspace/research/mechinterp_20260911/checkpoint_interventions.py",
        )
        .add_local_file(
            research_root / "historical_gameplay.py",
            "/workspace/research/mechinterp_20260911/historical_gameplay.py",
        )
    )
checkpoints = modal.Volume.from_name(_VOLUME_NAME, create_if_missing=False)
secret = modal.Secret.from_name("hal")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(
    image=image,
    secrets=[secret],
    gpu="H100",
    cpu=32.0,
    memory=64 * 1024,
    ephemeral_disk=10 * 1024,
    timeout=60 * 60,
    max_containers=1,
    single_use_containers=True,
    include_source=False,
    volumes={"/checkpoints": checkpoints},
)
def evaluate_remote(
    intervention: str,
    protocol_name: str,
    confirm_volume_write: bool,
) -> dict[str, object]:
    """Run one three-condition protocol and preserve replays on the mounted Volume."""
    from research.mechinterp_20260911 import historical_gameplay

    if not confirm_volume_write:
        raise ValueError("persistent replay output requires explicit Volume-write approval")
    try:
        protocol = historical_gameplay.PROTOCOLS[protocol_name]
    except KeyError as error:
        raise ValueError(f"unknown gameplay protocol {protocol_name!r}") from error
    root = Path("/checkpoints")
    early = root / "p1fyyp1z-step-0024576.model.pt"
    slump = root / "p1fyyp1z-step-0057344.model.pt"
    replay_root = root / "gameplay_replays" / protocol_name / intervention
    with tempfile.TemporaryDirectory(prefix="hal-gameplay-result-") as temporary:
        output = Path(temporary) / "result.json"
        historical_gameplay.run_gameplay(
            early,
            slump,
            intervention,
            protocol,
            output,
            replay_root,
            device=torch.device("cuda"),
            execution=historical_gameplay.CudaExecution(target="modal_h100", compiled=True),
        )
        result = json.loads(output.read_bytes())
    result["execution"] = {
        "app": _APP_NAME,
        "checkpoint_volume": _VOLUME_NAME,
        "image": _IMAGE_TAG,
        "source_sha256": {
            "checkpoint_interventions.py": _sha256(
                Path("/workspace/research/mechinterp_20260911/checkpoint_interventions.py")
            ),
            "historical_gameplay.py": _sha256(Path("/workspace/research/mechinterp_20260911/historical_gameplay.py")),
        },
    }
    return result


@app.local_entrypoint()
def main(
    intervention: str,
    output: str,
    protocol: str = "short32",
    confirm_source_upload: bool = False,
    confirm_volume_write: bool = False,
) -> None:
    """Launch only after approval to upload the mounted source snapshot."""
    if not confirm_source_upload:
        raise SystemExit("launch requires --confirm-source-upload after explicit approval")
    if not confirm_volume_write:
        raise SystemExit("launch requires --confirm-volume-write after explicit approval")
    if protocol == "historical96":
        raise SystemExit("the 96-boot protocol is not enabled by this bounded entrypoint")
    result = evaluate_remote.remote(intervention, protocol, confirm_volume_write)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False, sort_keys=True) + "\n")
    print(f"saved {output_path}")
