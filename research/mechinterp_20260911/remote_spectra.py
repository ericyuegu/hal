"""Run the checkpoint spectral audit beside its validated Modal Volume exports."""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import modal

SOURCE_SHA = "52da5b2de1d402b86404fc4d24e382409649e525"
IMAGE_TAG = "ghcr.io/ericyuegu/hal:cuda13"
VOLUME_NAME = "hal-mechinterp-20260911"
app = modal.App("hal-mechinterp-spectra-20260911")
worker_image = modal.Image.from_registry(IMAGE_TAG).workdir("/workspace")
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    worker_image = worker_image.add_local_dir(root / "hal", "/workspace/hal").add_local_file(
        Path(__file__).with_name("spectra.py"), "/workspace/spectra.py"
    )
checkpoints = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@app.function(image=worker_image, gpu="H100", cpu=8, memory=32768, timeout=1800, volumes={"/checkpoints": checkpoints})
def inspect_remote() -> dict[str, object]:
    root = Path("/checkpoints")
    metadata = json.loads((root / "model_exports.json").read_bytes())
    by_name = {record["output"]: record for record in metadata["exports"]}
    specifications = {
        "muon": (("p1fyyp1z", 24576), ("p1fyyp1z", 57344), ("p1fyyp1z", 98304), ("p1fyyp1z", 131072)),
        "adamw": (("tgpuo1be", 57344), ("tgpuo1be", 131072)),
    }
    output: dict[str, object] = {
        "image_tag": IMAGE_TAG,
        "volume": VOLUME_NAME,
        "source_sha": SOURCE_SHA,
        "source_files": {
            str(path.relative_to("/workspace")): file_sha256(path)
            for path in sorted(Path("/workspace/hal").rglob("*.py"))
        },
    }
    with tempfile.TemporaryDirectory(prefix="hal-mechinterp-spectra-") as temporary:
        for optimizer, checkpoints_for_run in specifications.items():
            result_path = Path(temporary) / f"{optimizer}.json"
            command = [
                sys.executable,
                "/workspace/spectra.py",
                "--device",
                "cuda",
                "--analysis-git-sha",
                SOURCE_SHA,
                "--output",
                str(result_path),
            ]
            for run_id, step in checkpoints_for_run:
                name = f"{run_id}-step-{step:07d}.model.pt"
                path = root / name
                record = by_name[name]
                if path.stat().st_size != record["bytes"] or file_sha256(path) != record["output_sha256"]:
                    raise ValueError(f"export failed its size/SHA validation: {name}")
                command.extend(("--checkpoint", f"{optimizer}_{step}={path}"))
            subprocess.run(command, check=True, cwd="/workspace")
            output[optimizer] = json.loads(result_path.read_bytes())
    return output


@app.local_entrypoint()
def main(output_dir: str = "research/mechinterp_20260911/artifacts/spectra") -> None:
    result = inspect_remote.remote()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "historical_spectra.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {path}; raw checkpoints stayed on the research Volume.")
