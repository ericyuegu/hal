import dataclasses
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("o65_delivery", Path(__file__).parents[1] / "scripts/o65_delivery.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
Blob = _MODULE.Blob
extract = _MODULE.extract
read_manifest = _MODULE.read_manifest
sha256 = _MODULE.sha256
verify_blob = _MODULE.verify_blob

MANIFEST = Path(__file__).parents[1] / "experiments/o65/pilot-inputs.json"


def test_committed_delivery_manifest_is_strict(tmp_path: Path) -> None:
    manifest = read_manifest(MANIFEST)
    assert manifest.msl_commit == "8fc56d6f2523e978bf6037d8147c57778bd206fa"
    raw = dataclasses.asdict(manifest)
    raw["checkpoint"]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="proxy or statistics identity"):
        read_manifest(path)
    raw = dataclasses.asdict(manifest)
    raw["inputs"]["extra"] = "unsupported"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="blob fields"):
        read_manifest(path)


def test_mismatch_rejected_before_extraction(tmp_path: Path) -> None:
    path = tmp_path / "blob"
    path.write_bytes(b"abc")
    blob = Blob("version/source.tar", sha256(path), 3)
    verify_blob(path, blob)
    path.write_bytes(b"abd")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        verify_blob(path, blob)
    path.write_bytes(b"abc\n")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        verify_blob(path, blob)


@pytest.mark.parametrize("kind", ["symlink", "traversal", "special"])
def test_unsafe_archives_rejected(tmp_path: Path, kind: str) -> None:
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as target:
        info = tarfile.TarInfo("../escaped" if kind == "traversal" else "link")
        if kind == "symlink":
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        elif kind == "special":
            info.type = tarfile.FIFOTYPE
        else:
            info.size = 3
        target.addfile(info, io.BytesIO(b"abc"))
    with pytest.raises((ValueError, tarfile.FilterError)):
        extract(archive, tmp_path / "unpacked")
    assert not (tmp_path / "escaped").exists()


def test_extract_keeps_files_and_rejects_existing_destination(tmp_path: Path) -> None:
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as target:
        info = tarfile.TarInfo("nested/source.c")
        info.size = 3
        target.addfile(info, io.BytesIO(b"abc"))
    destination = tmp_path / "unpacked"
    extract(archive, destination)
    assert (destination / "nested/source.c").read_bytes() == b"abc"
    with pytest.raises(FileExistsError):
        extract(archive, destination)
