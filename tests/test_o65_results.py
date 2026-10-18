import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("o65_results", Path(__file__).parents[1] / "scripts/o65_results.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
archive_results = _MODULE.archive_results
sha256 = _MODULE.sha256
validate_prefix = _MODULE.validate_prefix
verify_archive = _MODULE.verify_archive


def test_result_archive_excludes_inputs_and_checks_content(tmp_path: Path) -> None:
    for name in ("pilot/train/model.pt", "startup.log", "pilot/delivery/payloads/final.pt", "tmp/compiler.o"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    archive, manifest = archive_results(tmp_path)
    raw = json.loads(manifest.read_text())
    assert raw["format"] == "hal.o65.results.v1"
    assert raw["archive_sha256"] == sha256(archive)
    with tarfile.open(archive) as source:
        assert source.getnames() == ["pilot/train/model.pt", "startup.log"]
    assert {record["path"] for record in raw["files"]} == {"pilot/train/model.pt", "startup.log"}
    _MODULE.unpack_results(archive, manifest, tmp_path / "restored")
    assert (tmp_path / "restored/pilot/train/model.pt").read_text() == "pilot/train/model.pt"
    verify_archive(archive, raw["archive_sha256"], raw["archive_size_bytes"])
    with archive.open("ab") as target:
        target.write(b"broken")
    with pytest.raises(ValueError, match="archive hash or size"):
        verify_archive(archive, raw["archive_sha256"], raw["archive_size_bytes"])


def test_result_archive_rejects_symlinks(tmp_path: Path) -> None:
    (tmp_path / "link").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink"):
        archive_results(tmp_path)


def test_result_file_manifest_rejects_mismatch_and_unsafe_names(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"weights")
    archive, manifest = archive_results(tmp_path)
    raw = json.loads(manifest.read_text())
    raw["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="file mismatch"):
        _MODULE.unpack_results(archive, manifest, tmp_path / "mismatch")
    raw["files"][0]["path"] = "../escaped"
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="file identity"):
        _MODULE.unpack_results(archive, manifest, tmp_path / "unsafe")


@pytest.mark.parametrize("prefix", ["", "other/run", "o65/pilot-results/../run", "o65/pilot-results//run"])
def test_results_reject_unsafe_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match="results prefix"):
        validate_prefix(prefix)
