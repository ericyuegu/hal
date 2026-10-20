import importlib.util
import io
import subprocess
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
sha256 = _MODULE.sha256
verify_blob = _MODULE.verify_blob


def test_cloned_source_checks_commit_archive_and_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    source = repo / "source.c"
    source.write_text("qualified source\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "--quiet",
            "-m",
            "Source",
        ],
        check=True,
    )
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    archive = tmp_path / "source.tar"
    subprocess.run(["git", "-C", str(repo), "archive", f"--output={archive}", "HEAD"], check=True)
    digest = sha256(archive)
    _MODULE.verify_source(repo, commit, digest, archive)
    with pytest.raises(ValueError, match="commit mismatch"):
        _MODULE.verify_source(repo, "0" * 40, digest, archive)
    with pytest.raises(ValueError, match="archive mismatch"):
        _MODULE.verify_source(repo, commit, "0" * 64, archive)
    source.write_text("changed source\n")
    with pytest.raises(ValueError, match="source is dirty"):
        _MODULE.verify_source(repo, commit, digest, archive)


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
