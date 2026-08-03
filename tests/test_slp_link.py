"""Staging rules for the slippilab link CLI: one served path per source, no collisions."""

import struct
from pathlib import Path

import pytest

from hal.scripts import slp_link

_HEADER = b"{U\x03raw[$U#l"
_EVENTS = bytes([0x35, 0x04, 0x38, 0x00, 0x04]) + bytes([0x38, 1, 2, 3, 4])


def _slp(path, *, finalized: bool) -> None:
    """Write a minimal Slippi raw file; unfinalized means rawLength == 0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _HEADER + struct.pack(">i", len(_EVENTS) if finalized else 0) + _EVENTS
    path.write_bytes(body + (b"U\x08metadata{}}" if finalized else b""))


@pytest.fixture
def serve_dir(tmp_path, monkeypatch):
    served = tmp_path / "served"
    served.mkdir()
    monkeypatch.setattr(slp_link, "SERVE_DIR", served)
    return served


def test_same_basename_in_different_dirs_stays_distinct(tmp_path, serve_dir):
    """The collision that made every `boot_*/A-c0000-o0.slp` serve boot_000's replay."""
    sources = [tmp_path / "boot_000" / "Game.slp", tmp_path / "boot_001" / "Game.slp"]
    for src in sources:
        _slp(src, finalized=True)

    urls = [slp_link._link(src) for src in sources]
    staged = [slp_link._staged_path(src) for src in sources]

    assert urls[0] != urls[1]
    assert staged[0] != staged[1]
    assert [s.resolve() for s in staged] == sources  # each link points at its own source


def test_underscores_in_dir_names_do_not_collide(tmp_path, serve_dir):
    """`a/b__c/x.slp` vs `a/b/c/x.slp` — the flattened-name scheme mapped both to one file."""
    sources = [tmp_path / "b__c" / "x.slp", tmp_path / "b" / "c" / "x.slp"]
    for src in sources:
        _slp(src, finalized=True)

    assert slp_link._staged_path(sources[0]) != slp_link._staged_path(sources[1])
    assert slp_link._link(sources[0]) != slp_link._link(sources[1])


def test_restages_dangling_link(tmp_path, serve_dir):
    """`Path.exists()` follows symlinks, so a dangling staged link used to raise FileExistsError."""
    src = tmp_path / "Game.slp"
    _slp(src, finalized=True)
    staged = slp_link._staged_path(src)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.symlink_to(tmp_path / "gone.slp")
    assert staged.is_symlink() and not staged.exists()

    slp_link._link(src)

    assert staged.resolve() == src


def test_link_is_idempotent(tmp_path, serve_dir):
    src = tmp_path / "Game.slp"
    _slp(src, finalized=True)
    assert slp_link._link(src) == slp_link._link(src)
    assert slp_link._staged_path(src).resolve() == src


def test_unfinalized_is_staged_as_a_finalized_copy(tmp_path, serve_dir):
    """A match killed mid-game leaves rawLength == 0; slippilab needs the repaired bytes."""
    src = tmp_path / "Game.slp"
    _slp(src, finalized=False)

    slp_link._link(src)
    staged = slp_link._staged_path(src)

    assert not staged.is_symlink()  # a copy, so the source stays untouched on disk
    assert src.read_bytes() != staged.read_bytes()
    assert struct.unpack(">i", staged.read_bytes()[11:15])[0] == len(_EVENTS)
    assert staged.read_bytes().endswith(b"U\x08metadata{}}")


def test_restages_copy_once_the_source_finalizes(tmp_path, serve_dir):
    """Linking a live match stages a repaired copy; re-linking after it closes must not serve that."""
    src = tmp_path / "Game.slp"
    _slp(src, finalized=False)
    slp_link._link(src)
    assert not slp_link._staged_path(src).is_symlink()

    _slp(src, finalized=True)  # the match ended and Dolphin closed the file
    slp_link._link(src)

    assert slp_link._staged_path(src).resolve() == src  # now a symlink to the real thing


def test_non_slp_fails_loud(tmp_path, serve_dir):
    src = tmp_path / "Game.slp"
    src.write_bytes(b"not a slippi file at all")
    with pytest.raises(SystemExit, match="not a Slippi"):
        slp_link._link(src)


def test_collect_walks_dirs_and_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(slp_link, "REPO_DIR", str(tmp_path))
    for name in ["boot_000/Game.slp", "boot_001/Game.slp", "notes.txt"]:
        _slp(tmp_path / name, finalized=True)

    found = slp_link._collect([tmp_path])

    assert found == [tmp_path / "boot_000" / "Game.slp", tmp_path / "boot_001" / "Game.slp"]
    assert slp_link._collect([tmp_path, tmp_path / "boot_000" / "Game.slp"]) == found  # deduped
    assert slp_link._collect([tmp_path / "boot_000"]) == [tmp_path / "boot_000" / "Game.slp"]


def test_collect_resolves_relative_paths_against_the_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(slp_link, "REPO_DIR", str(tmp_path))
    _slp(tmp_path / "runs" / "Game.slp", finalized=True)

    assert slp_link._collect([Path("runs")]) == [tmp_path / "runs" / "Game.slp"]


def test_collect_rejects_missing_path(tmp_path):
    with pytest.raises(SystemExit, match="no such file or directory"):
        slp_link._collect([tmp_path / "nope"])
