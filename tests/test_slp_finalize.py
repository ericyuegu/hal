"""Byte-level tests for the unclosed-.slp finalizer and the mid-frame trimmer."""

import struct
from pathlib import Path

from hal.data.slp_finalize import finalize_bytes
from hal.data.slp_finalize import finalize_slp
from hal.data.slp_finalize import is_finalized
from hal.data.slp_finalize import trim_to_last_frame
from hal.data.slp_finalize import trim_to_last_frame_bytes

_HEADER = b"{U\x03raw[$U#l"
_BOOKEND = 0x3C
_FOOTER = b"U\x08metadata{}}"
# Any trailing metadata object; the trimmer must keep it byte for byte.
_METADATA = b"U\x08metadata{U\x05namesS\x02hi}}"


def _unfinalized(*, trailing_partial: bool) -> bytes:
    """A minimal Slippi raw stream with rawLength == 0: an EVENT_PAYLOADS event
    declaring one 4-byte command (0x38), two complete 0x38 events, and optionally
    a half-written third one that finalize must drop."""
    event_payloads = bytes([0x35, 0x04, 0x38, 0x00, 0x04])  # 0x35, size=4, {0x38: 4}
    full = bytes([0x38, 1, 2, 3, 4]) + bytes([0x38, 5, 6, 7, 8])
    partial = bytes([0x38, 9, 9]) if trailing_partial else b""
    return _HEADER + struct.pack(">i", 0) + event_payloads + full + partial


def _events(frames: int, *, torn: bool) -> bytes:
    """``frames`` complete frames plus an optional torn one.

    A frame is a 4-byte 0x38 event followed by a Frame Bookend (0x3C); the torn
    frame is a 0x38 with no bookend after it, which is what a match killed at its
    frame budget leaves behind.
    """
    event_payloads = bytes([0x35, 0x07, 0x38, 0x00, 0x04, _BOOKEND, 0x00, 0x04])
    body = b"".join(bytes([0x38, i, 0, 0, 0]) + bytes([_BOOKEND, i, 0, 0, 0]) for i in range(frames))
    return event_payloads + body + (bytes([0x38, 9, 9, 9, 9]) if torn else b"")


def _slp(events: bytes, *, finalized: bool, tail: bytes = _METADATA) -> bytes:
    return _HEADER + struct.pack(">i", len(events) if finalized else 0) + events + (tail if finalized else b"")


def _write(tmp_path: Path, data: bytes) -> Path:
    path = tmp_path / "Game.slp"
    path.write_bytes(data)
    return path


def test_finalize_backfills_length_and_appends_footer():
    data = _unfinalized(trailing_partial=False)
    out = finalize_bytes(data)
    assert out is not None
    # rawLength = EVENT_PAYLOADS (5) + two 0x38 events (10) = 15
    assert struct.unpack(">i", out[11:15])[0] == 15
    assert out.endswith(b"U\x08metadata{}}")
    # identical to the input but with rawLength backfilled and the footer appended
    assert out == data[:11] + struct.pack(">i", 15) + data[15:] + b"U\x08metadata{}}"


def test_finalize_drops_half_written_trailing_event():
    out = finalize_bytes(_unfinalized(trailing_partial=True))
    assert out is not None
    # the 3-byte partial 0x38 is dropped: rawLength still 15, footer right after
    assert struct.unpack(">i", out[11:15])[0] == 15
    assert out[15 : 15 + 15] == _unfinalized(trailing_partial=False)[15:]
    assert out.endswith(b"U\x08metadata{}}")


def test_finalize_is_idempotent_on_finalized():
    once = finalize_bytes(_unfinalized(trailing_partial=True))
    assert once is not None
    assert finalize_bytes(once) is None  # already finalized → no-op


def test_finalize_rejects_non_slp():
    assert finalize_bytes(b"not a slippi file at all") is None


def test_finalize_slp_in_place(tmp_path):
    f = tmp_path / "Game.slp"
    f.write_bytes(_unfinalized(trailing_partial=True))
    assert not is_finalized(f)
    assert finalize_slp(f) is True
    assert is_finalized(f)
    assert finalize_slp(f) is False  # second call is a no-op


def test_trim_drops_the_torn_trailing_frame(tmp_path):
    f = _write(tmp_path, _slp(_events(2, torn=True), finalized=True))
    assert trim_to_last_frame(f) is True
    # rawLength now ends on the second bookend, and the metadata block is untouched.
    assert f.read_bytes() == _slp(_events(2, torn=False), finalized=True)
    assert trim_to_last_frame(f) is False  # second call is a no-op


def test_trim_keeps_a_frame_aligned_file(tmp_path):
    data = _slp(_events(2, torn=False), finalized=True)
    f = _write(tmp_path, data)
    assert trim_to_last_frame(f) is False
    assert f.read_bytes() == data


def test_trim_finalizes_the_file_it_trims(tmp_path):
    """A file killed before Dolphin wrote rawLength gets both repairs at once."""
    f = _write(tmp_path, _slp(_events(2, torn=True), finalized=False))
    assert trim_to_last_frame(f) is True
    assert is_finalized(f)
    assert f.read_bytes() == _slp(_events(2, torn=False), finalized=True, tail=_FOOTER)


def test_trim_clamps_a_raw_length_past_the_end(tmp_path):
    """A finalized file truncated mid-frame: rawLength claims more than is there,
    and the metadata block is gone with the rest."""
    data = _slp(_events(2, torn=True), finalized=True)
    f = _write(tmp_path, data[: len(data) - len(_METADATA) - 3])
    assert trim_to_last_frame(f) is True
    assert f.read_bytes() == _slp(_events(2, torn=False), finalized=True, tail=_FOOTER)


def test_trim_needs_a_frame_bookend(tmp_path):
    # An event stream with no bookend at all: nothing can be salvaged frame-wise.
    assert trim_to_last_frame(_write(tmp_path, _unfinalized(trailing_partial=True))) is False


def test_trim_rejects_non_slp(tmp_path):
    assert trim_to_last_frame(_write(tmp_path, b"not a slippi file at all")) is False


def test_trim_rejects_a_truncated_command_table():
    assert trim_to_last_frame_bytes(_HEADER + struct.pack(">i", 0) + bytes([0x35, 0x07, 0x38])) is None
