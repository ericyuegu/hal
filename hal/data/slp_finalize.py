"""Finalize a Slippi ``.slp`` that Dolphin never closed.

Slippi-Ishiiruka backfills the ``raw`` element's length and writes the trailing
``metadata`` object only when it processes a GAME_END event (or closes the file
cleanly). A closed-loop match stopped at ``max_frames`` is abandoned mid-game —
the emulator is killed while still IN_GAME — so its ``.slp`` keeps
``rawLength == 0`` and ends mid-event at the last flushed block. Dolphin itself
reads such a file (it treats ``rawLength == 0`` as "read frames to EOF"), but
peppi / slippilab / any strict UBJSON reader cannot.

``finalize_slp`` repairs one in place: backfill ``rawLength`` to the end of the
last *complete* event (dropping a half-written trailing event) and append a
minimal ``metadata`` object so the top-level UBJSON object closes.
"""

import struct
from pathlib import Path

# UBJSON header of every .slp: '{', key "raw" (U\x03raw), then an optimized
# uint8-array header '[$U#l' and a big-endian int32 rawLength, then the events.
_HEADER = b"{U\x03raw[$U#l"
_LEN_OFF = len(_HEADER)  # offset of the int32 rawLength
_RAW_START = _LEN_OFF + 4  # first byte of the event stream
_EVENT_PAYLOADS = 0x35  # first event; its payload declares every command's size
# Close the root object: key "metadata" (U\x08metadata) -> empty object -> '}'.
# Settings/frames come from the raw events, not metadata, so empty is enough.
_FOOTER = b"U\x08metadata{}}"


def is_finalized(path: str | Path) -> bool:
    """True if ``path`` is a Slippi raw file with ``rawLength`` already set."""
    with open(path, "rb") as fh:
        head = fh.read(_RAW_START)
    return head[:_LEN_OFF] == _HEADER and struct.unpack(">i", head[_LEN_OFF:_RAW_START])[0] != 0


def finalize_bytes(data: bytes) -> bytes | None:
    """Return finalized ``.slp`` bytes, or None if ``data`` is already finalized
    or isn't a Slippi raw file."""
    if data[:_LEN_OFF] != _HEADER or struct.unpack(">i", data[_LEN_OFF:_RAW_START])[0] != 0:
        return None
    end = _last_complete_event_end(data)
    out = bytearray(data[:end])
    struct.pack_into(">i", out, _LEN_OFF, end - _RAW_START)
    out += _FOOTER
    return bytes(out)


def finalize_slp(path: str | Path) -> bool:
    """Repair an unfinalized ``.slp`` in place. Returns True if modified, False
    if already finalized or not a Slippi raw file."""
    finalized = finalize_bytes(Path(path).read_bytes())
    if finalized is None:
        return False
    Path(path).write_bytes(finalized)
    return True


def finalize_replay_dir(replay_dir: str | Path) -> list[Path]:
    """Finalize every unfinalized ``.slp`` directly under ``replay_dir``;
    returns the repaired paths."""
    return [slp for slp in sorted(Path(replay_dir).glob("*.slp")) if finalize_slp(slp)]


def _last_complete_event_end(data: bytes) -> int:
    """Offset just past the last fully-written event in the raw stream. Walks
    the events using the per-command payload sizes the file declares up front,
    so a half-written trailing event (killed mid-flush) is dropped."""
    if data[_RAW_START] != _EVENT_PAYLOADS:
        raise ValueError(f"expected EVENT_PAYLOADS (0x35) at offset {_RAW_START}, got {data[_RAW_START]:#x}")
    declared = data[_RAW_START + 1]  # payload size of the EVENT_PAYLOADS event itself
    sizes = {_EVENT_PAYLOADS: declared}
    p = _RAW_START + 2
    for _ in range((declared - 1) // 3):
        sizes[data[p]] = struct.unpack(">H", data[p + 1 : p + 3])[0]
        p += 3
    cur = end_of_last = _RAW_START + 1 + declared
    n = len(data)
    while cur < n and data[cur] in sizes:
        nxt = cur + 1 + sizes[data[cur]]
        if nxt > n:
            break  # half-written trailing event — drop it
        cur = end_of_last = nxt
    return end_of_last
