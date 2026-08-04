"""Repair a Slippi ``.slp`` that Dolphin never closed cleanly.

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

``trim_to_last_frame`` repairs the finer tear inside that envelope. The last
complete EVENT can sit in the middle of a frame — one port's post-frame event
written and the other's not — and peppi builds its port columns as one struct
array, so ragged children make it panic and the whole match is lost. Cutting
back to the last complete FRAME (the Frame Bookend event) drops the torn
remainder and leaves every port the same number of frames.
"""

import struct
from pathlib import Path

# UBJSON header of every .slp: '{', key "raw" (U\x03raw), then an optimized
# uint8-array header '[$U#l' and a big-endian int32 rawLength, then the events.
_HEADER = b"{U\x03raw[$U#l"
_LEN_OFF = len(_HEADER)  # offset of the int32 rawLength
_RAW_START = _LEN_OFF + 4  # first byte of the event stream
_EVENT_PAYLOADS = 0x35  # first event; its payload declares every command's size
_FRAME_BOOKEND = 0x3C  # last event of a frame group (slp 3.0.0 and up)
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
    end, _ = _scan_events(data, len(data))
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


def trim_to_last_frame_bytes(data: bytes) -> bytes | None:
    """Return ``data`` cut back to its last complete frame, or None if it carries
    no complete frame to cut back to.

    ``rawLength`` is rewritten to the new end. A trailing ``metadata`` object is
    kept as it is (it carries the player names a viewer reads); a stream with none
    — never finalized, or truncated past its own ``rawLength`` — gets the minimal
    footer, so this also finalizes what it trims. A stream already ending on a
    frame boundary comes back unchanged.

    None covers: not a Slippi raw file, a header or command-size table cut short,
    and no Frame Bookend at all (a boot that started no frame, or a replay older
    than slp 3.0.0).
    """
    # The header, the first event and the whole command-size table must be there
    # before the walk can read a size out of the stream.
    if data[:_LEN_OFF] != _HEADER or len(data) <= _RAW_START + 1 or data[_RAW_START] != _EVENT_PAYLOADS:
        return None
    if len(data) < _RAW_START + 1 + data[_RAW_START + 1]:
        return None
    raw_length = struct.unpack(">i", data[_LEN_OFF:_RAW_START])[0]
    raw_end = min(_RAW_START + raw_length, len(data)) if raw_length > 0 else len(data)
    _, end_of_last_frame = _scan_events(data, raw_end)
    if end_of_last_frame is None:
        return None
    out = bytearray(data[:end_of_last_frame])
    struct.pack_into(">i", out, _LEN_OFF, end_of_last_frame - _RAW_START)
    return bytes(out) + (data[raw_end:] or _FOOTER)


def trim_to_last_frame(path: str | Path) -> bool:
    """Cut a ``.slp`` back to its last complete frame, in place. Returns True if
    modified, False if there is nothing to cut or the file is unusable.

    Use it on a replay peppi refuses to read: a match killed at a frame budget
    ends mid-frame, and peppi panics on the ragged port columns. The events after
    the last Frame Bookend are the torn remainder, so dropping them costs one
    frame and saves the match. Repeated calls after the first are no-ops.
    """
    data = Path(path).read_bytes()
    trimmed = trim_to_last_frame_bytes(data)
    if trimmed is None or trimmed == data:
        return False
    Path(path).write_bytes(trimmed)
    return True


def _scan_events(data: bytes, end: int) -> tuple[int, int | None]:
    """Walk the raw event stream up to ``end``: ``(end of the last complete
    event, end of the last complete frame)``.

    Both offsets are absolute. The frame offset is None when no Frame Bookend
    closed. The walk uses the per-command payload sizes the file declares up
    front, so a half-written trailing event (killed mid-flush) is dropped."""
    if data[_RAW_START] != _EVENT_PAYLOADS:
        raise ValueError(f"expected EVENT_PAYLOADS (0x35) at offset {_RAW_START}, got {data[_RAW_START]:#x}")
    declared = data[_RAW_START + 1]  # payload size of the EVENT_PAYLOADS event itself
    sizes = {_EVENT_PAYLOADS: declared}
    p = _RAW_START + 2
    for _ in range((declared - 1) // 3):
        sizes[data[p]] = struct.unpack(">H", data[p + 1 : p + 3])[0]
        p += 3
    cur = end_of_last_event = _RAW_START + 1 + declared
    end_of_last_frame: int | None = None
    while cur < end and data[cur] in sizes:
        nxt = cur + 1 + sizes[data[cur]]
        if nxt > end:
            break  # half-written trailing event — drop it
        if data[cur] == _FRAME_BOOKEND:
            end_of_last_frame = nxt
        cur = end_of_last_event = nxt
    return end_of_last_event, end_of_last_frame
