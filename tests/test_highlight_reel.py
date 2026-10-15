import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from scripts.highlight_reel import DIP_PEAK
    from scripts.highlight_reel import EDITS
    from scripts.highlight_reel import FPS
    from scripts.highlight_reel import Clip
    from scripts.highlight_reel import Cue
    from scripts.highlight_reel import GameSpec
    from scripts.highlight_reel import Moment
    from scripts.highlight_reel import confirmed_rows
    from scripts.highlight_reel import current_step
    from scripts.highlight_reel import edit_from
    from scripts.highlight_reel import locate
    from scripts.highlight_reel import plan
    from scripts.highlight_reel import replay_cuts
    from scripts.highlight_reel import replay_dip
    from scripts.highlight_reel import validate_cues
    from scripts.highlight_reel import wrap
finally:
    sys.path.pop(0)

SOURCE = Path("source.mp4")


def test_half_speed_replay_doubles_its_length_and_segments_are_contiguous() -> None:
    segments = plan((Clip(SOURCE, 0, 10, 12, 1, False, 0), Clip(SOURCE, 0, 10.5, 11.5, 0.5, True, 0)))
    assert [s.frames for s in segments] == [2 * FPS, 2 * FPS]
    assert [s.output_frame for s in segments] == [0, 2 * FPS]


def test_locate_maps_replay_output_frames_back_to_source_time() -> None:
    segments = plan((Clip(SOURCE, 0, 10, 12, 1, False, 5), Clip(SOURCE, 0, 10.5, 11.5, 0.5, True, 5)))
    segment, t, frame = locate(segments, 2 * FPS + FPS)
    assert segment.clip.replay
    assert t == pytest.approx(11.0)
    assert frame == round(11.0 * 60000 / 1001 + 5)
    with pytest.raises(ValueError):
        locate(segments, 4 * FPS)


def test_current_step_advances_at_each_step_frame() -> None:
    cue = Cue(100, 200, "Caption", ("Roll catch", "up-smash", "dash attack"), (110, 160))
    assert [current_step(cue, f) for f in (100, 109, 110, 159, 160, 199)] == [0, 0, 1, 1, 2, 2]


def test_replay_dip_peaks_on_cuts_into_and_out_of_replays_only() -> None:
    segments = plan(
        (
            Clip(SOURCE, 0, 0, 1, 1, False, 0),
            Clip(SOURCE, 0, 0, 1, 1, False, 0),
            Clip(SOURCE, 0, 0, 0.5, 0.5, True, 0),
            Clip(SOURCE, 0, 0, 1, 1, False, 0),
        )
    )
    cuts = replay_cuts(segments)
    assert cuts == (2 * FPS, 3 * FPS)
    assert replay_dip(cuts, FPS) == 0
    assert replay_dip(cuts, 2 * FPS) == pytest.approx(DIP_PEAK)
    assert replay_dip(cuts, 2 * FPS - 30) == 0


def test_validate_cues_rejects_missing_or_out_of_window_step_frames() -> None:
    for edit in EDITS.values():
        validate_cues(edit.cues)
    with pytest.raises(ValueError):
        validate_cues(((Cue(0, 10, "Caption", ("Shine", "drill")),),))
    with pytest.raises(ValueError):
        validate_cues(((Cue(0, 10, "Caption", ("Shine", "drill"), (12,)),),))
    with pytest.raises(ValueError):
        validate_cues(((Cue(0, 10, "Caption", (), (5,)),),))


def test_validate_cues_rejects_overlapping_cues() -> None:
    validate_cues(((Cue(0, 10, "First"), Cue(10, 20, "Second")),))
    with pytest.raises(ValueError):
        validate_cues(((Cue(0, 10, "First"), Cue(9, 20, "Second")),))


def test_wrap_keeps_no_break_spaces_together() -> None:
    font = ImageFont.load_default(size=20)
    width = font.getlength("Stock at") + 1
    assert wrap("Stock at 9%.", font, width) == ["Stock at", "9%."]
    assert wrap("Stock at\u00a09%.", font, width) == ["Stock", "at\u00a09%."]


def test_confirmed_rows_take_the_last_occurrence_after_a_rollback() -> None:
    # Frames 3 and 4 are rolled back once: the stream rewinds to 3 after reaching 4.
    frames = np.array([1, 2, 3, 4, 3, 4, 5])
    assert confirmed_rows(frames).tolist() == [0, 1, 4, 5, 6]
    with pytest.raises(ValueError):
        confirmed_rows(np.array([1, 3]))


def test_edit_from_numbers_games_by_first_appearance_and_maps_frames_to_vod_seconds() -> None:
    first, second = GameSpec(Path("a.slp"), "Grandmaster"), GameSpec(Path("b.slp"), "Diamond 1")
    moments = (
        Moment(first, SOURCE, -100, 500, 620, "One"),
        Moment(second, SOURCE, 0, 60, 120, "Two"),
        Moment(first, SOURCE, -100, 200, 300, "Three"),
    )
    edit = edit_from(moments)
    assert edit.games == (first, second)
    assert [clip.game for clip in edit.clips] == [0, 1, 0]
    assert [cue.text for cue in edit.cues[0]] == ["Three", "One"]
    clip = edit.clips[0]
    assert round(clip.start * 60000 / 1001 + clip.offset) == 500
    assert round(clip.end * 60000 / 1001 + clip.offset) == 620


def test_edit_from_shares_one_cue_across_consecutive_moments_with_the_same_caption() -> None:
    game, other = GameSpec(Path("a.slp"), "Grandmaster"), GameSpec(Path("b.slp"), "Diamond 1")
    moments = (
        Moment(game, SOURCE, 0, 100, 200, "Same"),
        Moment(game, SOURCE, 0, 400, 500, "Same"),
        Moment(other, SOURCE, 0, 60, 120, "Other"),
        Moment(game, SOURCE, 0, 700, 800, "Same"),
    )
    edit = edit_from(moments)
    assert [(cue.start, cue.end) for cue in edit.cues[0]] == [(100, 500), (700, 800)]
