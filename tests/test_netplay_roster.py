import pytest

from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.roster import PLAYER_IDENTITY_SIDECAR_SHA256
from hal.netplay_service.roster import PROFESSIONAL_ROSTER
from hal.scripts.build_netplay_roster import Replay
from hal.scripts.build_netplay_roster import RosterEntry
from hal.scripts.build_netplay_roster import build_roster
from hal.scripts.build_netplay_roster import main_and_alternate_codes
from hal.scripts.build_netplay_roster import render_module


def _replay(frames: int, *players: tuple[str, str]) -> Replay:
    return Replay(frames, tuple(code for code, _ in players), tuple(name for _, name in players))


def _mango_replays() -> tuple[Replay, ...]:
    # MANG#0 is the main code; MANG#9 is an alt that meets many opponents; FAN#1 only ever meets MANG#9,
    # so it is an opponent even though it never shares a replay with the main code.
    main = [_replay(100, ("MANG#0", "mang"), (f"OPP#{index}", f"opp {index}")) for index in range(12)]
    alt = [_replay(100, ("MANG#9", "mang"), (f"ALT#{index}", f"alt {index}")) for index in range(6)]
    fan = [_replay(100, ("MANG#9", "mang"), ("FAN#1", "biggest fan")) for _ in range(2)]
    return (*main, *alt, *fan)


def test_alternate_accounts_never_meet_the_main_code_and_meet_many_opponents() -> None:
    main, alternates = main_and_alternate_codes(_mango_replays())
    assert main == "MANG#0"
    assert alternates == {"MANG#9"}


def test_roster_ranks_codes_by_train_frames_without_alternates() -> None:
    replays = {
        "mang0": _mango_replays(),
        "zain": (
            _replay(5_000, ("ZAIN#0", "zain"), ("FAN#1", "biggest fan")),
            _replay(5_000, ("ZAIN#0", "zain"), ("OPP#0", "opp 0")),
        ),
    }
    vocabulary = frozenset(code for group in replays.values() for replay in group for code in replay.codes)
    roster = build_roster(replays, vocabulary, 3)
    assert roster == (
        RosterEntry("ZAIN#0", "Zain", 10_000),
        RosterEntry("FAN#1", "biggest fan", 5_200),
        RosterEntry("OPP#0", "opp 0", 5_100),
    )
    assert "MANG#9" not in {entry.code for entry in build_roster(replays, vocabulary, 30)}


def test_roster_rejects_codes_outside_the_vocabulary() -> None:
    with pytest.raises(ValueError, match="absent from the sidecar vocabulary"):
        build_roster({"mang0": _mango_replays()}, frozenset({"MANG#0"}), 2)


def test_roster_rejects_repeated_labels() -> None:
    replays = {
        "mang0": (_replay(100, ("MANG#0", "x"), ("A#1", "same")), _replay(100, ("MANG#0", "x"), ("B#1", "Same")))
    }
    with pytest.raises(ValueError, match="labels repeat"):
        build_roster(replays, frozenset({"MANG#0", "A#1", "B#1"}), 3)


def test_rendered_module_round_trips_labels_with_quotes() -> None:
    entries = (RosterEntry("JAH#516", "Jah Ridin'", 10), RosterEntry("Q#1", 'say "hi"', 5))
    namespace: dict[str, object] = {}
    exec(render_module(entries, "a" * 64, "b" * 64), namespace)
    assert namespace["PROFESSIONAL_ROSTER"] == (("JAH#516", "Jah Ridin'", 10), ("Q#1", 'say "hi"', 5))


def test_committed_roster_is_the_top_fifty_from_the_059_sidecar() -> None:
    assert PLAYER_IDENTITY_SIDECAR_SHA256 == "54ccf8a2497fe240313117297ca2ea31158e08db2cc53c67e7aa46853a8dac1c"
    assert len(PROFESSIONAL_ROSTER) == 50
    codes = [code for code, _, _ in PROFESSIONAL_ROSTER]
    labels = [label.casefold() for _, label, _ in PROFESSIONAL_ROSTER]
    frames = [count for _, _, count in PROFESSIONAL_ROSTER]
    assert len(set(codes)) == len(codes) and len(set(labels)) == len(labels)
    assert frames == sorted(frames, reverse=True)
    values = [choice.value for choice in IMITATIONS]
    assert values == ["MASKED", *codes, "PLATINUM", "DIAMOND", "MASTER"]
