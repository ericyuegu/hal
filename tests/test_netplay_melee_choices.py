import melee
import pytest

from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import Choice
from hal.netplay_service.melee_choices import character_from_choice
from hal.netplay_service.melee_choices import stage_from_choice
from hal.netplay_service.self_play import DemoSelection
from hal.netplay_service.self_play import selection


def test_young_link_keeps_the_public_name_and_resolves_to_the_playable_fighter() -> None:
    chosen = selection(16, 1)
    assert chosen.characters == ("YOUNG_LINK", "MARIO")
    assert chosen.stage == "DREAMLAND"
    assert chosen.matchup().players[0].character is melee.Character.YLINK
    assert character_from_choice("YOUNG_LINK") is melee.Character.YLINK


@pytest.mark.parametrize("character", CHARACTERS, ids=lambda choice: choice.value)
@pytest.mark.parametrize("stage", STAGES, ids=lambda choice: choice.value)
def test_every_published_choice_configures_both_ports(character: Choice, stage: Choice) -> None:
    chosen = DemoSelection((character.value, character.value), ("MASTER", "MASTER"), stage.value)
    matchup = chosen.matchup()
    assert tuple(player.port for player in matchup.players) == (1, 2)
    assert all(player.character is character_from_choice(character.value) for player in matchup.players)
    assert matchup.stage is stage_from_choice(stage.value)


def test_catalog_resolves_to_26_distinct_playable_characters_and_six_stages() -> None:
    assert len({character_from_choice(choice.value) for choice in CHARACTERS}) == 26
    assert len({stage_from_choice(choice.value) for choice in STAGES}) == 6


@pytest.mark.parametrize("value", ["YLINK", "UNKNOWN_CHARACTER", ""])
def test_internal_names_and_unknown_characters_are_not_public_choices(value: str) -> None:
    with pytest.raises(ValueError, match="unsupported character"):
        character_from_choice(value)


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported stage"):
        stage_from_choice("UNKNOWN_STAGE")
