"""Current replay-selection command configuration."""

import tyro

from hal.data.replay_selection import FilterConfig


def test_cli_keeps_default_stages_and_can_disable_the_filter() -> None:
    base = ["--index", "index.jsonl", "--output", "paths.txt"]
    defaults = tyro.cli(FilterConfig, args=base)
    no_stage_filter = tyro.cli(FilterConfig, args=[*base, "--stages"])

    assert defaults.stages == (
        "BATTLEFIELD",
        "FINAL_DESTINATION",
        "FOUNTAIN_OF_DREAMS",
        "POKEMON_STADIUM",
        "DREAMLAND",
        "YOSHIS_STORY",
    )
    assert no_stage_filter.stages == ()
