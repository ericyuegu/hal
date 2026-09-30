"""Run the G4 Cody Fox model through Slippi ranked sets and stream Dolphin."""

import tyro

from hal.eval.ranked import RankedConfig
from hal.eval.ranked import run


def main() -> None:
    run(tyro.cli(RankedConfig))


if __name__ == "__main__":
    main()
