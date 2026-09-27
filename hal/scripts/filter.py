"""Select replay paths from an index."""

import tyro

from hal.data.replay_selection import FilterConfig
from hal.data.replay_selection import select_replay_paths

if __name__ == "__main__":
    select_replay_paths(tyro.cli(FilterConfig))
