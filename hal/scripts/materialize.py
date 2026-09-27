"""Extract selected replays into MDS data."""

import tyro

from hal.data.mds_materialization import process_replays

if __name__ == "__main__":
    tyro.cli(process_replays)
