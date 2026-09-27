"""Index replay files from a directory or archive."""

import tyro

from hal.data.index_builder import build_index

if __name__ == "__main__":
    tyro.cli(build_index)
