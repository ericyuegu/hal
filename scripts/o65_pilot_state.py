"""Publish a complete pilot snapshot to its Modal Volume and R2 backup."""

import tyro

from hal.training.self_play_publication import Args
from hal.training.self_play_publication import publish

if __name__ == "__main__":
    publish(tyro.cli(Args))
