"""Prepare professional replay sources for selection."""

import tyro

from hal.data.professional_replays import PrepareProfessionalConfig
from hal.data.professional_replays import prepare_professional

if __name__ == "__main__":
    prepare_professional(tyro.cli(PrepareProfessionalConfig))
