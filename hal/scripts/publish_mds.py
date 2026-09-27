"""Audit and publish a staged MDS corpus."""

import tyro

from hal.data.mds_publication import publish_mds

if __name__ == "__main__":
    tyro.cli(publish_mds)
