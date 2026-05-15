"""CLI: download integration fixtures into `<repo>/data/`.

Idempotent: skips fixtures whose local sha256 already matches.

Usage:
    python -m hal.scripts.fetch                  # all fixtures
    python -m hal.scripts.fetch --name dev.7z    # one fixture
"""

import tyro

from hal.fixtures import ALL
from hal.fixtures import BY_NAME
from hal.fixtures import ensure
from hal.fixtures import ensure_all


def fetch(name: str | None = None) -> None:
    """Fetch one fixture by name, or all if --name omitted."""
    if name is None:
        ensure_all()
        return
    if name not in BY_NAME:
        raise SystemExit(f"unknown fixture {name!r}; known: {sorted(f.name for f in ALL)}")
    ensure(BY_NAME[name])


def main() -> None:
    tyro.cli(fetch)


if __name__ == "__main__":
    main()
