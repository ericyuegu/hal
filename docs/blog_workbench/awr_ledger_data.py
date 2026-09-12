"""Extract the per-frame traces behind the AWR ledger figure.

Reads four .slp games from the explorer fixture and writes
figures_src/awr-ledger.json: percents and stocks for both ports, run-length
encoded, plus the match facts. The page turns them into rewards, returns and
AWR weights in the browser.

Run:  uv run awr_ledger_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import melee

from hal.data.extract import extract_replay

FIXTURE = Path.home() / "data/scratch/awr-explorer/slps"
GAMES = [
    ("diamond-diamond-00018278aa8bb1c30871ab4c.slp", "ranked · diamond vs diamond"),
    ("master-master-000551fbfa951d4cce5f3f7a.slp", "ranked · master vs master"),
    ("h2h-bc-baseline-wins-cptfalcon-vs-marth-yoshis_story.slp", "model vs model · BC baseline wins"),
    ("h2h-lopsided-awr-strong-b0.3-detach-wins-4-0-falco-vs-falco.slp", "model vs model · AWR wins 4–0"),
]
OUT = Path(__file__).resolve().parent / "figures_src" / "awr-ledger.json"


NAMES = {"Cptfalcon": "Captain Falcon", "Yoshis Story": "Yoshi’s Story", "Pokemon Stadium": "Pokémon Stadium"}


def pretty(name: str) -> str:
    name = name.replace("_", " ").title()
    return NAMES.get(name, name)


def rle(values: list) -> list:
    out: list = []
    for v in values:
        if out and out[-1][0] == v:
            out[-1][1] += 1
        else:
            out.append([v, 1])
    return out


def main() -> None:
    games = []
    for name, label in GAMES:
        sample = extract_replay(str(FIXTURE / name))
        if sample is None:
            raise SystemExit(f"could not parse {name}")
        frame = sample["frame"]
        if int(frame[-1]) - int(frame[0]) + 1 != len(frame):
            raise SystemExit(f"{name}: frame ids not contiguous")
        games.append(
            {
                "file": name,
                "label": label,
                "p1": pretty(melee.Character(int(sample["p1_character"][0])).name),
                "p2": pretty(melee.Character(int(sample["p2_character"][0])).name),
                "stage": pretty(melee.Stage(int(sample["stage"][0])).name),
                "frames": len(frame),
                "p1Percent": rle([round(float(v) * 10) for v in sample["p1_percent"]]),
                "p2Percent": rle([round(float(v) * 10) for v in sample["p2_percent"]]),
                "p1Stock": rle([int(v) for v in sample["p1_stock"]]),
                "p2Stock": rle([int(v) for v in sample["p2_stock"]]),
            }
        )
        print(f"{name}: {len(frame)} frames · {games[-1]['p1']} vs {games[-1]['p2']} · {games[-1]['stage']}")
    OUT.write_text(json.dumps({"fps": 60, "games": games}, separators=(",", ":")))
    print(f"{OUT}: {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
