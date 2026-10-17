# /// script
# requires-python = ">=3.14"
# dependencies = ["cairosvg==2.9.1"]
# ///
"""Rebuild the packaged controller layers with `uv run scripts/prepare_overlay_assets.py`."""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg


def main() -> None:
    directory = Path(__file__).resolve().parents[1] / "hal/netplay_service/overlay_assets/controller"
    root = ET.parse(directory / "controller.svg").getroot()
    cairosvg.svg2png(bytestring=ET.tostring(root), write_to=str(directory / "base.png"))
    for name in ("aButton", "bButton", "xButton", "yButton", "zButton", "lTrigger", "rTrigger", "startButton"):
        layer = ET.Element(root.tag, root.attrib)
        element = next(child for child in root if child.get("id") == name)
        element = copy.deepcopy(element)
        color = {"aButton": "#2AB88F", "bButton": "#DA6B75", "zButton": "#A594CF"}.get(name, "#0E6E63")
        element.set("fill", color)
        element.set("stroke", color)
        layer.append(element)
        cairosvg.svg2png(bytestring=ET.tostring(layer), write_to=str(directory / f"{name}.png"))


if __name__ == "__main__":
    main()
