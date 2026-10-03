"""Build the HAL C adapter against a selected melee-sim-light public header."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from hal.data.schema import MDS_PER_FRAME_DTYPES
from hal.sim import native_bridge
from hal.sim.native import OBSERVATION_FIELDS
from hal.wire import ACTION_CHANNELS
from hal.wire import BUTTON_BITS
from hal.wire import mask_value


def schema_header() -> str:
    fields = []
    masks = []
    for name in OBSERVATION_FIELDS:
        dtype = np.dtype(MDS_PER_FRAME_DTYPES[name])
        if dtype not in (np.dtype("float32"), np.dtype("int32")):
            raise ValueError(f"unsupported native column dtype: {name}: {dtype}")
        fields.append((name, dtype.str))
        masks.append(int(np.array(mask_value(dtype), dtype=dtype).view(np.uint32)))
    buttons = [BUTTON_BITS[name.removeprefix("button_")] for name in ACTION_CHANNELS[6:]]
    identity = hashlib.sha256(json.dumps((fields, masks, buttons), separators=(",", ":")).encode()).hexdigest()
    enums = ",\n".join(f"  HAL_COL_{name.upper()} = {i}" for i, name in enumerate(OBSERVATION_FIELDS))
    return (
        f'#define HAL_SCHEMA_HASH "{identity}"\n'
        f"#define HAL_COLUMN_COUNT {len(fields)}\n"
        f"enum {{\n{enums}\n}};\n"
        "static const uint32_t hal_masks[] = {" + ",".join(str(v) + "u" for v in masks) + "};\n"
        "static const uint16_t hal_buttons[] = {" + ",".join(map(str, buttons)) + "};\n"
    )


def schema_hash() -> str:
    return schema_header().split('"')[1]


def build(native_source: Path, output: Path, compiler: str = "cc") -> Path:
    """Compile explicitly; importing or opening an environment never invokes a compiler."""
    source = Path(__file__).with_name("native_adapter.c")
    header = native_source.resolve() / "src" / "api.h"
    if not header.is_file():
        raise FileNotFoundError(header)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hal-native-", dir=output.parent) as temporary:
        temp = Path(temporary)
        (temp / "hal_native_schema.h").write_text(schema_header())
        library = temp / output.name
        subprocess.run(
            [
                compiler,
                "-std=c11",
                "-O3",
                "-fPIC",
                "-shared",
                "-fno-fast-math",
                "-ffp-contract=off",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                str(header.parent),
                "-I",
                str(temp),
                str(source),
                "-o",
                str(library),
                "-ldl",
                "-lm",
            ],
            check=True,
        )
        library.replace(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=native_bridge.library_path())
    parser.add_argument("--compiler", default="cc")
    args = parser.parse_args()
    print(build(args.native_source, args.output, args.compiler))


if __name__ == "__main__":
    main()
