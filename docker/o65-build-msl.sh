#!/usr/bin/env bash
# Run inside the HAL image, after the delivery hashes have passed.
set -euo pipefail

delivery="${1:?pass the installed delivery directory}"
for tool in gcc make readelf; do
  command -v "$tool" >/dev/null || { echo "MSL build requires ${tool} in the HAL image" >&2; exit 1; }
done
gcc --version | head -1
export PYTHONPATH="${delivery}/msl${PYTHONPATH:+:${PYTHONPATH}}"
python - "$delivery/inputs/msl/data/raw" <<'PY'
import sys
from pathlib import Path
from tools.data.raw import validate_raw_dir
validate_raw_dir(Path(sys.argv[1]), verify_hashes=True)
PY
# The strict O1 release library produced bit-identical frames to the O0 library and steps 2.4x faster.
# The image sets MSL_CORE_LIBRARY to build/melee_core/python-release/libmelee_core.so.
make -C "$delivery/msl" native python-release -j "${MSL_BUILD_JOBS:-2}" PY="$(command -v python)"
python -m hal.sim.native_build --native-source "$delivery/msl" --output "$delivery/libhal_native.so"
