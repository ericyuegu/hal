import subprocess
import sys


def test_package_root_is_inert_and_submodules_remain_importable() -> None:
    code = """
import sys
import hal
assert 'melee' not in sys.modules
from hal import r2, streams
assert r2 is not None and streams is not None
assert 'melee' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
