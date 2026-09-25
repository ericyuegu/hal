The netplay libmelee source archive is based on ericyuegu/libmelee commit
`222a399cd1368d33458b4c36f4823f7220ca889d`, with `libmelee-read-only.patch`
and the local version `0.47.0+hal.realtime.1`.

`Console.step(flush_controllers=False)` does not write inputs, including while
processing game-start or rollback events. The default preserves the original
behavior. Without this option, draining observations replays controller writes.
HAL's netplay worker explicitly flushes the state selected for the latest frame.

The source archive keeps installs reproducible before the fork change is
published. Replace it with a tested upstream Git pin once that commit contains
the same interface. Regression coverage is in `tests/test_netplay_realtime.py`.
Build with `uv build --sdist` after applying the patch and setting the version.
The source and its LGPL license are included in the archive.
