from pathlib import Path

from hal.fixtures import ISO
from hal.fixtures import NETPLAY_EMULATOR


def test_netplay_static_fixture_pins() -> None:
    assert ISO.r2_key == "fixtures/ssbm.ciso"
    assert ISO.sha256 == "b7de482eb955c8a96b6746dfa043b69ae7bf6c7c2a09ac382b9da126faa7055c"
    assert NETPLAY_EMULATOR.url == (
        "https://github.com/project-slippi/Ishiiruka/releases/download/v3.6.4/Slippi_Online-x86_64.AppImage"
    )
    assert NETPLAY_EMULATOR.r2_key is None
    assert NETPLAY_EMULATOR.sha256 == "e0f984e5bbecb98e3a746da1f173a475b06c3a1ba6b73e2e31bbe85a5f5a5e8a"
    assert NETPLAY_EMULATOR.size_bytes == 111_679_992
    assert NETPLAY_EMULATOR.dest == Path("data/emulator/slippi-3.6.4/Slippi_Online-x86_64.AppImage")
    assert NETPLAY_EMULATOR.executable
