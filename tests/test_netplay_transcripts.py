import importlib.util
import json
from pathlib import Path

from hal.paths import REPO_DIR

_SPEC = importlib.util.spec_from_file_location(
    "record_netplay_transcripts", Path(REPO_DIR) / "scripts" / "record_netplay_transcripts.py"
)
assert _SPEC is not None and _SPEC.loader is not None
recorder = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recorder)

COMMITTED = Path(REPO_DIR) / "web" / "netplay-api" / "test" / "transcripts"


def test_committed_transcripts_match_the_python_service(tmp_path: Path) -> None:
    recorder.record_all(tmp_path)
    produced = {path.name: json.loads(path.read_text()) for path in tmp_path.glob("*.json")}
    committed = {path.name: json.loads(path.read_text()) for path in COMMITTED.glob("*.json")}
    assert produced == committed


def test_recording_is_deterministic(tmp_path: Path) -> None:
    recorder.record_all(tmp_path / "first")
    recorder.record_all(tmp_path / "second")
    for path in sorted((tmp_path / "first").glob("*.json")):
        assert path.read_text() == (tmp_path / "second" / path.name).read_text()
