import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_PATH = Path(__file__).parents[1] / "experiments" / "052_policy_world_v8_loader.py"
_SPEC = importlib.util.spec_from_file_location("test_exp052", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_git_sha_uses_modal_source_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    sha = "a" * 40
    monkeypatch.setenv("HAL_GIT_SHA", sha)
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("git must not run when HAL_GIT_SHA is set"),
    )

    assert _MODULE._git_sha() == sha


def test_git_sha_rejects_invalid_modal_source_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAL_GIT_SHA", "not-a-sha")

    with pytest.raises(ValueError, match="40-character lowercase hexadecimal"):
        _MODULE._git_sha()


def test_resume_check_uses_distinct_mosaic_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_roots: list[Path] = []

    class FakeLoader:
        def __init__(self, batches: tuple[str, ...]) -> None:
            self.batches = batches

        def __iter__(self):
            return iter(self.batches)

        def state_dict(self) -> dict[str, object]:
            return {"schema": 1, "mds": {"epoch": 0}}

        def load_state_dict(self, _state: dict[str, object]) -> None:
            return None

    loaders = iter((FakeLoader(("discard", "batch")), FakeLoader(("batch",))))

    def make_loader(_args, _data, *, cache_root: Path, num_workers: int):
        assert num_workers == 0
        cache_roots.append(cache_root)
        return next(loaders)

    class FakeModel:
        def cuda(self):
            return self

        def train(self):
            return self

        def parameters(self) -> tuple[object, ...]:
            return ()

        def state_dict(self) -> dict[str, object]:
            return {}

        def load_state_dict(self, _state: dict[str, object]) -> None:
            return None

    class FakeOptimizer:
        def state_dict(self) -> dict[str, object]:
            return {}

        def load_state_dict(self, _state: dict[str, object]) -> None:
            return None

    monkeypatch.setattr(_MODULE, "make_treatment_loader", make_loader)
    monkeypatch.setattr(_MODULE, "_batch_hash", lambda batch: batch)
    monkeypatch.setattr(_MODULE, "BenchmarkPolicy", FakeModel)
    monkeypatch.setattr(_MODULE.torch.optim, "AdamW", lambda *_args, **_kwargs: FakeOptimizer())
    monkeypatch.setattr(_MODULE, "_update", lambda *_args, **_kwargs: (0.0, 0.0))
    monkeypatch.setattr(_MODULE, "_model_hash", lambda _model: "model")
    monkeypatch.setattr(_MODULE.torch.cuda, "manual_seed_all", lambda _seed: None)
    monkeypatch.setattr(_MODULE.torch.cuda, "synchronize", lambda: None)
    args = SimpleNamespace(cache_root=tmp_path, seed=0)

    assert _MODULE.verify_treatment_resume(args, object())
    assert cache_roots == [tmp_path / "resume-source", tmp_path / "resume-restored"]


def test_cgroup_memory_supports_modal_v1_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "memory.current"
    usage = tmp_path / "memory.usage_in_bytes"
    usage.write_text("123\n")
    monkeypatch.setattr(_MODULE, "_CGROUP_MEMORY_PATHS", (missing, usage))

    assert _MODULE._cgroup_memory() == 123


def test_first_batch_time_includes_loader_iterator_start(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    ticks = iter((10.0, 13.0))

    def monotonic() -> float:
        events.append("clock")
        return next(ticks)

    class Rows:
        def __iter__(self):
            events.append("iter")
            return iter(("batch",))

    monkeypatch.setattr(_MODULE.time, "monotonic", monotonic)

    _iterator, batch, elapsed = _MODULE._start_loader(Rows())

    assert events == ["clock", "iter", "clock"]
    assert batch == "batch"
    assert elapsed == 3.0


def test_next_batch_starts_the_next_loader_epoch() -> None:
    class Rows:
        def __init__(self) -> None:
            self.epochs = 0

        def __iter__(self):
            self.epochs += 1
            return iter((self.epochs,))

    rows = Rows()
    iterator = iter(rows)
    assert next(iterator) == 1

    iterator, batch = _MODULE._next_batch(rows, iterator)

    assert batch == 2
    assert rows.epochs == 2
    with pytest.raises(StopIteration):
        next(iterator)
