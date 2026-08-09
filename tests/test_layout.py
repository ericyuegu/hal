import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def _invalid_imports(paths: list[Path], prefix: str) -> dict[str, list[str]]:
    bad: dict[str, list[str]] = {}
    for path in paths:
        names = sorted(name for name in _imports(path) if name == prefix or name.startswith(f"{prefix}."))
        if names:
            bad[str(path.relative_to(_ROOT))] = names
    return bad


def test_host_scripts_do_not_import_hal() -> None:
    assert _invalid_imports(list((_ROOT / "scripts").glob("*.py")), "hal") == {}


def test_package_commands_do_not_import_other_commands() -> None:
    assert _invalid_imports(list((_ROOT / "hal" / "scripts").glob("*.py")), "hal.scripts") == {}


def test_data_does_not_import_training() -> None:
    assert _invalid_imports(list((_ROOT / "hal" / "data").glob("*.py")), "hal.training") == {}
