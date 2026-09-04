import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _maintained_experiments() -> list[Path]:
    paths: list[Path] = []
    for path in (_ROOT / "experiments").glob("[0-9][0-9][0-9]_*.py"):
        if int(path.name[:3]) >= 51:
            paths.append(path)
    return paths


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


def test_data_does_not_import_training() -> None:
    assert _invalid_imports(list((_ROOT / "hal" / "data").rglob("*.py")), "hal.training") == {}


def test_package_does_not_import_host_scripts() -> None:
    assert _invalid_imports(list((_ROOT / "hal").rglob("*.py")), "scripts") == {}


def test_package_does_not_import_experiments() -> None:
    assert _invalid_imports(list((_ROOT / "hal").rglob("*.py")), "experiments") == {}


def test_maintained_code_imports_only_public_hal_names() -> None:
    private_imports: dict[str, list[str]] = {}
    paths = [
        *(_ROOT / "hal").rglob("*.py"),
        *(_ROOT / "scripts").rglob("*.py"),
        *_maintained_experiments(),
    ]
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        names = sorted(
            f"{node.module}.{alias.name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None and node.module.startswith("hal.")
            for alias in node.names
            if alias.name.startswith("_")
        )
        if names:
            private_imports[str(path.relative_to(_ROOT))] = names
    assert private_imports == {}


def test_simulation_does_not_import_torch() -> None:
    assert _invalid_imports(list((_ROOT / "hal" / "sim").rglob("*.py")), "torch") == {}


def test_maintained_experiments_do_not_compose_other_experiments() -> None:
    paths = _maintained_experiments()
    assert _invalid_imports(paths, "experiments") == {}
    assert _invalid_imports(paths, "importlib") == {}
    for path in paths:
        tree = ast.parse(path.read_text(), filename=str(path))
        module_getattr = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == "__getattr__"
        ]
        assert module_getattr == [], f"{path.relative_to(_ROOT)} defines module __getattr__"
