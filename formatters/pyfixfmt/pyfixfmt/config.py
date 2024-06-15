from pathlib import Path
from typing import Any
from typing import Final
from typing import List
from typing import MutableMapping
from typing import Optional
from typing import Tuple
from typing import Union

import black
import isort
import toml

DEFAULT_PYPROJECT_TOML: Final = Path("./pyproject.toml")


def attempt_loading_toml(path: Path, is_failure_ok: bool = True) -> Optional[MutableMapping[str, Any]]:
    try:
        return toml.load(path)
    except (toml.TomlDecodeError, OSError, FileNotFoundError) as e:
        if is_failure_ok:
            return None
        raise e


class Config:
    def __init__(self, pyproject_config: Optional[MutableMapping[str, Any]] = None) -> None:
        self.pyproject_config = pyproject_config or {}

    def _read_config_path(self, selection: Tuple[str, ...]) -> MutableMapping[str, Any]:
        selected = self.pyproject_config
        for key in selection:
            if key not in selected:
                return {}
            selected = selected[key]
        return selected

    @property
    def formatters_config(self) -> MutableMapping[str, Any]:
        return self._read_config_path(("tool", "formatters", "python"))

    @property
    def do_not_remove_imports(self) -> List[str]:
        return self.formatters_config.get("do_not_remove_imports", ["__init__.py"])

    @property
    def do_not_do_anything_with_imports(self) -> List[str]:
        return self.formatters_config.get("do_not_do_anything_with_imports", [])

    @property
    def do_not_autotype(self) -> List[str]:
        return self.formatters_config.get("do_not_autotype", [])

    @property
    def isort_config(self) -> isort.api.Config:
        config = self._read_config_path(("tool", "isort"))
        return isort.api.Config(**config)

    @property
    def black_config(self) -> black.FileMode:
        config = self._read_config_path(("tool", "black"))
        default_config = {
            "target_versions": ["PY39", "PY311"],
            "line_length": 119,
        }
        for key, default_value in default_config.items():
            if key not in config:
                config[key] = default_value
        return black.FileMode(
            target_versions={black.TargetVersion[x] for x in config["target_versions"]},
            line_length=config["line_length"],
        )

    def disable_import_flaking(self) -> None:
        # The way this check is implemented, adding an empty string to it will
        # match all files, so this has the effect of disabling it everywhere
        self.formatters_config["do_not_remove_imports"] = [""]


def resolve_config(source_root: Path, explicit_path: Optional[Union[str, Path]], is_verbose: bool) -> Config:
    if is_verbose:
        print("Attempting to load a config for pyfixfmt.")

    if explicit_path:
        if is_verbose:
            print(f"Using configured --config {explicit_path}")
        return Config(attempt_loading_toml(Path(explicit_path), is_failure_ok=False))

    config_path = Path(source_root / "pyproject.toml")
    using_default_message = "Will use default isort and black configurations"

    if not (config_path.exists() and config_path.is_file()):
        if is_verbose:
            print(f"Cannot find toml file in {source_root}. {using_default_message}")
        return Config()

    if is_verbose:
        print(f"Discovered pyproject.toml: {config_path}")

    pyproject_toml = attempt_loading_toml(
        config_path,
        is_failure_ok=True,
    )
    if is_verbose and pyproject_toml is None:
        print(f"Failed to parse {config_path}. {using_default_message}")

    return Config(pyproject_toml)
