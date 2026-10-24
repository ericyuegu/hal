"""Read warm Triton 3.6.0 kernels through a local disk cache on Modal."""

import json
import os
import shutil
from pathlib import Path

import triton
from triton.runtime.cache import FileCacheManager


class CacheOnlyManager(FileCacheManager):
    """Reject a Triton 3.6.0 kernel miss before its LLVM compilation starts.

    Driver glue can still compile through get_file. Kernel groups must be complete
    and local. This uses Triton's cache-manager extension rather than a patch.
    """

    def __init__(self, key: str, override: bool = False, dump: bool = False) -> None:
        if triton.__version__ != "3.6.0":
            raise RuntimeError("cache-only kernels require tested Triton 3.6.0")
        super().__init__(key, override=override, dump=dump)

    def get_group(self, filename: str) -> dict[str, str]:
        group = super().get_group(filename)
        if group is None or filename not in group:
            raise RuntimeError(f"required Triton kernel cache miss: {self.key}/{filename}")
        entry = json.loads((Path(self.cache_dir) / f"__grp__{filename}").read_text())
        children = entry.get("child_paths")
        if not isinstance(children, dict) or set(group) != set(children):
            raise ValueError("required Triton kernel cache group is incomplete")
        if any(Path(path).parent != Path(self.cache_dir) for path in group.values()):
            raise ValueError("required Triton kernel cache group is not local")
        return group


class LocalCacheManager(FileCacheManager):
    """Copy only requested artifacts from the read-only warm cache.

    Triton 3.6.0 groups contain absolute paths. Copying a group file alone
    leaves kernel reads on the network volume. Rebuild groups with local paths.
    Remove this adapter when Triton provides a local read-through file cache.
    """

    def __init__(self, key: str, override: bool = False, dump: bool = False) -> None:
        if triton.__version__ != "3.6.0":
            raise RuntimeError("local Triton cache requires tested Triton 3.6.0")
        super().__init__(key, override=override, dump=dump)
        self.source = None if override or dump else Path(os.environ["HAL_TRITON_CACHE_SOURCE"]) / key
        if self.source is not None and self.source.resolve() == Path(self.cache_dir).resolve():
            raise ValueError("local Triton cache must differ from its warm source")

    def get_file(self, filename: str) -> str | None:
        local = super().get_file(filename)
        if local is not None or self.source is None:
            return local
        source = self.source / filename
        if not source.is_file():
            return None
        target = Path(self.cache_dir) / filename
        temporary = target.with_name(f".{target.name}.{os.getpid()}.partial")
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return str(target)

    def get_group(self, filename: str) -> dict[str, str] | None:
        local = super().get_group(filename)
        if local is not None or self.source is None:
            return local
        source = self.source / f"__grp__{filename}"
        if not source.is_file():
            return None
        entry = json.loads(source.read_text())
        if not isinstance(entry, dict) or set(entry) != {"child_paths"}:
            raise ValueError("invalid Triton cache group")
        children = entry["child_paths"]
        if not isinstance(children, dict) or not children:
            raise ValueError("invalid Triton cache group")
        group: dict[str, str] = {}
        for name, path in children.items():
            if not isinstance(name, str) or not isinstance(path, str) or Path(path).parent != self.source:
                raise ValueError("Triton cache group points outside its kernel directory")
            copied = self.get_file(Path(path).name)
            if copied is None:
                raise ValueError(f"Triton cache group is missing {name}")
            group[name] = copied
        self.put_group(filename, group)
        return group
