"""Versioned, hash-validated policy bundles."""

import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Iterator
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_FORMAT_VERSION: Final[int] = 1
_MANIFEST_NAME: Final[str] = "manifest.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class BundleMember:
    """Identity of one immutable file inside a policy bundle."""

    name: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("bundle member name must be a string")
        if Path(self.name).name != self.name or self.name == _MANIFEST_NAME:
            raise ValueError(f"bundle member must be a plain non-manifest filename, got {self.name!r}")
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 0:
            raise ValueError(f"bundle member size must be a non-negative integer, got {self.size!r}")
        if not isinstance(self.sha256, str) or _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError(f"bundle member SHA-256 is invalid: {self.sha256!r}")


@dataclass(frozen=True, slots=True)
class PolicyBundleManifest:
    """Generic bundle metadata understood without importing a model backend."""

    policy_name: str
    backend: str
    backend_version: int
    action_schema: str
    observation_schema: str
    required_observation_fields: tuple[str, ...]
    supported_transport_delays: tuple[int, ...]
    requires_player_code: bool
    source_sha256: str
    backend_config: str
    members: tuple[BundleMember, ...]
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.format_version, int)
            or isinstance(self.format_version, bool)
            or self.format_version != _FORMAT_VERSION
        ):
            raise ValueError(f"unsupported policy bundle format {self.format_version}; expected {_FORMAT_VERSION}")
        for label, value in (
            ("policy_name", self.policy_name),
            ("backend", self.backend),
            ("backend_config", self.backend_config),
            ("action_schema", self.action_schema),
            ("observation_schema", self.observation_schema),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"bundle {label} must be non-empty")
        if (
            not isinstance(self.backend_version, int)
            or isinstance(self.backend_version, bool)
            or self.backend_version < 1
        ):
            raise ValueError("bundle backend_version must be a positive integer")
        if not isinstance(self.source_sha256, str) or _SHA256_PATTERN.fullmatch(self.source_sha256) is None:
            raise ValueError("bundle source_sha256 must be a lowercase SHA-256")
        if not isinstance(self.requires_player_code, bool):
            raise ValueError("bundle requires_player_code must be a boolean")
        if any(not isinstance(name, str) or not name for name in self.required_observation_fields):
            raise ValueError("bundle observation fields must be non-empty strings")
        if any(
            not isinstance(delay, int) or isinstance(delay, bool) or delay < 0
            for delay in self.supported_transport_delays
        ):
            raise ValueError("bundle transport delays must be non-negative integers")
        if tuple(sorted(set(self.supported_transport_delays))) != self.supported_transport_delays:
            raise ValueError("bundle transport delays must be sorted and unique")
        if len(set(self.required_observation_fields)) != len(self.required_observation_fields):
            raise ValueError("bundle observation fields must be unique")
        if any(not isinstance(member, BundleMember) for member in self.members):
            raise ValueError("bundle members must be BundleMember values")
        names = tuple(member.name for member in self.members)
        if len(set(names)) != len(names):
            raise ValueError("bundle member names must be unique")
        if self.backend_config not in names:
            raise ValueError("bundle backend_config must name a bundle member")

    def to_json(self) -> bytes:
        """Return the canonical persisted representation."""
        payload = {
            "action_schema": self.action_schema,
            "backend": self.backend,
            "backend_config": self.backend_config,
            "backend_version": self.backend_version,
            "format_version": self.format_version,
            "members": [
                {"name": member.name, "sha256": member.sha256, "size": member.size} for member in self.members
            ],
            "observation_schema": self.observation_schema,
            "policy_name": self.policy_name,
            "required_observation_fields": list(self.required_observation_fields),
            "requires_player_code": self.requires_player_code,
            "source_sha256": self.source_sha256,
            "supported_transport_delays": list(self.supported_transport_delays),
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()

    @classmethod
    def from_json(cls, encoded: bytes) -> PolicyBundleManifest:
        """Parse a manifest and reject missing, extra, or mistyped fields."""
        try:
            payload = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("policy bundle manifest is not valid JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("policy bundle manifest must be an object")
        expected = {
            "action_schema",
            "backend",
            "backend_config",
            "backend_version",
            "format_version",
            "members",
            "observation_schema",
            "policy_name",
            "required_observation_fields",
            "requires_player_code",
            "source_sha256",
            "supported_transport_delays",
        }
        if set(payload) != expected:
            raise ValueError(
                f"policy bundle manifest fields differ: missing={sorted(expected - payload.keys())}, "
                f"unexpected={sorted(payload.keys() - expected)}"
            )
        members_raw = payload["members"]
        if not isinstance(members_raw, list):
            raise ValueError("policy bundle members must be a list")
        members = []
        for raw in members_raw:
            if not isinstance(raw, dict) or set(raw) != {"name", "sha256", "size"}:
                raise ValueError("policy bundle member entries have the wrong fields")
            members.append(BundleMember(name=raw["name"], size=raw["size"], sha256=raw["sha256"]))
        fields_raw = payload["required_observation_fields"]
        delays_raw = payload["supported_transport_delays"]
        if not isinstance(fields_raw, list) or any(not isinstance(value, str) for value in fields_raw):
            raise ValueError("required_observation_fields must be a list of strings")
        if not isinstance(delays_raw, list):
            raise ValueError("supported_transport_delays must be a list")
        string_fields = (
            "action_schema",
            "backend",
            "backend_config",
            "observation_schema",
            "policy_name",
            "source_sha256",
        )
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise ValueError("policy bundle string fields have invalid types")
        if not isinstance(payload["requires_player_code"], bool):
            raise ValueError("requires_player_code must be a boolean")
        return cls(
            policy_name=payload["policy_name"],
            backend=payload["backend"],
            backend_version=payload["backend_version"],
            action_schema=payload["action_schema"],
            observation_schema=payload["observation_schema"],
            required_observation_fields=tuple(fields_raw),
            supported_transport_delays=tuple(delays_raw),
            requires_player_code=payload["requires_player_code"],
            source_sha256=payload["source_sha256"],
            backend_config=payload["backend_config"],
            members=tuple(members),
            format_version=payload["format_version"],
        )


@dataclass(frozen=True, slots=True)
class BundleDescription:
    """Manifest fields supplied before member hashes are known."""

    policy_name: str
    backend: str
    backend_version: int
    required_observation_fields: tuple[str, ...]
    supported_transport_delays: tuple[int, ...]
    requires_player_code: bool
    source_sha256: str
    backend_config: str = "backend.json"
    action_schema: str = "hal.controller.v1"
    observation_schema: str = "hal.flat.numeric.v1"


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _zip_info(name: str, *, compressed: bool) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED if compressed else zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def write_policy_bundle(
    destination: str | Path,
    description: BundleDescription,
    members: Mapping[str, str | Path],
) -> PolicyBundleManifest:
    """Write a deterministic bundle and atomically replace its destination."""
    destination_path = Path(destination)
    if not members:
        raise ValueError("policy bundle needs at least one member")
    member_paths = {name: Path(path) for name, path in members.items()}
    identities = []
    for name, path in sorted(member_paths.items()):
        size, sha256 = _file_identity(path)
        identities.append(BundleMember(name=name, size=size, sha256=sha256))
    manifest = PolicyBundleManifest(
        policy_name=description.policy_name,
        backend=description.backend,
        backend_version=description.backend_version,
        action_schema=description.action_schema,
        observation_schema=description.observation_schema,
        required_observation_fields=description.required_observation_fields,
        supported_transport_delays=description.supported_transport_delays,
        requires_player_code=description.requires_player_code,
        source_sha256=description.source_sha256,
        backend_config=description.backend_config,
        members=tuple(identities),
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_suffix(destination_path.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            archive.writestr(_zip_info(_MANIFEST_NAME, compressed=True), manifest.to_json())
            for member in manifest.members:
                with (
                    member_paths[member.name].open("rb") as source,
                    archive.open(_zip_info(member.name, compressed=False), "w", force_zip64=True) as output,
                ):
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
        os.replace(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def read_policy_manifest(path: str | Path) -> PolicyBundleManifest:
    """Read the manifest without loading model assets."""
    try:
        with zipfile.ZipFile(path) as archive:
            return PolicyBundleManifest.from_json(archive.read(_MANIFEST_NAME))
    except (KeyError, zipfile.BadZipFile) as error:
        raise ValueError(f"invalid policy bundle {path}") from error


@contextmanager
def extract_policy_bundle(path: str | Path) -> Iterator[tuple[PolicyBundleManifest, Path]]:
    """Validate every byte and expose members in a temporary directory."""
    bundle_path = Path(path)
    try:
        archive = zipfile.ZipFile(bundle_path)
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"cannot open policy bundle {bundle_path}: {error}") from error
    with archive, tempfile.TemporaryDirectory(prefix="hal-policy-") as temporary:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(set(names)) != len(names):
            raise ValueError("policy bundle contains duplicate member names")
        try:
            manifest = PolicyBundleManifest.from_json(archive.read(_MANIFEST_NAME))
        except KeyError as error:
            raise ValueError("policy bundle has no manifest.json") from error
        expected = {_MANIFEST_NAME, *(member.name for member in manifest.members)}
        if set(names) != expected:
            raise ValueError(
                f"policy bundle members differ: missing={sorted(expected - set(names))}, "
                f"unexpected={sorted(set(names) - expected)}"
            )
        root = Path(temporary)
        for member in manifest.members:
            digest = hashlib.sha256()
            size = 0
            destination = root / member.name
            with archive.open(member.name) as source, destination.open("wb") as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
                    output.write(chunk)
            if size != member.size or digest.hexdigest() != member.sha256:
                raise ValueError(f"policy bundle member {member.name!r} failed size or SHA-256 validation")
        yield manifest, root
