import json
from pathlib import Path

import pytest

from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import Choice
from hal.netplay_service.domain import PolicyConfig
from hal.netplay_service.domain import account_connect_code
from hal.netplay_service.domain import validate_desired_return


def _config() -> PolicyConfig:
    return PolicyConfig(
        bundle_sha256="a" * 64,
        bundle_r2_key=f"netplay/policies/{'a' * 64}.halpolicy",
        vocabulary_sha256="b" * 64,
        characters=CHARACTERS,
        imitations=IMITATIONS,
        stages=STAGES,
        online_delays=(2, 3),
        desired_return_range=(0.0, 40.0),
        default_desired_return=20.0,
        temperature_range=(0.8, 1.1),
        default_temperature=1.0,
        masked_identity=False,
    )


def test_policy_config_round_trips_the_worker_payload() -> None:
    payload = _config().to_payload()
    assert payload["schema_version"] == 1
    assert payload["characters"][0] == {"value": "FOX", "label": "Fox"}  # type: ignore[index]
    assert payload["online_delays"] == [2, 3]
    assert PolicyConfig.from_payload(payload) == _config()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": 1}, "fields changed"),
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": 2}, "schema_version"),
        ({"bundle_sha256": "A" * 64}, "bundle_sha256"),
        ({"online_delays": [1]}, "online_delays"),
        ({"default_desired_return": 41.0}, "desired_return"),
        ({"characters": [{"value": "FOX"}]}, "value and label"),
        ({"masked_identity": 0}, "masked_identity"),
        ({"stages": [{"value": "CORNERIA", "label": "Corneria"}]}, "stages has unsupported values"),
        ({"characters": [{"value": "SANDBAG", "label": "Sandbag"}]}, "characters has unsupported values"),
        ({"temperature_range": [0.5, 1.1]}, "temperature must be in"),
        ({"desired_return_range": [0.0, 141.0]}, "desired_return must be in"),
    ],
)
def test_policy_config_rejects_drift(change: dict[str, object], message: str) -> None:
    payload = {**_config().to_payload(), **change}
    with pytest.raises(ValueError, match=message):
        PolicyConfig.from_payload(payload)


def test_policy_config_rejects_duplicate_choices() -> None:
    with pytest.raises(ValueError, match="unique"):
        PolicyConfig(**{**_fields(), "stages": (Choice("BATTLEFIELD", "Battlefield"),) * 2})


@pytest.mark.parametrize("value", [None, -20.0, 0.0, 20.0, 120.0, 140.0])
def test_desired_return_accepts_the_spec_range(value: float | None) -> None:
    assert validate_desired_return(value) == value


@pytest.mark.parametrize("value", [-20.01, 140.01, True, float("inf"), float("nan")])
def test_desired_return_rejects_invalid_targets(value: float) -> None:
    with pytest.raises(ValueError, match=r"\[-20, 140\]"):
        validate_desired_return(value)


def _fields() -> dict[str, object]:
    config = _config()
    return {name: getattr(config, name) for name in PolicyConfig.__dataclass_fields__}


def test_account_connect_code_is_read_without_fallback(tmp_path: Path) -> None:
    account = tmp_path / "user.json"
    account.write_text(json.dumps({"connectCode": "HAL#1", "playKey": "secret"}))
    assert account_connect_code(account) == "HAL#1"
    account.write_text(json.dumps({"connectCode": "hal#1"}))
    with pytest.raises(ValueError, match="exact uppercase"):
        account_connect_code(account)
    account.write_text("[]")
    with pytest.raises(ValueError, match="must contain an object"):
        account_connect_code(account)
