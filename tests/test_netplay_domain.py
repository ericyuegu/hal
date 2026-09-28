import pytest

from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import Choice
from hal.netplay_service.domain import PolicyConfig


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
    ],
)
def test_policy_config_rejects_drift(change: dict[str, object], message: str) -> None:
    payload = {**_config().to_payload(), **change}
    with pytest.raises(ValueError, match=message):
        PolicyConfig.from_payload(payload)


def test_policy_config_rejects_duplicate_choices() -> None:
    with pytest.raises(ValueError, match="unique"):
        PolicyConfig(**{**_fields(), "stages": (Choice("BATTLEFIELD", "Battlefield"),) * 2})


def _fields() -> dict[str, object]:
    config = _config()
    return {name: getattr(config, name) for name in PolicyConfig.__dataclass_fields__}
