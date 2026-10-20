"""The comparison preserves match clusters and rejects changed inputs."""

import copy
from typing import Any

import pytest
from o65_statistical_comparison import compare


def records() -> tuple[dict[str, Any], dict[str, Any]]:
    common: dict[str, Any] = {
        key: "same"
        for key in (
            "seed",
            "max_frame",
            "checkpoint",
            "data_manifests",
            "environment",
            "sampling",
            "schedule",
            "simulator",
            "adapter",
            "simulator_manifest",
            "git_sha",
            "trainer_sha256",
        )
    }
    common["format"] = "hal.o65.dense-statistical-raw.v2"
    common["training"] = {}
    common["updates"] = [
        {
            "update": update,
            "matches": [
                {
                    "match": match,
                    "reward": 0.0,
                    "player_rewards": {"0": float(match), "1": -float(match)},
                    "kl": 0.1,
                    "entropy": 2.0,
                    "resets": 1,
                    "advantages": [1.0, 2.0],
                    "targets": [3.0, 4.0],
                    "episode_lengths": [512.0],
                }
                for match in range(64)
            ],
        }
        for update in range(3)
    ]
    control, dense = copy.deepcopy(common), copy.deepcopy(common)
    control["mode"], dense["mode"] = "control", "dense"
    control["training"]["importance_cap"] = 2
    return control, dense


def test_paired_match_intervals_preserve_three_update_cluster() -> None:
    control, dense = records()
    for update in dense["updates"]:
        for row in update["matches"]:
            row["player_rewards"] = {side: value + 2 for side, value in row["player_rewards"].items()}
    result = compare(control, dense)["statistics"]
    assert result["reward"]["mean_difference"] == 6
    assert result["reward"]["paired_match_bootstrap_95ci"] == [6, 6]
    for name in ("kl", "entropy", "advantages/mean", "targets/mean", "episode_length", "resets"):
        assert result[name]["paired_match_bootstrap_95ci"] == [0, 0]


def test_zero_sum_reward_changes_remain_visible_in_return_spread() -> None:
    control, dense = records()
    for update in dense["updates"]:
        for row in update["matches"]:
            row["player_rewards"]["0"] += 2
            row["player_rewards"]["1"] -= 2
    result = compare(control, dense)["statistics"]
    assert result["reward"]["paired_match_bootstrap_95ci"] == [0, 0]
    assert result["reward/std"]["paired_match_bootstrap_95ci"] == [6, 6]


@pytest.mark.parametrize("change", ["seed", "schedule", "training", "missing_match", "no_episodes", "student_sides"])
def test_comparison_rejects_incomplete_or_changed_evidence(change: str) -> None:
    control, dense = records()
    if change in ("seed", "schedule"):
        dense[change] = "changed"
    elif change == "training":
        dense["training"]["lr"] = 1
    elif change == "student_sides":
        dense["updates"][1]["matches"][0]["player_rewards"].pop("1")
    elif change == "missing_match":
        dense["updates"][0]["matches"].pop()
    else:
        for update in dense["updates"]:
            update["matches"][0]["episode_lengths"] = []
    with pytest.raises(ValueError):
        compare(control, dense)
