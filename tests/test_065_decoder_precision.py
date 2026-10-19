"""Persisted decoder precision identity checks."""

import dataclasses
import io
import json
import os
from pathlib import Path

import pytest
import torch
from test_065_loop import exp


def test_resume_rejects_a_changed_decoder_precision(tmp_path: Path) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp as loop_exp
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cpu", decoder_precision="bf16")
    manifest = loop_exp.save_checkpoint(run, tmp_path, PROVENANCE)
    other = transformer_trainer("cpu", decoder_precision="fp32")
    assert loop_exp._configuration_digest(run) != loop_exp._configuration_digest(other)
    with pytest.raises(ValueError, match="configuration mismatch"):
        loop_exp.restore_checkpoint(other, manifest, PROVENANCE)
    assert run.policy.frozen_copy().model.decoder_precision == "bf16"


def test_native_precision_is_validated_and_profile_preserves_it(tmp_path: Path) -> None:
    cpu = min(os.sched_getaffinity(0))
    profile = exp.GpuProfileCli(tmp_path, tmp_path, tmp_path, cpus=(cpu,), decoder_precision="fp32")
    assert profile.native().decoder_precision == "fp32"
    assert dataclasses.replace(profile, decoder_precision="bf16").native().decoder_precision == "bf16"
    with pytest.raises(ValueError, match="decoder precision"):
        dataclasses.replace(profile.native(), decoder_precision="fp16")


def test_resume_rejects_a_mismatched_actor_precision_before_model_mutation(tmp_path: Path) -> None:
    from test_065_loop import PROVENANCE
    from test_065_loop import exp as loop_exp
    from test_065_loop import transformer_trainer

    run = transformer_trainer("cpu", decoder_precision="bf16")
    manifest = loop_exp.save_checkpoint(run, tmp_path, PROVENANCE)
    raw = json.loads(manifest.read_text())
    shard = next(shard for shard in raw["shards"] if shard["role"] == "actor")
    actor = torch.load(tmp_path / shard["name"], weights_only=False)
    actor["decoder_precision"] = "fp32"
    payload = io.BytesIO()
    torch.save(actor, payload)
    shard["name"], shard["sha256"] = loop_exp._durable_file(tmp_path, payload.getvalue(), "actor.pt")
    manifest.write_text(json.dumps(raw))
    before = tuple(parameter.detach().clone() for parameter in run.policy.model.parameters())
    with pytest.raises(ValueError, match="actor precision"):
        loop_exp.restore_checkpoint(run, manifest, PROVENANCE)
    for actual, expected in zip(run.policy.model.parameters(), before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
