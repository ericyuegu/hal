"""Native proxy parity fixture and bounded complete-update measurements.

Run this module under the shared GPU lock. Recording requires an explicit output;
the checked-in fixture is never updated by pytest.
"""

import argparse
import contextlib
import cProfile
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixtures") / "o65_columnar.json"
_SPEC = importlib.util.spec_from_file_location(
    "hal_o65_columnar_capture", os.environ.get("HAL_O65_CAPTURE_SOURCE", str(ROOT / "experiments/065_selfplay_ppo.py"))
)
assert _SPEC is not None and _SPEC.loader is not None
exp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = exp
_SPEC.loader.exec_module(exp)


class CaptureShard(exp.NativeSimulatorShard):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.record_digest = hashlib.sha256()
        self.action_digest = hashlib.sha256()

    def _frames(self, frame: Any) -> Any:
        self.record_digest.update(frame.records.tobytes())
        self.action_digest.update(exp.native_controller_readback(frame.applied_action).tobytes())
        return super()._frames(frame)


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def fingerprints(run: Any, batch: Any) -> dict[str, str]:
    # Hash frames once per decision's last observation, in decision order. This
    # includes both sides and catches padding, stale storage, and reset readback.
    records = torch.stack([r.state.frames[-1].device_record for r in batch.rollout.records])
    actions = torch.stack([r.state.frames[-1].device_actions for r in batch.rollout.records])
    return {
        "records": tensor_hash(records),
        "processed_actions": tensor_hash(actions),
        "sampled_factors": tensor_hash(batch.decisions.sampled_indices_JF),
        "behavior_log_probs": tensor_hash(batch.decisions.behavior_log_probs_JF),
        "rewards": exp._state_hash([r.rewards for r in batch.rollout.records]),
        "batch": exp._batch_hash(batch),
        "advantages": tensor_hash(batch.advantages),
        "values": tensor_hash(batch.values),
        "old_log_probs": tensor_hash(batch.old.log_probs),
        "reference_log_probs": tensor_hash(batch.reference.log_probs),
        "actor": exp._state_hash(tuple(run.policy.actor_parameters())),
        "critic": exp._state_hash(tuple(run.policy.critic_parameters())),
    }


def capture(matches: int, output: Path, *, performance: bool) -> dict[str, object]:
    if matches not in (16, 64):
        raise ValueError("local capture requires 16 or 64 matches")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(not performance)
    torch.manual_seed(65)
    checkpoint = exp.load_evaluation_checkpoint(ROOT / "runs/proxy/final.pt", device=torch.device("cuda"))
    checkpoint.model.requires_grad_(True)
    checkpoint.model.percentile_head.requires_grad_(False)
    schedule = exp.development_config(checkpoint.model.head_offsets)
    policy = exp.TransformerPolicy(checkpoint.model, schedule, checkpoint.statistics, compiled=True)
    sampling = exp.read_native_sampler(ROOT / "experiments/o65/pilot-sampler.json")
    training = exp.TrainConfig(critic_warmup_updates=0, deterministic=not performance)
    source = Path(os.environ.get("HAL_MSL_SOURCE", "~/src/melee-sim-light")).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    with contextlib.ExitStack() as resources:
        shards = []
        cpus = sorted(os.sched_getaffinity(0))[:4]
        for index, cpu in enumerate(cpus):
            ids = tuple(range(index * matches // 4, (index + 1) * matches // 4))
            shard = (exp.NativeSimulatorShard if performance else CaptureShard)(
                source / "data", ids, cpu=cpu, device=policy.device, max_frame=28800 if performance else 0
            )
            resources.callback(shard.close)
            shards.append(shard)
        engine = exp.RolloutEngine(
            (tuple(shards),),
            policy,
            exp.EpisodeSampler(sampling, schedule, exp.CounterRng(65)),
            {0: policy.frozen_copy()},
        )
        engine.initialize()
        run = exp.Trainer(engine, training)
        rows = []
        for _ in range(3 if performance else 2):
            batch, metrics = run.run_update()
            row = metrics if performance else fingerprints(run, batch)
            if not performance:
                row["all_frame_records"] = exp._state_hash([s.record_digest.hexdigest() for s in shards])
                row["all_processed_actions"] = exp._state_hash([s.action_digest.hexdigest() for s in shards])
            rows.append(row)
            if not performance:
                assert batch.rollout.resets > 0 and batch.rollout.timeouts > 0
            del batch
        if performance:
            profiler = cProfile.Profile()
            profiler.runcall(engine.collect, run.update)
            profiler.dump_stats(output / "rollout.prof")
        result = {
            "matches": matches,
            "performance": performance,
            "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "environment": exp._environment(),
            "checkpoint": checkpoint.checkpoint_sha256,
            "sampler": exp.sha256(ROOT / "experiments/o65/pilot-sampler.json"),
            "native_data": exp.sha256(source / "data/raw/manifest.json"),
            "simulator": shards[0].shard.library_sha256,
            "adapter": exp.native_bridge.library_sha256(),
            "rows": rows,
        }
        (output / "capture.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        return result


@pytest.mark.integration
@pytest.mark.parametrize("matches", [16, 64])
def test_native_columnar_exact_parity(matches: int, tmp_path: Path) -> None:
    expected = json.loads(FIXTURE.read_text())[str(matches)]
    actual = capture(matches, tmp_path, performance=False)
    for name in ("environment", "checkpoint", "sampler", "native_data", "simulator", "adapter", "rows"):
        assert actual[name] == expected[name], name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    capture(args.matches, args.output, performance=args.performance)
