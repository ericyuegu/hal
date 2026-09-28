"""Record the Python netplay service's behavior as JSON transcripts.

The Worker port in web/netplay-api replays these transcripts and must produce
the same responses. This script and its transcripts are the parity contract
for the port; delete them with the Python service.
"""

import argparse
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from hal.netplay_service.api import ApiConfig
from hal.netplay_service.api import create_app
from hal.netplay_service.domain import CHARACTERS
from hal.netplay_service.domain import IMITATIONS
from hal.netplay_service.domain import STAGES
from hal.netplay_service.domain import Job
from hal.netplay_service.queue import InvalidTransitionError
from hal.netplay_service.queue import QueueStore
from hal.paths import REPO_DIR

START = 1_000_000.0
LEASE_SECONDS = 20.0
BOT_CODE = "HALBOT#1"
DEFAULT_OUTPUT = Path(REPO_DIR) / "web" / "netplay-api" / "test" / "transcripts"
_COMPARE = {
    "claim": "full",
    "get": "full",
    "end-session": "full",
    "finish-game": "status_field",
    "fail": "status_field",
}


def _job_body(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "player_code": job.player_code,
        "character": job.choices.character,
        "imitation": job.choices.imitation,
        "online_delay": job.choices.online_delay,
        "desired_return": job.choices.desired_return,
        "temperature": job.choices.temperature,
        "policy_revision": job.policy_revision,
        "requested_stage": job.choices.requested_stage,
        "status": job.status.value,
        "queue_position": job.queue_position,
        "attempt": job.attempt,
        "game_count": job.game_count,
        "connect_code": job.connect_code,
        "actual_stage": job.actual_stage,
        "last_result": job.last_result,
        "error_code": job.error_code,
        "connect_deadline": job.connect_deadline,
        "rematch_deadline": job.rematch_deadline,
        "cancel_after_game": job.cancel_after_game,
    }


class Recorder:
    def __init__(self, root: Path, name: str) -> None:
        self.name = name
        self.clock = START
        self.steps: list[dict[str, Any]] = []
        self.aliases: dict[str, str] = {}
        database = root / f"{name}.sqlite3"
        self.store = QueueStore(database, now=lambda: self.clock)
        # No `with` block: the app's background reaper must not run, so expiry
        # happens only at explicit `advance` steps.
        self.client = TestClient(create_app(ApiConfig(database, allowed_hosts=("testserver",)), self.store))

    def _alias(self, value: str, prefix: str) -> str:
        if value not in self.aliases:
            count = sum(alias.startswith(f"${prefix}") for alias in self.aliases.values()) + 1
            self.aliases[value] = f"${prefix}{count}"
        return self.aliases[value]

    def _normalize(self, body: Any) -> Any:
        if isinstance(body, dict):
            if isinstance(body.get("id"), str):
                self._alias(body["id"], "job")
            if isinstance(body.get("token"), str):
                self._alias(body["token"], "token")
            return {key: self._normalize(value) for key, value in body.items()}
        if isinstance(body, list):
            return [self._normalize(value) for value in body]
        if isinstance(body, str):
            return self.aliases.get(body, body)
        return body

    def _real(self, text: str) -> str:
        for real, alias in sorted(self.aliases.items(), key=lambda item: -len(item[1])):
            text = text.replace(alias, real)
        return text

    def player(self, method: str, path: str, *, token: str | None = None, body: Any = None) -> Any:
        headers = {} if token is None else {"Authorization": f"Bearer {self._real(token)}"}
        response = self.client.request(method, self._real(path), headers=headers, json=body)
        payload = self._normalize(response.json() if response.content else None)
        self.steps.append(
            {
                "kind": "player",
                "method": method,
                "path": path,
                "token": token,
                "body": body,
                "response": {"status": response.status_code, "body": payload},
            }
        )
        return payload

    def worker(self, worker: str, op: str, job: str | None = None, **args: Any) -> Any:
        try:
            status, body = self._call(worker, op, None if job is None else self._real(job), args)
        except InvalidTransitionError as error:
            status, body = 409, {"detail": str(error)}
        except ValueError as error:
            status, body = 422, {"detail": str(error)}
        payload = self._normalize(body)
        self.steps.append(
            {
                "kind": "worker",
                "worker": worker,
                "op": op,
                "job": job,
                "args": args,
                "compare": _COMPARE.get(op, "status"),
                "response": {"status": status, "body": payload},
            }
        )
        return payload

    def _call(self, worker: str, op: str, job: str | None, args: dict[str, Any]) -> tuple[int, Any]:
        owner = f"worker-{worker}"
        if op == "claim":
            claimed = self.store.claim_next(owner, lease_seconds=LEASE_SECONDS)
            return (204, None) if claimed is None else (200, _job_body(claimed))
        if op == "end-session":
            owners = [f"worker-{worker}:{slot}" for slot in range(args["slots"])]
            return 200, {"failed": self.store.fail_worker_generation(owners)}
        assert job is not None
        if op == "heartbeat":
            self.store.heartbeat(job, owner, lease_seconds=LEASE_SECONDS)
        elif op == "connecting":
            self.store.mark_connecting(job, owner, args["connect_code"])
        elif op == "playing":
            self.store.mark_playing(job, owner)
        elif op == "no-show":
            self.store.mark_no_show(job, owner)
        elif op == "no-contest":
            self.store.mark_no_contest(job, owner)
        elif op == "forfeit":
            self.store.forfeit_service_failure(job, owner)
        elif op == "finish-game":
            status = self.store.finish_game(job, owner, actual_stage=args["actual_stage"], result=args["result"])
            return 200, {"status": status.value}
        elif op == "fail":
            status = self.store.fail(job, owner, args["error_code"], retryable=args["retryable"])
            return 200, {"status": status.value}
        elif op == "replay":
            self.store.record_replay(
                job,
                args["game_number"],
                key=args["key"],
                sha256=args["sha256"],
                size=args["size"],
                etag=args["etag"],
            )
        elif op == "get":
            return 200, _job_body(self.store.get_worker_job(job, owner))
        else:
            raise AssertionError(op)
        return 200, None

    def advance(self, seconds: float) -> None:
        self.clock += seconds
        self.store.reap_expired()
        self.steps.append({"kind": "advance", "seconds": seconds})

    def create(self, player_code: str = "CRYO#610", **overrides: Any) -> Any:
        body = {"player_code": player_code, "character": "FOX", "imitation": "IBDW#0", "online_delay": 2}
        body.update(overrides)
        return self.player("POST", "/v1/jobs", body=body)

    def start_game(self, worker: str, job: str) -> None:
        self.worker(worker, "claim")
        self.worker(worker, "connecting", job, connect_code=BOT_CODE)
        self.worker(worker, "playing", job)


def create_poll_cancel(r: Recorder) -> None:
    job = r.create()
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.player("DELETE", f"/v1/jobs/{job['id']}", token=job["token"])
    r.player("DELETE", f"/v1/jobs/{job['id']}", token=job["token"])


def policy_revision(r: Recorder) -> None:
    job = r.create(desired_return=None, temperature=0.9)
    path = f"/v1/jobs/{job['id']}/policy"
    r.player("PATCH", path, token=job["token"], body={"desired_return": 30})
    r.player("PATCH", path, token=job["token"], body={"temperature": 1.05})
    r.player("PATCH", path, token=job["token"], body={"desired_return": None})
    r.player("PATCH", path, token=job["token"], body={})
    r.player("PATCH", path, token=job["token"], body={"temperature": None})
    r.player("DELETE", f"/v1/jobs/{job['id']}", token=job["token"])
    r.player("PATCH", path, token=job["token"], body={"desired_return": 10})


def credentials_hidden(r: Recorder) -> None:
    job = r.create()
    r.player("GET", f"/v1/jobs/{job['id']}", token="wrong-token")
    r.player("GET", "/v1/jobs/unknown-job-id", token=job["token"])
    r.player("GET", f"/v1/jobs/{job['id']}")
    r.player("DELETE", f"/v1/jobs/{job['id']}", token="wrong-token")


def one_active_per_player(r: Recorder) -> None:
    job = r.create()
    r.create()
    r.player("DELETE", f"/v1/jobs/{job['id']}", token=job["token"])
    r.create()


def fifo_and_retry_front(r: Recorder) -> None:
    first = r.create("AAAA#1")
    second = r.create("BBBB#2")
    third = r.create("CCCC#3")
    r.worker("a:0", "claim")
    r.player("GET", f"/v1/jobs/{second['id']}", token=second["token"])
    r.player("GET", f"/v1/jobs/{third['id']}", token=third["token"])
    r.worker("a:0", "fail", first["id"], error_code="dolphin_crash", retryable=True)
    r.player("GET", f"/v1/jobs/{first['id']}", token=first["token"])
    r.player("GET", f"/v1/jobs/{second['id']}", token=second["token"])
    r.worker("a:1", "claim")
    r.worker("a:1", "fail", first["id"], error_code="dolphin_crash", retryable=True)
    r.player("GET", f"/v1/jobs/{first['id']}", token=first["token"])
    r.worker("a:1", "claim")
    r.worker("a:0", "fail", second["id"], error_code="bad_state", retryable=False)
    r.player("GET", f"/v1/jobs/{third['id']}", token=third["token"])


def validation_errors(r: Recorder) -> None:
    r.create("CR")
    r.create("cryo#610")
    r.create(character="WALUIGI")
    r.create(imitation="NOBODY#0")
    r.create(online_delay=4)
    r.create(imitation="MASKED")
    r.create(desired_return=50)
    r.create(temperature=2.0)
    r.create(extra=True)
    r.player("POST", "/v1/jobs", body={"player_code": "CRYO#610"})


def connect_deadline_no_show(r: Recorder) -> None:
    job = r.create()
    r.worker("a:0", "claim")
    r.worker("a:0", "connecting", job["id"], connect_code=BOT_CODE)
    # Heartbeats keep the 20 s lease alive so the connect deadline is what expires.
    for _ in range(31):
        r.advance(19)
        r.worker("a:0", "heartbeat", job["id"])
    r.advance(10)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.advance(1)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.worker("a:0", "heartbeat", job["id"])
    r.create()


def worker_no_show(r: Recorder) -> None:
    job = r.create()
    r.worker("a:0", "claim")
    r.worker("a:0", "connecting", job["id"], connect_code=BOT_CODE)
    r.worker("a:0", "no-show", job["id"])
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])


def full_set(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    stages = ["BATTLEFIELD", "FINAL_DESTINATION", "DREAMLAND", "POKEMON_STADIUM", "YOSHIS_STORY"]
    for number, stage in enumerate(stages, start=1):
        r.worker("a:0", "heartbeat", job["id"])
        r.worker("a:0", "finish-game", job["id"], game_number=number, actual_stage=stage, result="loss")
        r.worker(
            "a:0",
            "replay",
            job["id"],
            game_number=number,
            key=f"replays/{number}.slp",
            sha256="ab" * 32,
            size=1000 + number,
            etag=f"etag-{number}",
        )
        r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
        if number < len(stages):
            r.player(
                "POST",
                f"/v1/jobs/{job['id']}/rematch",
                token=job["token"],
                body={"character": "FALCO", "imitation": "ZAIN#0", "stage": "BATTLEFIELD"},
            )
            r.worker("a:0", "playing", job["id"])
    r.worker("a:0", "get", job["id"])


def rematch_rules(r: Recorder) -> None:
    job = r.create(online_delay=3)
    rematch = f"/v1/jobs/{job['id']}/rematch"
    choice = {"character": "MARTH", "imitation": "MANG#0", "stage": "DREAMLAND"}
    r.player("POST", rematch, token=job["token"], body=choice)
    r.start_game("a:0", job["id"])
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    r.player("POST", rematch, token=job["token"], body={**choice, "stage": "HYRULE"})
    r.player("POST", rematch, token=job["token"], body=choice)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.worker("a:0", "playing", job["id"])
    r.worker("a:0", "finish-game", job["id"], game_number=2, actual_stage="DREAMLAND", result="tie")
    r.advance(600)
    r.player("POST", rematch, token=job["token"], body=choice)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])


def rematch_timeout(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    # Heartbeats keep the 20 s lease alive so the rematch deadline is what expires.
    for _ in range(31):
        r.advance(19)
        r.worker("a:0", "heartbeat", job["id"])
    r.advance(10)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.advance(1)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])


def cancel_during_play(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    r.player("DELETE", f"/v1/jobs/{job['id']}", token=job["token"])
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    queued = r.create("QUEUE#9")
    r.worker("a:0", "claim")
    r.player("DELETE", f"/v1/jobs/{queued['id']}", token=queued["token"])
    r.worker("a:0", "heartbeat", queued["id"])


def no_contest(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    r.worker("a:0", "no-contest", job["id"])
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])


def replay_recording(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    r.worker("a:0", "replay", job["id"], game_number=1, key="k", sha256="cd" * 32, size=5, etag="e")
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    replay = {"game_number": 1, "key": "k", "sha256": "cd" * 32, "size": 5, "etag": "e"}
    r.worker("a:0", "replay", job["id"], **replay)
    r.worker("a:0", "replay", job["id"], **replay)
    r.worker("a:0", "replay", job["id"], **{**replay, "etag": "other"})


def service_forfeit(r: Recorder) -> None:
    job = r.create()
    r.start_game("a:0", job["id"])
    r.worker("a:0", "forfeit", job["id"])
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    leased = r.create("LEASE#2")
    r.worker("a:0", "claim")
    r.worker("a:0", "forfeit", leased["id"])


def end_session(r: Recorder) -> None:
    playing = r.create("PLAY#1")
    leased = r.create("LEASE#2")
    other = r.create("OTHER#3")
    done = r.create("DONE#4")
    r.start_game("a:0", playing["id"])
    r.worker("a:1", "claim")
    r.start_game("b:0", other["id"])
    r.worker("b:0", "finish-game", other["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    r.player("DELETE", f"/v1/jobs/{done['id']}", token=done["token"])
    r.worker("a", "end-session", slots=2)
    for job in (playing, leased, other):
        r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])


def lease_expiry(r: Recorder) -> None:
    job = r.create()
    r.worker("a:0", "claim")
    r.advance(19)
    r.worker("a:0", "heartbeat", job["id"])
    r.advance(20)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.worker("a:1", "claim")
    r.advance(20)
    r.player("GET", f"/v1/jobs/{job['id']}", token=job["token"])
    r.worker("a:1", "heartbeat", job["id"])


def ownership_and_state(r: Recorder) -> None:
    job = r.create()
    r.worker("a:1", "connecting", job["id"], connect_code=BOT_CODE)
    r.worker("a:0", "claim")
    r.worker("a:1", "connecting", job["id"], connect_code=BOT_CODE)
    r.worker("a:0", "playing", job["id"])
    r.worker("a:0", "connecting", job["id"], connect_code="bad code")
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="BATTLEFIELD", result="win")
    r.worker("a:0", "finish-game", job["id"], game_number=1, actual_stage="HYRULE", result="win")
    r.worker("a:0", "no-contest", job["id"])
    r.worker("a:0", "fail", job["id"], error_code="", retryable=True)
    r.worker("a:1", "get", job["id"])
    r.worker("a:0", "get", job["id"])


SCENARIOS: dict[str, Callable[[Recorder], None]] = {
    function.__name__: function
    for function in (
        create_poll_cancel,
        policy_revision,
        credentials_hidden,
        one_active_per_player,
        fifo_and_retry_front,
        validation_errors,
        connect_deadline_no_show,
        worker_no_show,
        full_set,
        rematch_rules,
        rematch_timeout,
        cancel_during_play,
        no_contest,
        replay_recording,
        service_forfeit,
        end_session,
        lease_expiry,
        ownership_and_state,
    )
}


def policy_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "bundle_sha256": "0" * 64,
        "bundle_r2_key": "netplay/policies/transcripts.halpolicy",
        "vocabulary_sha256": "1" * 64,
        "characters": [{"value": choice.value, "label": choice.label} for choice in CHARACTERS],
        "imitations": [{"value": choice.value, "label": choice.label} for choice in IMITATIONS],
        "stages": [{"value": choice.value, "label": choice.label} for choice in STAGES],
        "online_delays": [2, 3],
        "desired_return_range": [0.0, 40.0],
        "default_desired_return": 20.0,
        "temperature_range": [0.8, 1.1],
        "default_temperature": 1.0,
        "masked_identity": False,
    }


def record_all(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        for name, scenario in SCENARIOS.items():
            recorder = Recorder(Path(scratch), name)
            scenario(recorder)
            transcript = {"name": name, "steps": recorder.steps}
            (output / f"{name}.json").write_text(json.dumps(transcript, indent=2, sort_keys=True) + "\n")
    (output / "policy.json").write_text(json.dumps(policy_config(), indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record_all(args.output)


if __name__ == "__main__":
    main()
