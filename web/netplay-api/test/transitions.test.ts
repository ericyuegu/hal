import { describe, expect, it } from "vitest";
import { HttpError } from "../src/domain";
import type { JobStore } from "../src/store";
import { CHOICES, withStore } from "./helpers";

function refused(fn: () => unknown): { status: number; detail: string } {
  try {
    fn();
  } catch (error) {
    if (error instanceof HttpError) return { status: error.status, detail: error.detail };
    throw error;
  }
  throw new Error("expected an HttpError");
}

function playing(store: JobStore, worker = "w0"): void {
  store.createJob("j1", "d1", "CRYO#610", CHOICES);
  store.claimNext(worker);
  store.markConnecting("j1", worker, "HALBOT#1");
  store.markPlaying("j1", worker);
}

describe("runner transitions", () => {
  it("claims in FIFO order with a 20 s lease", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", "AAAA#1", CHOICES);
      store.createJob("j2", "d2", "BBBB#2", CHOICES);
      expect(store.claimNext("w0")).toMatchObject({ id: "j1", status: "leased", attempt: 1 });
      expect(store.row("j1")).toMatchObject({ lease_owner: "w0", lease_expires_at: clock.now + 20 });
      expect(store.claimNext("w1")?.id).toBe("j2");
      expect(store.claimNext("w2")).toBeNull();
    }));

  it("gives a playing lease 60 s and other states 20 s", () =>
    withStore((store, clock) => {
      playing(store);
      expect(store.row("j1")?.lease_expires_at).toBe(clock.now + 60);
      clock.advance(10);
      store.heartbeat("j1", "w0");
      expect(store.row("j1")?.lease_expires_at).toBe(clock.now + 60);
      store.finishGame("j1", "w0", 1, "BATTLEFIELD", "win");
      store.heartbeat("j1", "w0");
      expect(store.row("j1")?.lease_expires_at).toBe(clock.now + 20);
    }));

  it("does not expire a playing lease after 30 s of silence", () =>
    withStore((store, clock) => {
      playing(store);
      clock.advance(30);
      expect(store.reapExpired()).toEqual([]);
      expect(store.row("j1")?.status).toBe("playing");
      clock.advance(30);
      expect(store.reapExpired()).toEqual(["j1"]);
    }));

  it("accepts a retried transition that was already applied", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      store.claimNext("w0");
      store.markConnecting("j1", "w0", "HALBOT#1");
      expect(store.markConnecting("j1", "w0", "HALBOT#1").status).toBe("connecting");
      store.markPlaying("j1", "w0");
      expect(store.markPlaying("j1", "w0").status).toBe("playing");
      store.finishGame("j1", "w0", 1, "BATTLEFIELD", "win");
      expect(store.finishGame("j1", "w0", 1, "BATTLEFIELD", "win")).toMatchObject({ status: "rematch_wait", game_count: 1 });
      expect(refused(() => store.finishGame("j1", "w0", 1, "DREAMLAND", "win")).detail).toBe(
        "game already recorded with a different result",
      );
      expect(Number(store.row("j1")?.game_count)).toBe(1);
    }));

  it("accepts retried no-show, no-contest, fail, and forfeit", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "AAAA#1", CHOICES);
      store.claimNext("w0");
      store.markConnecting("j1", "w0", "HALBOT#1");
      store.markNoShow("j1", "w0");
      expect(store.markNoShow("j1", "w0").status).toBe("no_show");

      store.createJob("j2", "d2", "BBBB#2", CHOICES);
      store.claimNext("w0");
      store.markConnecting("j2", "w0", "HALBOT#1");
      store.markPlaying("j2", "w0");
      store.markNoContest("j2", "w0");
      expect(store.markNoContest("j2", "w0").status).toBe("canceled");

      store.createJob("j3", "d3", "CCCC#3", CHOICES);
      store.claimNext("w0");
      store.fail("j3", "w0", "dolphin_crash", true);
      expect(store.fail("j3", "w0", "dolphin_crash", true).status).toBe("queued");

      store.createJob("j4", "d4", "DDDD#4", CHOICES);
      store.claimNext("w1");
      store.claimNext("w1");
      store.markConnecting("j3", "w1", "HALBOT#1");
      store.forfeit("j3", "w1");
      expect(store.forfeit("j3", "w1").error_code).toBe("service_failure_bot_forfeit");
    }));

  it("refuses transitions from other workers and wrong states", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      expect(refused(() => store.markConnecting("j1", "w0", "HALBOT#1"))).toEqual({
        status: 409,
        detail: "worker does not own this job",
      });
      store.claimNext("w0");
      expect(refused(() => store.markPlaying("j1", "w0"))).toEqual({
        status: 409,
        detail: "job status 'leased' is not valid for this operation",
      });
      expect(refused(() => store.heartbeat("j1", "w1"))).toEqual({
        status: 409,
        detail: "worker does not own an active job lease",
      });
      expect(refused(() => store.finishGame("j1", "w0", 1, "BATTLEFIELD", "")).status).toBe(422);
      expect(refused(() => store.fail("j1", "w0", "", true)).status).toBe(422);
    }));

  it("requires game numbers in order", () =>
    withStore((store) => {
      playing(store);
      expect(refused(() => store.finishGame("j1", "w0", 2, "BATTLEFIELD", "win")).detail).toBe(
        "game_number 2 does not follow game 0",
      );
    }));

  it("completes after five games or after a cancel during play", () =>
    withStore((store) => {
      playing(store);
      for (let number = 1; number <= 5; number += 1) {
        const job = store.finishGame("j1", "w0", number, "BATTLEFIELD", "loss");
        if (number < 5) {
          store.requestRematch("j1", "d1", "FOX", "IBDW#0", "BATTLEFIELD");
          store.markPlaying("j1", "w0");
        } else {
          expect(job.status).toBe("complete");
        }
      }
    }));

  it("refuses another worker's retry of an applied transition", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      store.claimNext("w0");
      store.markConnecting("j1", "w0", "HALBOT#1");
      expect(refused(() => store.markConnecting("j1", "w1", "HALBOT#1"))).toEqual({
        status: 409,
        detail: "worker does not own this job",
      });
    }));

  it("refuses another worker's retry of a recorded game", () =>
    withStore((store) => {
      playing(store);
      store.finishGame("j1", "w0", 1, "BATTLEFIELD", "win");
      expect(refused(() => store.finishGame("j1", "w1", 1, "BATTLEFIELD", "win")).status).toBe(409);
      expect(Number(store.row("j1")?.game_count)).toBe(1);
    }));

  it("refuses a stale fail retry after another worker claims the job", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "AAAA#1", CHOICES);
      store.claimNext("w0");
      store.fail("j1", "w0", "dolphin_crash", true);
      store.claimNext("w1");
      expect(refused(() => store.fail("j1", "w0", "dolphin_crash", true))).toEqual({
        status: 409,
        detail: "worker does not own this job",
      });
      expect(store.row("j1")).toMatchObject({ status: "leased", lease_owner: "w1" });
    }));

  it("refuses a fail retry from a worker that no longer applied the last change", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "AAAA#1", CHOICES);
      store.claimNext("w0");
      expect(store.fail("j1", "w0", "dolphin_crash", true).status).toBe("queued");
      store.claimNext("w1");
      expect(store.fail("j1", "w1", "dolphin_crash", true)).toMatchObject({ status: "failed", attempt: 2 });
      expect(store.row("j1")).toMatchObject({ lease_owner: null, last_worker: "w1" });
      expect(refused(() => store.fail("j1", "w0", "dolphin_crash", true))).toEqual({
        status: 409,
        detail: "worker does not own this job",
      });
      expect(store.row("j1")).toMatchObject({ status: "failed", last_worker: "w1" });
    }));

  it("completes after a cancel during play", () =>
    withStore((store) => {
      playing(store);
      store.cancel("j1", "d1");
      expect(store.finishGame("j1", "w0", 1, "BATTLEFIELD", "win").status).toBe("complete");
    }));

  it("retries a failed job once at the front of the queue", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "AAAA#1", CHOICES);
      store.createJob("j2", "d2", "BBBB#2", CHOICES);
      store.claimNext("w0");
      expect(store.fail("j1", "w0", "dolphin_crash", true)).toMatchObject({ status: "queued", queue_position: 1 });
      store.claimNext("w0");
      expect(store.fail("j1", "w0", "dolphin_crash", true).status).toBe("failed");
    }));

  it("fails only the named workers' leases", () =>
    withStore((store) => {
      playing(store, "a/slot-0");
      store.createJob("j2", "d2", "BBBB#2", CHOICES);
      store.claimNext("b/slot-0");
      expect(store.failWorkers(["a/slot-0", "a/slot-1"])).toEqual(["j1"]);
      expect(store.row("j1")).toMatchObject({ status: "failed", error_code: "service_failure_bot_forfeit", last_result: "win" });
      expect(store.row("j2")?.status).toBe("leased");
    }));

  it("records a replay once per game", () =>
    withStore((store) => {
      playing(store);
      expect(refused(() => store.recordReplay("j1", "w0", 1, "k", "a".repeat(64), 5, "e")).detail).toBe("game is absent");
      store.finishGame("j1", "w0", 1, "BATTLEFIELD", "win");
      store.recordReplay("j1", "w0", 1, "k", "a".repeat(64), 5, "e");
      store.recordReplay("j1", "w0", 1, "k", "a".repeat(64), 5, "e");
      expect(refused(() => store.recordReplay("j1", "w1", 1, "k", "a".repeat(64), 5, "e")).detail).toBe(
        "worker did not play this game",
      );
      expect(refused(() => store.recordReplay("j1", "w0", 1, "k", "a".repeat(64), 5, "other")).detail).toBe(
        "game already has a different replay",
      );
    }));

  it("reports the earliest deadline", () =>
    withStore((store, clock) => {
      expect(store.nextDeadline()).toBeNull();
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      store.claimNext("w0");
      expect(store.nextDeadline()).toBe(clock.now + 20);
      store.markConnecting("j1", "w0", "HALBOT#1");
      store.heartbeat("j1", "w0");
      expect(store.nextDeadline()).toBe(clock.now + 20);
    }));
});
