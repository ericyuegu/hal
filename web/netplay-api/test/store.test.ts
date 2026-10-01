import { describe, expect, it } from "vitest";
import { HttpError } from "../src/domain";
import type { ObservedReport } from "../src/requests";
import { NEW_JOB, withStore } from "./helpers";

function refused(fn: () => unknown): { status: number; detail: string } {
  try {
    fn();
  } catch (error) {
    if (error instanceof HttpError) return { status: error.status, detail: error.detail };
    throw error;
  }
  throw new Error("expected an HttpError");
}

function snapshot(seq: number, extra: Partial<ObservedReport> = {}): ObservedReport {
  return {
    seq,
    phase: "character_select",
    phase_seconds_left: 5,
    bot_code: "HAL#9000",
    seen_revision: 1,
    locked_revision: null,
    finished_games: [],
    ...extra,
  };
}

describe("job lifecycle", () => {
  it("creates settings at revision 1 and bumps it on each change", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      expect(store.getJob("j1", "d1").settings).toMatchObject({ revision: 1, character: "FOX", stage: null });
      const next = store.updateSettings("j1", "d1", { character: "FALCO", stage: "POKEMON_STADIUM" });
      expect(next.settings).toMatchObject({ revision: 2, character: "FALCO", stage: "POKEMON_STADIUM" });
    }));

  it("claims FIFO with a fresh snapshot and a 20 s lease", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.createJob("j2", "d2", { ...NEW_JOB, player_code: "AAAA#1" });
      expect(store.claimNext("w0")).toMatchObject({ id: "j1", status: "assigned", attempt: 1, observed: null });
      expect(store.row("j1")).toMatchObject({
        lease_owner: "w0",
        lease_expires_at: clock.now + 20,
        assigned_at: clock.now,
      });
    }));

  it("keeps the newest snapshot by seq and lengthens the lease in game", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      store.report("j1", "w0", snapshot(2, { phase: "in_game", phase_seconds_left: null }));
      expect(store.row("j1")?.lease_expires_at).toBe(clock.now + 60);
      const stale = store.report("j1", "w0", snapshot(1));
      expect(stale.view.observed?.phase).toBe("in_game");
      expect(stale.phaseChanged).toBe(false);
    }));

  it("converts the phase's seconds left to an absolute deadline", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      expect(store.report("j1", "w0", snapshot(1, { phase_seconds_left: 4.5 })).view.phase_deadline).toBe(
        clock.now + 4.5,
      );
    }));

  it("records each game once and refuses a changed result", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      const game = { number: 1, stage: "BATTLEFIELD", result: "win" as const };
      expect(store.report("j1", "w0", snapshot(1, { finished_games: [game] })).newGames).toEqual([game]);
      expect(store.report("j1", "w0", snapshot(2, { finished_games: [game] })).newGames).toEqual([]);
      expect(
        refused(() =>
          store.report("j1", "w0", snapshot(3, { finished_games: [{ ...game, result: "loss" }] })),
        ).status,
      ).toBe(409);
      expect(store.getJob("j1", "d1").games).toEqual([game]);
    }));

  it("refuses a seen revision that does not exist yet", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      expect(refused(() => store.report("j1", "w0", snapshot(1, { seen_revision: 2 }))).status).toBe(422);
    }));

  it("refuses reports and ends from another worker or after the end", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      expect(refused(() => store.report("j1", "w1", snapshot(1))).status).toBe(409);
      store.end("j1", "w0", "no_show", false);
      expect(refused(() => store.report("j1", "w0", snapshot(2))).status).toBe(409);
      expect(refused(() => store.end("j1", "w1", "no_show", false)).status).toBe(409);
    }));

  it("returns the current job when the same worker repeats an end", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      store.end("j1", "w0", "player_disconnected", false);
      expect(store.end("j1", "w0", "player_disconnected", false)).toMatchObject({
        status: "ended",
        end_reason: "player_disconnected",
      });
      store.createJob("j2", "d2", { ...NEW_JOB, player_code: "AAAA#1" });
      store.claimNext("w0");
      expect(store.end("j2", "w0", "service_failure", true).status).toBe("queued");
      expect(store.end("j2", "w0", "service_failure", true).status).toBe("queued");
    }));

  it("requeues a retryable failure at the front once, then ends it", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.createJob("j2", "d2", { ...NEW_JOB, player_code: "AAAA#1" });
      store.claimNext("w0");
      expect(store.end("j1", "w0", "service_failure", true)).toMatchObject({ status: "queued", queue_position: 1 });
      expect(store.claimNext("w0")?.id).toBe("j1");
      expect(store.end("j1", "w0", "service_failure", true)).toMatchObject({
        status: "ended",
        end_reason: "service_failure",
      });
    }));

  it("does not requeue a failed reservation the player asked to stop", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      store.leave("j1", "d1");
      expect(store.end("j1", "w0", "service_failure", true)).toMatchObject({
        status: "ended",
        end_reason: "player_canceled",
      });
    }));

  it("ends a queued job on leave and winds down an assigned one", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      expect(store.leave("j1", "d1")).toMatchObject({ status: "ended", end_reason: "player_canceled" });
      store.createJob("j2", "d2", NEW_JOB);
      store.claimNext("w0");
      expect(store.leave("j2", "d2")).toMatchObject({ status: "assigned", wind_down: "player" });
    }));

  it("counts lock requests while assigned", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      expect(refused(() => store.requestLock("j1", "d1")).status).toBe(409);
      store.claimNext("w0");
      expect(store.requestLock("j1", "d1").lock_requests).toBe(1);
    }));

  it("requeues an expired lease once, then ends it", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      clock.advance(21);
      store.getJob("j1", "d1");
      expect(store.reapExpired()).toEqual(["j1"]);
      expect(store.row("j1")).toMatchObject({ status: "queued", retry_front: 1, lease_owner: null, phase: null });
      store.claimNext("w0");
      clock.advance(21);
      store.getJob("j1", "d1");
      store.reapExpired();
      expect(store.row("j1")).toMatchObject({ status: "ended", end_reason: "service_failure" });
    }));

  it("releases a queued job whose page stopped polling, but never an assigned one", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.createJob("j2", "d2", { ...NEW_JOB, player_code: "AAAA#1" });
      store.claimNext("w0");
      for (let t = 0; t < 120; t += 10) {
        clock.advance(10);
        store.report("j1", "w0", snapshot(t + 1));
      }
      expect(store.reapExpired()).toEqual(["j2"]);
      expect(store.row("j2")).toMatchObject({ status: "ended", end_reason: "player_left" });
      expect(store.row("j1")?.status).toBe("assigned");
    }));

  it("records a poll at most every 10 s", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      const created = clock.now;
      clock.advance(9);
      store.getJob("j1", "d1");
      expect(store.row("j1")?.player_seen_at).toBe(created);
      clock.advance(1);
      store.getJob("j1", "d1");
      expect(store.row("j1")?.player_seen_at).toBe(clock.now);
    }));

  it("winds down a long reservation only when someone is waiting", () =>
    withStore((store, clock) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("w0");
      clock.advance(900);
      expect(store.applyYield()).toEqual([]);
      store.createJob("j2", "d2", { ...NEW_JOB, player_code: "AAAA#1" });
      expect(store.applyYield()).toEqual(["j1"]);
      expect(store.row("j1")?.wind_down).toBe("yield");
      expect(store.applyYield()).toEqual([]);
    }));

  it("fails held jobs of an ended session by the same requeue rule", () =>
    withStore((store) => {
      store.createJob("j1", "d1", NEW_JOB);
      store.claimNext("s/slot-0");
      expect(store.failWorkers(["s/slot-0"])).toEqual(["j1"]);
      expect(store.row("j1")?.status).toBe("queued");
    }));

  it("reports the earliest deadline", () =>
    withStore((store, clock) => {
      expect(store.nextDeadline()).toBeNull();
      store.createJob("j1", "d1", NEW_JOB);
      expect(store.nextDeadline()).toBe(clock.now + 120);
      store.claimNext("w0");
      expect(store.nextDeadline()).toBe(clock.now + 20);
    }));
});
