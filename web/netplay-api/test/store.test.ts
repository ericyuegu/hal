import { describe, expect, it } from "vitest";
import { HttpError } from "../src/domain";
import { CHOICES, withStore } from "./helpers";

function status(fn: () => unknown): number {
  try {
    fn();
  } catch (error) {
    if (error instanceof HttpError) return error.status;
    throw error;
  }
  return 200;
}

describe("JobStore player operations", () => {
  it("creates queued jobs with FIFO positions", () =>
    withStore((store) => {
      const first = store.createJob("j1", "d1", "AAAA#1", CHOICES);
      const second = store.createJob("j2", "d2", "BBBB#2", CHOICES);
      expect(first).toMatchObject({ id: "j1", status: "queued", queue_position: 1, attempt: 0, cancel_after_game: false });
      expect(second.queue_position).toBe(2);
      expect(store.queueDepth()).toBe(2);
      expect(store.activeCount()).toBe(0);
    }));

  it("allows one active job per player", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      let caught: unknown;
      try {
        store.createJob("j2", "d2", "CRYO#610", CHOICES);
      } catch (error) {
        caught = error;
      }
      expect(caught).toMatchObject({ status: 409, detail: "player CRYO#610 already has an active reservation" });
      store.cancel("j1", "d1");
      expect(store.createJob("j3", "d3", "CRYO#610", CHOICES).status).toBe("queued");
    }));

  it("hides jobs behind their token digest", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      expect(status(() => store.getJob("j1", "wrong"))).toBe(404);
      expect(status(() => store.getJob("missing", "d1"))).toBe(404);
      expect(store.getJob("j1", "d1").id).toBe("j1");
    }));

  it("advances the policy revision and refuses finished jobs", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      const updated = store.updatePolicy("j1", "d1", null, 0.9);
      expect(updated).toMatchObject({ desired_return: null, temperature: 0.9, policy_revision: 1 });
      store.cancel("j1", "d1");
      expect(status(() => store.updatePolicy("j1", "d1", 10, 1))).toBe(409);
    }));

  it("cancels queued jobs and returns terminal jobs unchanged", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      expect(store.cancel("j1", "d1").status).toBe("canceled");
      expect(store.cancel("j1", "d1").status).toBe("canceled");
    }));

  it("refuses a rematch unless the job waits for one", () =>
    withStore((store) => {
      store.createJob("j1", "d1", "CRYO#610", CHOICES);
      let caught: unknown;
      try {
        store.requestRematch("j1", "d1", "FALCO", "ZAIN#0", "BATTLEFIELD");
      } catch (error) {
        caught = error;
      }
      expect(caught).toMatchObject({ status: 409, detail: "job is not waiting for a rematch" });
    }));
});
