import { beforeEach, describe, expect, it } from "vitest";
import { POLICY, call, publish, resetQueue } from "./helpers";

beforeEach(async () => {
  await resetQueue();
});

describe("runner policy", () => {
  it("serves the active policy config to runners only", async () => {
    expect(await call("GET", "/v1/runner/policy", { runner: true })).toMatchObject({
      status: 503,
      body: { detail: "no policy has been published" },
    });
    await publish();
    const result = await call("GET", "/v1/runner/policy", { runner: true });
    expect(result.status).toBe(200);
    expect(result.body).toEqual(POLICY);
    expect((await call("GET", "/v1/runner/policy")).status).toBe(401);
  });
});
