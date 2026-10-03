import { env } from "cloudflare:test";
import { expect, it, vi } from "vitest";
import { handle } from "../src/http";

it.each(["/v1/capacity", "/v1/live", "/v1/runner/live", "/v1/admin/status"])(
  "maintenance refuses %s before accessing a Durable Object",
  async (path) => {
    const get = vi.spyOn(env.QUEUE, "get");
    try {
      const response = await handle(new Request(`https://20xx.xyz${path}`), { ...env, MAINTENANCE: "on" });
      expect(response.status).toBe(503);
      expect(response.headers.get("Retry-After")).toBe("300");
      expect(get).not.toHaveBeenCalled();
    } finally {
      get.mockRestore();
    }
  },
);
