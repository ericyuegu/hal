import { SELF } from "cloudflare:test";
import { expect, it } from "vitest";

it("answers unknown routes with a JSON 404", async () => {
  const response = await SELF.fetch("https://20xx.xyz/v1/nope");
  expect(response.status).toBe(404);
  expect(await response.json()).toEqual({ detail: "not found" });
});
