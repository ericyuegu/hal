import { describe, expect, it } from "vitest";
import { servePublic } from "../src/http";

function request(path: string): Request {
  return new Request(`https://cache-test.example${path}?${crypto.randomUUID()}`);
}

describe("public read cache", () => {
  it("serves a cached 200 without producing it again", async () => {
    const req = request("/v1/capacity");
    let produced = 0;
    const produce = async () => {
      produced += 1;
      return Response.json({ n: produced }, { headers: { "Cache-Control": "no-store" } });
    };
    const first = await servePublic(caches.default, req, 3, produce);
    const second = await servePublic(caches.default, req, 3, produce);
    expect(await first.json()).toEqual({ n: 1 });
    expect(await second.json()).toEqual({ n: 1 });
    expect(produced).toBe(1);
    expect(second.headers.get("Cache-Control")).toBe("no-store");
  });

  it("never caches an error", async () => {
    const req = request("/v1/options");
    let produced = 0;
    const produce = async () => {
      produced += 1;
      return Response.json({ detail: "no policy" }, { status: 503 });
    };
    await servePublic(caches.default, req, 10, produce);
    await servePublic(caches.default, req, 10, produce);
    expect(produced).toBe(2);
  });
});
