import { SELF } from "cloudflare:test";
import { beforeEach, describe, expect, it } from "vitest";
import { CREATE, RUNNER_TOKEN, call, publish, resetQueue, seedAccounts, startSession } from "./helpers";

beforeEach(async () => {
  await resetQueue();
});

async function open(jobId: string, session: string, slot: number): Promise<{ socket: WebSocket; messages: unknown[] }> {
  const response = await SELF.fetch(`https://20xx.xyz/v1/runner/jobs/${jobId}/live`, {
    headers: {
      Upgrade: "websocket",
      Authorization: `Bearer ${RUNNER_TOKEN}`,
      "X-HAL-Session": session,
      "X-HAL-Slot": String(slot),
    },
  });
  expect(response.status).toBe(101);
  const socket = response.webSocket!;
  const messages: unknown[] = [];
  socket.addEventListener("message", (event) => {
    messages.push(JSON.parse(event.data as string));
  });
  socket.accept();
  return { socket, messages };
}

async function until(check: () => boolean): Promise<void> {
  for (let attempt = 0; attempt < 50 && !check(); attempt += 1) await new Promise((resolve) => setTimeout(resolve, 10));
  expect(check()).toBe(true);
}

async function claimed(): Promise<{ session: string; job: { id: string; token: string } }> {
  await publish();
  await seedAccounts(2);
  const session = await startSession(2);
  const job = (await call("POST", "/v1/jobs", { body: CREATE })).body;
  await call("POST", `/v1/runner/sessions/${session}/claim`, { runner: true, body: { slot: 0 } });
  return { session, job };
}

describe("live settings", () => {
  it("sends current settings, then every change", async () => {
    const { session, job } = await claimed();
    const { messages } = await open(job.id, session, 0);
    await until(() => messages.length === 1);
    expect(messages[0]).toEqual({ type: "settings", revision: 0, desired_return: 20, temperature: 1 });
    await call("PATCH", `/v1/jobs/${job.id}/policy`, { token: job.token, body: { desired_return: 35 } });
    await until(() => messages.length === 2);
    expect(messages[1]).toEqual({ type: "settings", revision: 1, desired_return: 35, temperature: 1 });
  });

  it("releases the runner when the player cancels", async () => {
    const { session, job } = await claimed();
    const { messages, socket } = await open(job.id, session, 0);
    let closed = 0;
    socket.addEventListener("close", (event) => {
      closed = event.code;
    });
    await until(() => messages.length === 1);
    await call("DELETE", `/v1/jobs/${job.id}`, { token: job.token });
    await until(() => messages.length === 2 && closed === 1000);
    expect(messages[1]).toEqual({ type: "released" });
  });

  it("keeps serving the job after the runner closes its socket", async () => {
    const { session, job } = await claimed();
    const { messages, socket } = await open(job.id, session, 0);
    await until(() => messages.length === 1);
    const closed = new Promise((resolve) => socket.addEventListener("close", resolve));
    socket.close(1000, "done");
    await closed;
    const update = await call("PATCH", `/v1/jobs/${job.id}/policy`, { token: job.token, body: { desired_return: 35 } });
    expect(update.status).toBe(200);
    const again = await open(job.id, session, 0);
    await until(() => again.messages.length === 1);
    expect(again.messages[0]).toEqual({ type: "settings", revision: 1, desired_return: 35, temperature: 1 });
  });

  it("keeps serving the job after releasing a socket the runner never acknowledged", async () => {
    const { session, job } = await claimed();
    const response = await SELF.fetch(`https://20xx.xyz/v1/runner/jobs/${job.id}/live`, {
      headers: {
        Upgrade: "websocket",
        Authorization: `Bearer ${RUNNER_TOKEN}`,
        "X-HAL-Session": session,
        "X-HAL-Slot": "0",
      },
    });
    expect(response.status).toBe(101);
    const fail = await call("POST", `/v1/runner/jobs/${job.id}/fail`, {
      runner: { session, slot: 0 },
      body: { error_code: "dolphin_crashed", retryable: true },
    });
    expect(fail.status).toBe(200);
    expect(fail.body.status).toBe("queued");
    const update = await call("PATCH", `/v1/jobs/${job.id}/policy`, { token: job.token, body: { desired_return: 35 } });
    expect(update.status).toBe(200);
    expect((await call("DELETE", `/v1/jobs/${job.id}`, { token: job.token })).status).toBe(200);
  });

  it("refuses a socket from a worker that does not own the job", async () => {
    const { session, job } = await claimed();
    const response = await SELF.fetch(`https://20xx.xyz/v1/runner/jobs/${job.id}/live`, {
      headers: {
        Upgrade: "websocket",
        Authorization: `Bearer ${RUNNER_TOKEN}`,
        "X-HAL-Session": session,
        "X-HAL-Slot": "1",
      },
    });
    expect(response.status).toBe(409);
  });

  it("answers a bad upgrade with a JSON client error", async () => {
    const { session, job } = await claimed();
    const url = `https://20xx.xyz/v1/runner/jobs/${job.id}/live`;
    const auth = { Authorization: `Bearer ${RUNNER_TOKEN}` };
    const cases: [Record<string, string>, number][] = [
      [{ ...auth, "X-HAL-Session": session, "X-HAL-Slot": "0" }, 404],
      [{ ...auth, Upgrade: "websocket", "X-HAL-Slot": "0" }, 400],
      [{ ...auth, Upgrade: "websocket", "X-HAL-Session": session, "X-HAL-Slot": "zero" }, 400],
      [{ ...auth, Upgrade: "websocket", "X-HAL-Session": "missing", "X-HAL-Slot": "0" }, 404],
      [{ ...auth, Upgrade: "websocket", "X-HAL-Session": session, "X-HAL-Slot": "5" }, 422],
    ];
    for (const [headers, status] of cases) {
      const response = await SELF.fetch(url, { headers });
      expect(response.status).toBe(status);
      expect(await response.json()).toHaveProperty("detail");
    }
  });
});
