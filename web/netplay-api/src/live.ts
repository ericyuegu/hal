import { HttpError, MAX_BODY_BYTES, RUNNER_PROTOCOL_VERSION } from "./domain";

export type Peer =
  | { role: "browser"; job: string | null; seen: number; sent: number; acked: number; pending: [number, number][]; capacity?: string }
  | { role: "host"; session: string; seen: number; sent: number; acked: number; pending: [number, number][]; capacity?: string };

// Attachments and auto-response timestamps survive Durable Object hibernation.
export class LiveConnections {
  constructor(private readonly ctx: DurableObjectState, private readonly now: () => number) {
    ctx.setWebSocketAutoResponse(new WebSocketRequestResponsePair("ping", "pong"));
  }

  open(session: string | null): Response {
    if (session === null && this.ctx.getWebSockets("browser").length >= 100) {
      throw new HttpError(503, "Live connections are full. Try again later.");
    }
    if (session !== null) {
      for (const old of this.ctx.getWebSockets(`host:${session}`)) old.close(1012, "connection replaced");
    }
    const pair = new WebSocketPair();
    const peer: Peer = session === null
      ? { role: "browser", job: null, seen: this.now(), sent: 0, acked: 0, pending: [] }
      : { role: "host", session, seen: this.now(), sent: 0, acked: 0, pending: [] };
    this.ctx.acceptWebSocket(pair[1], [peer.role, ...(session === null ? [] : [`host:${session}`])]);
    pair[1].serializeAttachment(peer);
    this.send(pair[1], { type: "hello", protocol_version: RUNNER_PROTOCOL_VERSION });
    return new Response(null, { status: 101, webSocket: pair[0] });
  }

  peer(ws: WebSocket): Peer {
    return ws.deserializeAttachment() as Peer;
  }

  send(ws: WebSocket, message: unknown): void {
    if (ws.readyState !== WebSocket.OPEN) return;
    const peer = this.peer(ws);
    const value = message as { type?: string };
    if (value.type === "capacity") {
      const body = JSON.stringify(message);
      if (body === peer.capacity) return;
      peer.capacity = body;
    }
    const text = JSON.stringify({ sequence: peer.sent + 1, ...message as object });
    const bytes = new TextEncoder().encode(text).byteLength;
    const pending = peer.pending.reduce((sum, item) => sum + item[1], 0);
    const limit = peer.role === "host" ? 2 * 1024 * 1024 : 512 * 1024;
    if (peer.sent - peer.acked >= 64 || pending + bytes > limit) {
      ws.close(1013, "slow consumer; reconnect for a snapshot");
      return;
    }
    if (bytes > 256 * 1024) {
      ws.close(1009, "snapshot exceeds 256 KiB limit");
      return;
    }
    if (bytes <= MAX_BODY_BYTES) ws.send(text);
    else {
      const size = 2048;
      const count = Math.ceil(text.length / size);
      for (let page = 0; page < count; page++) {
        ws.send(JSON.stringify({ type: "page", sequence: peer.sent + 1, page, count, text: text.slice(page * size, (page + 1) * size) }));
      }
    }
    peer.pending.push([peer.sent + 1, bytes]);
    peer.sent += 1;
    ws.serializeAttachment(peer);
  }

  acknowledge(ws: WebSocket, sequence: number): void {
    const peer = this.peer(ws);
    if (!Number.isInteger(sequence) || sequence < peer.acked || sequence > peer.sent) {
      throw new HttpError(422, "invalid acknowledgement");
    }
    peer.acked = sequence;
    peer.pending = peer.pending.filter((item) => item[0] > sequence);
    ws.serializeAttachment(peer);
  }

  sockets(role?: "browser" | "host"): WebSocket[] {
    return this.ctx.getWebSockets(role);
  }

  lastSeen(ws: WebSocket): number {
    return Math.max(this.peer(ws).seen, (this.ctx.getWebSocketAutoResponseTimestamp(ws)?.getTime() ?? 0) / 1000);
  }

  presence(job: string, persisted: number): number {
    let seen = persisted;
    for (const ws of this.sockets("browser")) {
      const peer = this.peer(ws);
      if (peer.role !== "browser" || peer.job !== job) continue;
      const ping = this.ctx.getWebSocketAutoResponseTimestamp(ws);
      seen = Math.max(seen, peer.seen, ping === null ? 0 : ping.getTime() / 1000);
    }
    return seen;
  }
}
