import type { Env } from "./env";

export async function handle(_request: Request, _env: Env): Promise<Response> {
  return Response.json({ detail: "not found" }, { status: 404 });
}
