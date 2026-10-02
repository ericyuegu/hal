import type { Queue } from "./queue";

export interface Env {
  MAINTENANCE?: "on" | "off";
  QUEUE: DurableObjectNamespace<Queue>;
  JOB_RATE_LIMIT: RateLimit;
  // "on" in deployment, "off" in tests whose assertions read fresh capacity.
  EDGE_CACHE: "on" | "off";
  RUNNER_TOKEN_SHA256: string;
  ADMIN_TOKEN_SHA256: string;
  TWITCH_STREAM_KEY: string;
  // Set only by vitest.config.ts; enables the controllable clock and reset.
  HAL_TEST_CLOCK?: string;
}
