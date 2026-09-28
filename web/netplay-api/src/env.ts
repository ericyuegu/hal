import type { Queue } from "./queue";

export interface Env {
  QUEUE: DurableObjectNamespace<Queue>;
  JOB_RATE_LIMIT: RateLimit;
  RUNNER_TOKEN_SHA256: string;
  ADMIN_TOKEN_SHA256: string;
  TWITCH_STREAM_KEY: string;
  // Set only by vitest.config.ts; enables the controllable clock and reset.
  HAL_TEST_CLOCK?: string;
}
