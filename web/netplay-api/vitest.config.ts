import { cloudflareTest } from "@cloudflare/vitest-pool-workers";
import { defineConfig } from "vitest/config";

// Digests of "runner-test-token", "runner-other-token", and "admin-test-token".
const RUNNER_DIGESTS =
  "ba2f9b108067689cfe677aa6917d364a6dc00f98e1c0701dfdd62e7d7de257d6," +
  "165860d05bc9beff640e7acb9c414d85e8df436292b9c8e821e872288e48643e";
const ADMIN_DIGEST = "1d4f144f52846450e02414b4f60277722e181fe96d30a2392aef2a7838a6aeae";

export default defineConfig({
  plugins: [
    cloudflareTest({
      wrangler: { configPath: "./wrangler.jsonc" },
      miniflare: {
        bindings: {
          HAL_TEST_CLOCK: "1",
          EDGE_CACHE: "off",
          RUNNER_TOKEN_SHA256: RUNNER_DIGESTS,
          ADMIN_TOKEN_SHA256: ADMIN_DIGEST,
          TWITCH_STREAM_KEY: "live_test_stream_key",
        },
      },
    }),
  ],
});
