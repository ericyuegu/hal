declare namespace Cloudflare {
  // An interface cannot extend an import() type directly (TS2499), and
  // skipLibCheck hides that error in .d.ts files, so go through an alias.
  type AppEnv = import("../src/env").Env;
  interface Env extends AppEnv {}
}
