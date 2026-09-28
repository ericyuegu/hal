import type { Env } from "./env";
import { handle } from "./http";

export { Queue } from "./queue";

export default {
  fetch(request, env) {
    return handle(request, env);
  },
} satisfies ExportedHandler<Env>;
