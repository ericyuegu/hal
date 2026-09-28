import { DurableObject } from "cloudflare:workers";
import type { Env } from "./env";

export class Queue extends DurableObject<Env> {}
