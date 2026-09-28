import { HttpError } from "./domain";
import type { PolicyConfig } from "./policy";

export type Fields = Record<string, unknown>;

// Shape errors mirror FastAPI's request validation: status 422.
export function fields(raw: unknown, allowed: readonly string[], required: readonly string[]): Fields {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    throw new HttpError(422, "request body must be a JSON object");
  }
  const value = raw as Fields;
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) throw new HttpError(422, `unexpected field ${key}`);
  }
  for (const key of required) {
    if (!(key in value)) throw new HttpError(422, `missing field ${key}`);
  }
  return value;
}

export function str(value: unknown, name: string): string {
  if (typeof value !== "string") throw new HttpError(422, `${name} must be a string`);
  return value;
}

export function num(value: unknown, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new HttpError(422, `${name} must be a number`);
  return value;
}

export function int(value: unknown, name: string): number {
  if (!Number.isInteger(value)) throw new HttpError(422, `${name} must be an integer`);
  return value as number;
}

export function bool(value: unknown, name: string): boolean {
  if (typeof value !== "boolean") throw new HttpError(422, `${name} must be a boolean`);
  return value;
}

export function nullableNum(value: unknown, name: string): number | null {
  return value === null ? null : num(value, name);
}

function inRange(value: number, [low, high]: [number, number], name: string): number {
  if (value < low || value > high) throw new HttpError(422, `${name} must be in [${low}, ${high}]`);
  return value;
}

export interface CreateRequest {
  player_code: string;
  character: string;
  imitation: string;
  online_delay: number;
  desired_return: number | null;
  temperature: number;
}

export function parseCreate(raw: unknown, policy: PolicyConfig): CreateRequest {
  const value = fields(
    raw,
    ["player_code", "character", "imitation", "online_delay", "desired_return", "temperature"],
    ["player_code", "character", "imitation", "online_delay"],
  );
  const code = str(value.player_code, "player_code");
  if (code.length < 3 || code.length > 13) throw new HttpError(422, "player_code must have 3 to 13 characters");
  const desired = "desired_return" in value ? nullableNum(value.desired_return, "desired_return") : 20;
  return {
    player_code: code,
    character: str(value.character, "character"),
    imitation: str(value.imitation, "imitation"),
    online_delay: int(value.online_delay, "online_delay"),
    desired_return: desired === null ? null : inRange(desired, policy.desired_return_range, "desired_return"),
    temperature:
      "temperature" in value
        ? inRange(num(value.temperature, "temperature"), policy.temperature_range, "temperature")
        : 1,
  };
}

export interface PolicyUpdate {
  desired_return?: number | null;
  temperature?: number | null;
}

export function parsePolicyUpdate(raw: unknown, policy: PolicyConfig): PolicyUpdate {
  const value = fields(raw, ["desired_return", "temperature"], []);
  if (Object.keys(value).length === 0) throw new HttpError(422, "provide desired_return or temperature");
  const update: PolicyUpdate = {};
  if ("desired_return" in value) {
    const desired = nullableNum(value.desired_return, "desired_return");
    update.desired_return = desired === null ? null : inRange(desired, policy.desired_return_range, "desired_return");
  }
  if ("temperature" in value) {
    const temperature = nullableNum(value.temperature, "temperature");
    update.temperature =
      temperature === null ? null : inRange(temperature, policy.temperature_range, "temperature");
  }
  return update;
}

export interface RematchRequest {
  character: string;
  imitation: string;
  stage: string;
}

export function parseRematch(raw: unknown, _policy: PolicyConfig): RematchRequest {
  const value = fields(raw, ["character", "imitation", "stage"], ["character", "imitation", "stage"]);
  return {
    character: str(value.character, "character"),
    imitation: str(value.imitation, "imitation"),
    stage: str(value.stage, "stage"),
  };
}
