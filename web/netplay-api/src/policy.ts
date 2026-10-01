import { HttpError, pyRepr } from "./domain";

export interface Choice {
  value: string;
  label: string;
}

export interface PolicyConfig {
  schema_version: 1;
  bundle_sha256: string;
  bundle_r2_key: string;
  vocabulary_sha256: string;
  characters: Choice[];
  imitations: Choice[];
  stages: Choice[];
  online_delays: number[];
  desired_return_range: [number, number];
  default_desired_return: number;
  temperature_range: [number, number];
  default_temperature: number;
  masked_identity: boolean;
}

const FIELDS = [
  "schema_version",
  "bundle_sha256",
  "bundle_r2_key",
  "vocabulary_sha256",
  "characters",
  "imitations",
  "stages",
  "online_delays",
  "desired_return_range",
  "default_desired_return",
  "temperature_range",
  "default_temperature",
  "masked_identity",
] as const;
const SHA256 = /^[0-9a-f]{64}$/;

function invalid(detail: string): never {
  throw new HttpError(422, `policy config: ${detail}`);
}

function choices(value: unknown, name: string): Choice[] {
  if (!Array.isArray(value) || value.length === 0) invalid(`${name} must be a non-empty list`);
  const seen = new Set<string>();
  for (const item of value) {
    if (typeof item !== "object" || item === null || Array.isArray(item)) invalid(`${name} entries must be objects`);
    const entry = item as Record<string, unknown>;
    if (Object.keys(entry).sort().join() !== "label,value") invalid(`${name} entries need exactly value and label`);
    if (typeof entry.value !== "string" || !entry.value || typeof entry.label !== "string" || !entry.label) {
      invalid(`${name} values and labels must be non-empty strings`);
    }
    if (seen.has(entry.value)) invalid(`${name} repeats ${entry.value}`);
    seen.add(entry.value);
  }
  return value as Choice[];
}

function range(value: unknown, name: string): [number, number] {
  if (
    !Array.isArray(value) ||
    value.length !== 2 ||
    !value.every((item) => typeof item === "number" && Number.isFinite(item)) ||
    value[0] >= value[1]
  ) {
    invalid(`${name} must be two increasing finite numbers`);
  }
  return value as [number, number];
}

function within(value: unknown, [low, high]: [number, number], name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < low || value > high) {
    invalid(`${name} must lie in its range`);
  }
  return value;
}

export function parsePolicyConfig(raw: unknown): PolicyConfig {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) invalid("must be an object");
  const value = raw as Record<string, unknown>;
  const keys = Object.keys(value).sort();
  if (keys.join() !== [...FIELDS].sort().join()) invalid(`fields must be exactly ${FIELDS.join(", ")}`);
  if (value.schema_version !== 1) invalid("schema_version must be 1");
  for (const name of ["bundle_sha256", "vocabulary_sha256"] as const) {
    if (typeof value[name] !== "string" || !SHA256.test(value[name] as string)) invalid(`${name} must be SHA-256 hex`);
  }
  if (typeof value.bundle_r2_key !== "string" || !value.bundle_r2_key) invalid("bundle_r2_key must be non-empty");
  choices(value.characters, "characters");
  choices(value.imitations, "imitations");
  choices(value.stages, "stages");
  const delays = value.online_delays;
  if (!Array.isArray(delays) || delays.length === 0 || !delays.every((delay) => delay === 2 || delay === 3)) {
    invalid("online_delays must be a non-empty subset of [2, 3]");
  }
  within(value.default_desired_return, range(value.desired_return_range, "desired_return_range"), "default_desired_return");
  within(value.default_temperature, range(value.temperature_range, "temperature_range"), "default_temperature");
  if (typeof value.masked_identity !== "boolean") invalid("masked_identity must be a boolean");
  return value as unknown as PolicyConfig;
}

export function checkChoice(list: Choice[], value: string, name: string): string {
  if (!list.some((choice) => choice.value === value)) throw new HttpError(422, `unsupported ${name} ${pyRepr(value)}`);
  return value;
}

export function optionsBody(policy: PolicyConfig) {
  return {
    characters: policy.characters,
    imitations: policy.masked_identity
      ? policy.imitations
      : policy.imitations.filter((choice) => choice.value !== "MASKED"),
    stages: policy.stages,
    online_delays: policy.online_delays,
    desired_return_range: policy.desired_return_range,
    default_desired_return: policy.default_desired_return,
    temperature_range: policy.temperature_range,
    default_temperature: policy.default_temperature,
  };
}
