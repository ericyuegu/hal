import { describe, expect, it } from "vitest";
import { HttpError, pyRepr, sameDigest, sha256Hex, validatePlayerCode, workerId } from "../src/domain";
import { checkChoice, optionsBody, parsePolicyConfig } from "../src/policy";
import { parseCreate, parseEnd, parseReport, parseSettingsUpdate } from "../src/requests";
import policyJson from "./transcripts/policy.json";

const policy = parsePolicyConfig(policyJson);

function error(fn: () => unknown): HttpError {
  try {
    fn();
  } catch (caught) {
    if (caught instanceof HttpError) return caught;
    throw caught;
  }
  throw new Error("expected an HttpError");
}

describe("domain", () => {
  it("validates connect codes exactly like the Python service", () => {
    expect(validatePlayerCode("CRYO#610")).toBe("CRYO#610");
    expect(error(() => validatePlayerCode("cryo#610"))).toMatchObject({
      status: 422,
      detail: "player_code must be an exact uppercase Slippi connect code such as CRYO#610",
    });
    expect(error(() => validatePlayerCode("ABCDEFGHI#1")).status).toBe(422);
  });

  it("quotes values like Python repr", () => {
    expect(pyRepr("FOX")).toBe("'FOX'");
    expect(pyRepr("it's")).toBe(`"it's"`);
  });

  it("derives worker IDs from session and slot", () => {
    expect(workerId("s1", 0)).toBe("s1/slot-0");
  });

  it("hashes and compares digests", async () => {
    expect(await sha256Hex("runner-test-token")).toBe(
      "ba2f9b108067689cfe677aa6917d364a6dc00f98e1c0701dfdd62e7d7de257d6",
    );
    expect(sameDigest("ab", "ab")).toBe(true);
    expect(sameDigest("ab", "ac")).toBe(false);
    expect(sameDigest("ab", "abc")).toBe(false);
  });
});

describe("policy", () => {
  it("rejects unknown or missing fields", () => {
    expect(error(() => parsePolicyConfig({ ...policyJson, extra: 1 })).status).toBe(422);
    const { bundle_sha256: _, ...missing } = policyJson;
    expect(error(() => parsePolicyConfig(missing)).status).toBe(422);
    expect(error(() => parsePolicyConfig({ ...policyJson, bundle_sha256: "XYZ" })).status).toBe(422);
    expect(error(() => parsePolicyConfig({ ...policyJson, desired_return_range: [40, 0] })).status).toBe(422);
  });

  it("names unsupported choices like the Python service", () => {
    expect(error(() => checkChoice(policy.characters, "WALUIGI", "character")).detail).toBe(
      "unsupported character 'WALUIGI'",
    );
  });

  it("hides MASKED unless the policy supports it", () => {
    expect(optionsBody(policy).imitations.some((choice) => choice.value === "MASKED")).toBe(false);
    const masked = { ...policy, masked_identity: true };
    expect(optionsBody(masked).imitations.some((choice) => choice.value === "MASKED")).toBe(true);
    expect(optionsBody(policy)).not.toHaveProperty("max_games");
  });
});

describe("requests", () => {
  const base = { player_code: "CRYO#610", character: "FOX", imitation: "IBDW#0", online_delay: 2 };

  it("applies create defaults", () => {
    expect(parseCreate(base, policy)).toEqual({ ...base, stage: null, desired_return: 20, temperature: 1 });
    expect(parseCreate({ ...base, desired_return: null }, policy).desired_return).toBeNull();
  });

  it("takes create defaults from the policy", () => {
    const custom = { ...policy, default_desired_return: 25, default_temperature: 0.9 };
    expect(parseCreate(base, custom)).toMatchObject({ desired_return: 25, temperature: 0.9 });
  });

  it("rejects malformed create bodies with 422", () => {
    for (const body of [
      null,
      [],
      { ...base, extra: 1 },
      { player_code: "CRYO#610" },
      { ...base, player_code: "CR" },
      { ...base, online_delay: 2.5 },
      { ...base, desired_return: 41 },
      { ...base, temperature: 0.5 },
      { ...base, character: 7 },
    ]) {
      expect(error(() => parseCreate(body, policy)).status).toBe(422);
    }
  });

  it("requires at least one settings field", () => {
    expect(error(() => parseSettingsUpdate({}, policy)).detail).toBe("provide at least one setting");
    expect(parseSettingsUpdate({ temperature: 1 }, policy)).toEqual({ temperature: 1 });
  });

  it("parses a full report", () => {
    const report = parseReport({
      seq: 4,
      phase: "character_select",
      phase_seconds_left: 3.5,
      bot_code: "HAL#9000",
      seen_revision: 2,
      locked_revision: 1,
      finished_games: [{ number: 1, stage: "BATTLEFIELD", result: "no_contest" }],
    });
    expect(report.finished_games[0]).toEqual({ number: 1, stage: "BATTLEFIELD", result: "no_contest" });
  });

  it("refuses an unknown phase, result, or a locked revision ahead of the seen one", () => {
    const baseReport = {
      seq: 1,
      phase: "in_game",
      phase_seconds_left: null,
      bot_code: null,
      seen_revision: 1,
      locked_revision: 1,
      finished_games: [],
    };
    expect(() => parseReport({ ...baseReport, phase: "rematch_wait" })).toThrow("phase");
    expect(() =>
      parseReport({
        ...baseReport,
        finished_games: [{ number: 1, stage: "BATTLEFIELD", result: "tie" }],
      }),
    ).toThrow("result");
    expect(() => parseReport({ ...baseReport, locked_revision: 2 })).toThrow("locked_revision");
  });

  it("accepts only runner end reasons", () => {
    expect(parseEnd({ reason: "no_show", retryable: false })).toEqual({ reason: "no_show", retryable: false });
    expect(() => parseEnd({ reason: "player_left", retryable: false })).toThrow("reason");
  });

  it("parses partial settings and allows null stage and desired_return", () => {
    expect(parseSettingsUpdate({ stage: null, desired_return: null }, policy)).toEqual({
      stage: null,
      desired_return: null,
    });
    expect(() => parseSettingsUpdate({}, policy)).toThrow("provide");
    expect(() => parseSettingsUpdate({ revision: 3 }, policy)).toThrow("unexpected field");
  });
});
