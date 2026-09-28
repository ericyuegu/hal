import { describe, expect, it } from "vitest";
import { HttpError, pyRepr, sameDigest, sha256Hex, validatePlayerCode, workerId } from "../src/domain";
import { checkChoice, optionsBody, parsePolicyConfig } from "../src/policy";
import { parseCreate, parsePolicyUpdate, parseRematch } from "../src/requests";
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
    expect(optionsBody(policy)).toMatchObject({ max_games: 5, no_show_seconds: 600, rematch_seconds: 600 });
  });
});

describe("requests", () => {
  const base = { player_code: "CRYO#610", character: "FOX", imitation: "IBDW#0", online_delay: 2 };

  it("applies create defaults", () => {
    expect(parseCreate(base, policy)).toEqual({ ...base, desired_return: 20, temperature: 1 });
    expect(parseCreate({ ...base, desired_return: null }, policy).desired_return).toBeNull();
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

  it("requires at least one policy field", () => {
    expect(error(() => parsePolicyUpdate({}, policy)).detail).toBe("provide desired_return or temperature");
    expect(parsePolicyUpdate({ temperature: null }, policy)).toEqual({ temperature: null });
  });

  it("requires every rematch field", () => {
    expect(error(() => parseRematch({ character: "FOX", imitation: "IBDW#0" }, policy)).status).toBe(422);
  });
});
