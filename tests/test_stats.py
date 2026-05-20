"""Pinning tests for ``hal.data.stats``.

These tests stand alone — no MDS fixtures, no Dolphin. They guard the math
(Welford against numpy, merge associativity, mixture analytics) and the
on-disk contract (JSON round-trip, NaN masking).
"""

import math

import numpy as np
import pytest

from hal.data.stats import FeatureStats
from hal.data.stats import FeatureStatsSufficient
from hal.data.stats import StatsAccumulator
from hal.data.stats import dump_sufficient_stats
from hal.data.stats import load_and_merge_stats
from hal.data.stats import load_sufficient_stats

FEATURE = "x"
TOL = 1e-9


def _acc(name: str, values: np.ndarray) -> StatsAccumulator:
    acc = StatsAccumulator([name])
    acc.update(name, values)
    return acc


def test_welford_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(loc=3.5, scale=2.1, size=10_000).astype(np.float64)
    acc = StatsAccumulator([FEATURE])
    for chunk_start in range(0, data.size, 137):
        acc.update(FEATURE, data[chunk_start : chunk_start + 137])
    suff = acc.to_sufficient()[FEATURE]
    assert suff.count == data.size
    assert math.isclose(suff.mean, float(data.mean()), rel_tol=TOL, abs_tol=TOL)
    pop_var = float(data.var(ddof=0))
    assert math.isclose(suff.m2 / suff.count, pop_var, rel_tol=TOL, abs_tol=TOL)


def test_merge_associativity() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=500)
    b = rng.normal(5.0, 2.0, size=700)
    c = rng.normal(-3.0, 0.5, size=900)

    ab = _acc(FEATURE, a).merge(_acc(FEATURE, b))
    ab_c = ab.merge(_acc(FEATURE, c))

    bc = _acc(FEATURE, b).merge(_acc(FEATURE, c))
    a_bc = _acc(FEATURE, a).merge(bc)

    left = ab_c.to_sufficient()[FEATURE]
    right = a_bc.to_sufficient()[FEATURE]
    assert left.count == right.count
    assert math.isclose(left.mean, right.mean, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(left.m2, right.m2, rel_tol=TOL, abs_tol=TOL)
    assert left.min == right.min
    assert left.max == right.max


def test_roundtrip_json(tmp_path) -> None:
    rng = np.random.default_rng(2)
    acc = StatsAccumulator(["a", "b"])
    acc.update("a", rng.normal(0.0, 1.0, size=300))
    acc.update("b", rng.normal(10.0, 5.0, size=400))
    suff = acc.to_sufficient()

    path = tmp_path / "stats.json"
    dump_sufficient_stats(path, suff, split="train", mds_schema_version=2)
    loaded = load_sufficient_stats(path, expected_mds_schema_version=2)

    assert set(loaded) == set(suff)
    for name in suff:
        assert loaded[name].count == suff[name].count
        for field in ("mean", "m2", "min", "max"):
            assert math.isclose(getattr(loaded[name], field), getattr(suff[name], field), rel_tol=TOL, abs_tol=TOL)


def test_schema_version_mismatch_raises(tmp_path) -> None:
    path = tmp_path / "stats.json"
    dump_sufficient_stats(
        path, {FEATURE: FeatureStatsSufficient(1, 0.0, 0.0, 0.0, 0.0)}, split="train", mds_schema_version=2
    )
    with pytest.raises(ValueError, match="mds_schema_version"):
        load_sufficient_stats(path, expected_mds_schema_version=99)


def test_nan_mask() -> None:
    rng = np.random.default_rng(3)
    clean = rng.normal(0.0, 1.0, size=1000)
    contaminated = clean.copy()
    nan_indices = rng.choice(contaminated.size, size=250, replace=False)
    contaminated[nan_indices] = np.nan

    clean_finite = np.delete(clean, nan_indices)
    expected = _acc(FEATURE, clean_finite).to_sufficient()[FEATURE]
    actual = _acc(FEATURE, contaminated).to_sufficient()[FEATURE]

    assert actual.count == expected.count
    assert math.isclose(actual.mean, expected.mean, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(actual.m2, expected.m2, rel_tol=TOL, abs_tol=TOL)
    assert actual.min == expected.min
    assert actual.max == expected.max


def test_mixture_math(tmp_path) -> None:
    """Two streams μ₁=0,σ₁=1 and μ₂=4,σ₂=2 mixed with proportions 0.25 / 0.75.

    Analytic mixture: μ_mix = 0.25·0 + 0.75·4 = 3.0
    Var_mix = 0.25·(1 + 9) + 0.75·(4 + 1) = 2.5 + 3.75 = 6.25
    """
    s1 = {FEATURE: FeatureStatsSufficient(count=1000, mean=0.0, m2=1.0 * 1000, min=-5.0, max=5.0)}
    s2 = {FEATURE: FeatureStatsSufficient(count=1000, mean=4.0, m2=4.0 * 1000, min=-2.0, max=10.0)}

    p1 = tmp_path / "s1.json"
    p2 = tmp_path / "s2.json"
    dump_sufficient_stats(p1, s1, split="train", mds_schema_version=2)
    dump_sufficient_stats(p2, s2, split="train", mds_schema_version=2)

    merged = load_and_merge_stats([p1, p2], proportions=[0.25, 0.75])
    result = merged[FEATURE]
    assert math.isclose(result.mean, 3.0, rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(result.std, math.sqrt(6.25), rel_tol=TOL, abs_tol=TOL)
    assert result.min == -5.0
    assert result.max == 10.0


def test_unweighted_merge_reduces_to_welford(tmp_path) -> None:
    """proportions=None ≡ Welford merge over the union."""
    rng = np.random.default_rng(4)
    a = rng.normal(0.0, 1.0, size=1000)
    b = rng.normal(5.0, 2.0, size=1500)
    s1 = _acc(FEATURE, a).to_sufficient()
    s2 = _acc(FEATURE, b).to_sufficient()

    p1 = tmp_path / "s1.json"
    p2 = tmp_path / "s2.json"
    dump_sufficient_stats(p1, s1, split="train", mds_schema_version=2)
    dump_sufficient_stats(p2, s2, split="train", mds_schema_version=2)

    merged = load_and_merge_stats([p1, p2], proportions=None)[FEATURE]

    union = np.concatenate([a, b])
    assert math.isclose(merged.mean, float(union.mean()), rel_tol=TOL, abs_tol=TOL)
    assert math.isclose(merged.std, float(union.std(ddof=0)), rel_tol=TOL, abs_tol=TOL)


def test_finalize_empty_returns_unit_placeholder() -> None:
    """count=0 (e.g. p1_nana_* when no Ice Climbers) is a real case — the
    accumulator never observed the feature. Return a unit-Gaussian
    placeholder so ``normalize`` / ``standardize`` produce well-defined
    math; downstream the column is NaN-masked anyway."""
    empty = FeatureStatsSufficient(count=0, mean=0.0, m2=0.0, min=math.inf, max=-math.inf)
    placeholder = empty.finalize()
    assert placeholder == FeatureStats(mean=0.0, std=1.0, min=-1.0, max=1.0)


def test_unknown_feature_raises() -> None:
    acc = StatsAccumulator(["known"])
    with pytest.raises(KeyError, match="not registered"):
        acc.update("unknown", np.array([1.0, 2.0]))
