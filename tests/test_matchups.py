"""Tests for the deterministic eval-matchup allocator (``hal.eval.matchups``).

The cached ``MATCHUP_PRIOR`` is data; what needs proving is that ``matchups_for``
is a pure, prefix-stable, weight-proportional function of ``n`` so eval coverage
is identical across runs and only grows (never reshuffles) with parallelism.
"""

from collections import Counter

from melee import Character

from hal.eval.matchups import MATCHUP_PRIOR
from hal.eval.matchups import matchups_for
from hal.policy import INCLUDED_CHARACTERS


def test_prior_is_in_policy_and_canonically_ordered() -> None:
    included = set(INCLUDED_CHARACTERS)
    for a, b, w in MATCHUP_PRIOR:
        assert a in included and b in included
        assert a.value <= b.value, f"({a.name},{b.name}) not canonically ordered"
        assert w > 0
    # weights strictly describe a real distribution; the most common is Fox/Falco.
    assert MATCHUP_PRIOR[0][:2] == (Character.FOX, Character.FALCO)


def test_empty_and_negative() -> None:
    assert matchups_for(0) == []
    try:
        matchups_for(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for n<0")


def test_deterministic() -> None:
    assert matchups_for(64) == matchups_for(64)


def test_prefix_stable() -> None:
    """Growing parallelism only appends — the divisor-method order is n-independent."""
    big = matchups_for(200)
    for n in (1, 7, 16, 64, 128, 199):
        assert matchups_for(n) == big[:n]


def test_returns_exactly_n() -> None:
    for n in (1, 3, 50, 300, 1000):
        assert len(matchups_for(n)) == n


def test_first_slot_is_most_common_matchup() -> None:
    ((ego, opp),) = matchups_for(1)
    assert {ego, opp} == {Character.FOX, Character.FALCO}


def test_weight_proportional_at_large_n() -> None:
    """At large n the per-matchup slot share tracks its weight share (divisor method)."""
    n = 20_000
    alloc = matchups_for(n)
    # Collapse orientation back to the unordered matchup key.
    got = Counter(tuple(sorted((e.value, o.value))) for e, o in alloc)
    total_w = sum(w for _, _, w in MATCHUP_PRIOR)
    for a, b, w in MATCHUP_PRIOR[:20]:  # check the head, where counts are statistically meaningful
        key = tuple(sorted((a.value, b.value)))
        assert abs(got[key] / n - w / total_w) < 0.003, f"{a.name}/{b.name} share off"


def test_orientation_alternates_so_model_plays_both_sides() -> None:
    """A non-mirror matchup with >=2 slots appears in both ego/opp orientations."""
    alloc = matchups_for(2000)
    # Fox/Falco is the top non-mirror matchup; it gets many slots.
    fox_falco = [(e, o) for e, o in alloc if {e, o} == {Character.FOX, Character.FALCO}]
    assert (Character.FOX, Character.FALCO) in fox_falco
    assert (Character.FALCO, Character.FOX) in fox_falco
