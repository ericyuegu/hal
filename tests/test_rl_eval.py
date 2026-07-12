"""Pure-CPU tests for the self-play RL judge (``melee_eval``).

No Dolphin, no GPU: these pin the parts a biased eval would silently break — winner
attribution under port alternation (ema on p1 in even boots, p2 in odd), retry-invariant
parity (a re-queued failed boot must be assigned AND attributed under its new global boot
index — the fake-runner tests drive ``run_h2h`` end-to-end without Dolphin), censoring of
each boot's budget-truncated final segment, the finite-readings discard, the bootstrap
CIs, and the G3 verdict logic — plus the JSON round-trip. The H2H acting path itself
shares the Dolphin-integration-tested collector machinery (live smoke covers it).
"""

import json
from collections.abc import Callable

import numpy as np
import torch
from melee_eval import H2HMatch
from melee_eval import LoadedPolicy
from melee_eval import attribute_h2h
from melee_eval import ema_port_for_boot
from melee_eval import g3_vs_cpu_pass
from melee_eval import readout_from_traj
from melee_eval import run_h2h
from melee_eval import summarize_h2h
from melee_eval import vs_cpu_net_stock_rate
from melee_eval import write_report
from nets_melee import ArchConfig
from nets_melee import PolicyValueNet

from hal.eval.harness import SessionConfig
from hal.eval.scoring import MatchSummary
from hal.sim.trajectory import Trajectory
from hal.sim.vec import Slot
from hal.sim.vec import VecMatch


# --- port alternation -------------------------------------------------------
def test_ema_port_alternates_by_boot_parity() -> None:
    assert ema_port_for_boot(0) == 1
    assert ema_port_for_boot(1) == 2
    assert ema_port_for_boot(2) == 1
    assert ema_port_for_boot(3) == 2


# --- winner attribution -----------------------------------------------------
def _summary(*, p1_stocks: int, p2_stocks: int, p1_dmg: float, p2_dmg: float, frames: int = 3600) -> MatchSummary:
    return MatchSummary(
        frames=frames,
        p1_stocks_left=p1_stocks,
        p2_stocks_left=p2_stocks,
        p1_damage_taken=p1_dmg,
        p2_damage_taken=p2_dmg,
    )


def test_attribute_h2h_ema_on_p1() -> None:
    # ema (p1) ends with 3 stocks, il (p2) with 1; ema dealt what p2 received.
    m = attribute_h2h(_summary(p1_stocks=3, p2_stocks=1, p1_dmg=40.0, p2_dmg=120.0), ema_port=1)
    assert m.ema_stocks_left == 3
    assert m.il_stocks_left == 1
    assert m.ema_damage_dealt == 120.0  # p2 received
    assert m.il_damage_dealt == 40.0  # p1 received


def test_attribute_h2h_ema_on_p2_swaps_sides() -> None:
    # SAME raw readings, but ema is now p2 — attribution must flip, not report p1 as ema.
    m = attribute_h2h(_summary(p1_stocks=3, p2_stocks=1, p1_dmg=40.0, p2_dmg=120.0), ema_port=2)
    assert m.ema_stocks_left == 1
    assert m.il_stocks_left == 3
    assert m.ema_damage_dealt == 40.0  # p1 received (dealt by ema=p2)
    assert m.il_damage_dealt == 120.0


# --- trajectory readout + discard -------------------------------------------
def _traj(p1_stock: list[float], p2_stock: list[float], p1_pct: list[float], p2_pct: list[float]) -> Trajectory:
    n = len(p1_stock)
    return Trajectory(
        frame_id=np.arange(n, dtype=np.int64),
        post={
            1: {"stock": np.array(p1_stock, np.float64), "percent": np.array(p1_pct, np.float64)},
            2: {"stock": np.array(p2_stock, np.float64), "percent": np.array(p2_pct, np.float64)},
        },
        random_seed=np.zeros(n, np.int64),
    )


def test_readout_uses_last_finite_stock() -> None:
    # trailing NaN (IN_GAME -> menu) must not zero the deciding stock reading.
    traj = _traj([4, 4, 3, np.nan], [4, 3, 1, np.nan], [0, 20, 40, np.nan], [0, 50, 130, np.nan])
    m = readout_from_traj(traj, ema_port=1)
    assert m is not None
    assert m.ema_stocks_left == 3
    assert m.il_stocks_left == 1


def test_readout_discards_all_nan_match() -> None:
    traj = _traj([np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan])
    assert readout_from_traj(traj, ema_port=1) is None


def test_readout_discards_single_frame() -> None:
    assert readout_from_traj(_traj([4], [4], [0], [0]), ema_port=1) is None


# --- H2H summary + bootstrap CI ---------------------------------------------
def _match(ema_stocks: int, il_stocks: int, ema_dmg: float = 100.0, il_dmg: float = 100.0) -> H2HMatch:
    return H2HMatch(
        ema_stocks_left=ema_stocks,
        il_stocks_left=il_stocks,
        ema_damage_dealt=ema_dmg,
        il_damage_dealt=il_dmg,
        frames=3600,
    )


def test_summary_win_rate_excludes_ties_from_denominator() -> None:
    matches = [_match(3, 1), _match(2, 2), _match(1, 3), _match(4, 0)]  # 2 wins, 1 tie, 1 loss
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    assert s["n_matches"] == 4
    assert s["n_ties"] == 1
    assert s["ema_win_rate"] == 2 / 3  # 2 wins out of 3 decided (tie dropped from denominator)
    assert s["mean_stock_diff"] == (2 + 0 - 2 + 4) / 4


def test_summary_degenerate_all_wins_ci_above_half() -> None:
    matches = [_match(4, 0) for _ in range(20)]
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    assert s["ema_win_rate"] == 1.0
    lo, _ = s["win_rate_ci95"]
    assert lo > 0.5
    assert s["g3_h2h"] == "PASS"  # CI excludes 0.5 and stock diff > 0


def test_summary_5050_ci_straddles_half() -> None:
    matches = [_match(3, 1) for _ in range(50)] + [_match(1, 3) for _ in range(50)]
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    assert s["ema_win_rate"] == 0.5
    lo, hi = s["win_rate_ci95"]
    assert lo < 0.5 < hi
    assert abs(s["mean_stock_diff"]) < 1e-9
    assert s["g3_h2h"] == "FAIL"  # CI straddles 0.5


def test_summary_g3_fail_when_stock_diff_negative_even_if_ci_excludes_half() -> None:
    # ema loses most matches: win-rate CI excludes 0.5 (below it) but stock diff < 0 -> FAIL.
    matches = [_match(0, 4) for _ in range(20)]
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    hi = s["win_rate_ci95"][1]
    assert hi < 0.5  # CI excludes 0.5
    assert s["mean_stock_diff"] < 0
    assert s["g3_h2h"] == "FAIL"


def test_summary_all_ties_fails_g3_on_nonfinite_ci() -> None:
    # All ties -> no decided matches -> nan win rate and nan CI. The PASS condition must
    # explicitly require a finite CI, not lean on nan-comparison quirks.
    matches = [_match(2, 2) for _ in range(10)]
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    assert s["n_ties"] == 10
    assert not np.isfinite(s["ema_win_rate"])
    assert not np.isfinite(s["win_rate_ci95"][0])
    assert s["g3_h2h"] == "FAIL"


def test_summary_reports_damage_per_min() -> None:
    matches = [_match(3, 1, ema_dmg=180.0, il_dmg=60.0)]  # 1 min match
    s = summarize_h2h(matches, n_discarded=0, seed=0)
    assert s["ema_damage_per_min"] == 180.0
    assert s["il_damage_per_min"] == 60.0


# --- vs-CPU no-regression verdict -------------------------------------------
def _cpu_summary(stocks_taken: int, stocks_lost: int, frames: int = 3600) -> MatchSummary:
    # ego on p1: taken = 4 - p2_left, lost = 4 - p1_left.
    return MatchSummary(
        frames=frames,
        p1_stocks_left=4 - stocks_lost,
        p2_stocks_left=4 - stocks_taken,
        p1_damage_taken=float(stocks_lost) * 100,
        p2_damage_taken=float(stocks_taken) * 100,
    )


def test_vs_cpu_net_stock_rate() -> None:
    summaries = [_cpu_summary(stocks_taken=3, stocks_lost=1)]  # net +2 in 1 minute
    assert vs_cpu_net_stock_rate(summaries) == 2.0


def test_g3_vs_cpu_pass_when_not_worse() -> None:
    # ckpt clearly better than a weak baseline -> PASS (not a regression).
    ckpt = [_cpu_summary(stocks_taken=3, stocks_lost=1) for _ in range(30)]
    assert g3_vs_cpu_pass(ckpt, baseline_net_rate=0.0, seed=0)


def test_g3_vs_cpu_fail_on_regression() -> None:
    # ckpt clearly worse than a strong baseline -> FAIL.
    ckpt = [_cpu_summary(stocks_taken=1, stocks_lost=3) for _ in range(30)]
    assert not g3_vs_cpu_pass(ckpt, baseline_net_rate=2.0, seed=0)


# --- run_h2h retry parity + censoring (fake runner, no Dolphin) ---------------
_TINY_CFG = ArchConfig(
    d_model=32, n_layers=2, n_heads=2, L_ctx=16, char_vocab=8, char_dim=4, stage_vocab=8, stage_dim=2
)


def _loaded_policy() -> LoadedPolicy:
    torch.manual_seed(0)
    net = PolicyValueNet(_TINY_CFG).eval()
    # stats={} is fine: the fake runner never steps the policy, only probes handle_of.
    return LoadedPolicy(net=net, L_ctx=_TINY_CFG.L_ctx, refresh_every=4, stats={}, warm_start="test")


def _session_cfg() -> SessionConfig:
    return SessionConfig(iso_path="unused.ciso", dolphin_path="unused")  # never booted by the fake runner


def _port_wins_traj(winner_port: int) -> Trajectory:
    """3-frame match where ``winner_port`` keeps 4 stocks and the other ends at 0."""
    win_stock, lose_stock = [4.0, 4.0, 4.0], [4.0, 2.0, 0.0]
    if winner_port == 1:
        return _traj(win_stock, lose_stock, [0.0, 0.0, 0.0], [50.0, 100.0, 150.0])
    return _traj(lose_stock, win_stock, [50.0, 100.0, 150.0], [0.0, 0.0, 0.0])


def _partial_traj() -> Trajectory:
    """A budget-truncated in-progress segment (would read as a tie if it were scored)."""
    return _traj([4.0, 3.0], [4.0, 3.0], [10.0, 20.0], [10.0, 20.0])


def _fake_runner(
    fail_first_call_boots: set[int],
    call_log: list[list[VecMatch]],
    assigned_ports_log: list[int] | None = None,
) -> Callable:
    """A ``run_matches_vec`` stand-in that plays every match as a win for whichever port the
    FACTORY'S policy routed to the "ema" handle — so if attribution ever disagrees with
    assignment (the retry sign-flip bug), that match scores as an il win and the test's
    win-rate == 1.0 assertion fails. Boots listed in ``fail_first_call_boots`` return empty
    (never reached IN_GAME) on the fake's first call only, exercising the re-queue path.
    ``assigned_ports_log`` records the factory-assigned ema port per boot, in launch order."""
    if assigned_ports_log is None:
        assigned_ports_log = []
    calls = {"n": 0}

    def runner(session_cfg, matches, factory, *, max_frames, max_parallel, base_slippi_port, start_retries):
        assert start_retries == 0, "run_h2h must disable internal subset retries (Slot.match would desync)"
        assert max_parallel == len(matches)
        call_log.append(list(matches))
        policy = factory()
        boots: list[list[Trajectory]] = []
        for j in range(len(matches)):
            ema_port = 1 if policy.handle_of(Slot(j, 1)) == "ema" else 2
            assert policy.handle_of(Slot(j, ema_port)) == "ema"
            assert policy.handle_of(Slot(j, 3 - ema_port)) == "il"
            assigned_ports_log.append(ema_port)
            if calls["n"] == 0 and j in fail_first_call_boots:
                boots.append([])
                continue
            boots.append([_port_wins_traj(ema_port), _partial_traj()])
        calls["n"] += 1
        return boots

    return runner


def _run_h2h_fake(runner: Callable, *, n_matches_target: int, n_boots: int) -> dict:
    return run_h2h(
        _loaded_policy(),
        _loaded_policy(),
        session_cfg=_session_cfg(),
        n_matches_target=n_matches_target,
        n_boots=n_boots,
        max_frames=1000,
        temp=1.0,
        seed=0,
        device="cpu",
        base_slippi_port=55000,
        runner=runner,
    )


def test_h2h_requeued_boot_keeps_assignment_and_attribution_aligned() -> None:
    """Boot 1 fails in wave 0 and is re-queued under a NEW global boot index; every played
    match is won by the ema-ASSIGNED port, so the summary win rate is 1.0 iff attribution
    used the same parity as assignment for every boot — including the re-queued one."""
    call_log: list[list[VecMatch]] = []
    ports_log: list[int] = []
    s = _run_h2h_fake(_fake_runner({1}, call_log, ports_log), n_matches_target=4, n_boots=2)
    assert s["ema_win_rate"] == 1.0  # any sign-flip would drop this below 1
    assert s["n_matches"] == 4
    assert s["n_ties"] == 0 and s["n_discarded"] == 0
    assert s["g3_h2h"] == "PASS"
    # The failed boot's matchup was re-run in a later wave (identity preserved through the queue).
    failed = call_log[0][1].matchup
    assert any(m.matchup == failed for wave in call_log[1:] for m in wave)
    # Assignment followed the GLOBAL boot index parity across waves and the re-queue:
    # boots 0..3 alternate 1,2,1,2 and the re-run (boot 4) derives afresh -> port 1.
    assert ports_log == [ema_port_for_boot(b) for b in range(len(ports_log))]
    assert set(ports_log) == {1, 2}  # both parities actually exercised


def test_h2h_censors_each_boots_final_segment() -> None:
    """Each boot's last segment is the match in progress at the frame budget — it must be
    dropped as censored (never scored), and counted."""
    call_log: list[list[VecMatch]] = []
    s = _run_h2h_fake(_fake_runner(set(), call_log), n_matches_target=4, n_boots=2)
    assert s["n_matches"] == 4
    n_boots_run = sum(len(wave) for wave in call_log)
    assert s["n_censored"] == n_boots_run  # one censored (in-progress) segment per successful boot
    assert s["n_ties"] == 0  # the partial (tied) segments never entered scoring
    assert s["n_dropped_boots"] == 0  # every boot produced a match, so none exhausted its requeues


# --- JSON round-trip --------------------------------------------------------
def test_json_round_trip(tmp_path) -> None:
    matches = [_match(3, 1), _match(2, 2), _match(1, 3)]
    report = {
        "mode": "h2h",
        "policy": "ema",
        "git_sha": "deadbeef",
        "temp": 1.0,
        **summarize_h2h(matches, n_discarded=2, seed=0),
    }
    out = tmp_path / "eval.json"
    write_report(out, report)
    loaded = json.loads(out.read_text())
    for key in ("mode", "n_matches", "n_ties", "n_discarded", "ema_win_rate", "win_rate_ci95", "g3_h2h"):
        assert key in loaded
    assert loaded["n_discarded"] == 2
    assert isinstance(loaded["win_rate_ci95"], list) and len(loaded["win_rate_ci95"]) == 2
