"""Unit contracts for hal.training.scoring — the shared discretizers + proper-scoring
rules + PF-ODE bits/dim helper used by every action-chunk experiment.

The discretizers are the single source of truth for turning continuous action targets
into class indices and back; the proper scores must be exact at their analytic anchors
(uniform → 1 bit, perfect → 0 bits); the PF-ODE NLL must match the closed form of a
known flow (the critical correctness gate, since divergence sign/time-direction are easy
to get wrong)."""

import math

import torch

from hal.training import scoring


# --- discretizers ------------------------------------------------------------
def test_binspec_round_trip_within_half_a_bin():
    spec = scoring.BinSpec(lo=-1.0, hi=1.0, n_bins=21)
    x = torch.linspace(-1.0, 1.0, 257)
    recon = spec.centers()[spec.to_idx(x)]
    assert (recon - x).abs().max() <= spec.width / 2 + 1e-6


def test_odd_bins_center_on_neutral():
    spec = scoring.BinSpec(lo=-1.0, hi=1.0, n_bins=21)
    assert torch.isclose(spec.centers()[spec.to_idx(torch.tensor(0.0))], torch.tensor(0.0), atol=1e-6)


def test_binspec_clamps_out_of_range():
    spec = scoring.BinSpec(lo=0.0, hi=1.0, n_bins=8)
    idx = spec.to_idx(torch.tensor([-5.0, 5.0]))
    assert idx.tolist() == [0, spec.n_bins - 1]


def test_bins_to_idx_per_channel_round_trip():
    # 6 continuous channels: sticks [-1,1]x4, triggers [0,1]x2
    cont = torch.tensor([[0.0, -1.0, 1.0, 0.5, 0.0, 1.0]])
    idx = scoring.bins_to_idx(cont, scoring.CONT_BINSPECS)
    recon = scoring.idx_to_centers(idx, scoring.CONT_BINSPECS)
    widths = torch.tensor([s.width for s in scoring.CONT_BINSPECS])
    assert idx.shape == cont.shape
    assert (recon - cont).abs().squeeze(0).le(widths / 2 + 1e-6).all()


# --- joint-2D stick clusters -------------------------------------------------
def test_stick_centers_live_in_target_space():
    for centers in (scoring.STICK_CLUSTER_CENTERS_MAIN, scoring.STICK_CLUSTER_CENTERS_C):
        assert centers.ndim == 2 and centers.shape[1] == 2
        assert centers.min() >= -1.0 and centers.max() <= 1.0
    # neutral, the four cardinals are present (canonical [-1,1] coords)
    main = scoring.STICK_CLUSTER_CENTERS_MAIN
    for pt in ([0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
        assert (main == torch.tensor(pt)).all(dim=1).any()


def test_nearest_cluster_is_identity_on_the_centers():
    centers = scoring.STICK_CLUSTER_CENTERS_MAIN
    idx = scoring.nearest_cluster(centers, centers)
    assert torch.equal(idx, torch.arange(centers.shape[0]))
    assert torch.equal(scoring.cluster_to_xy(idx, centers), centers)


# --- button class map (single-label) -----------------------------------------
def test_buttons_to_class_none_and_single():
    b = torch.zeros(3, 8)
    b[1, 2] = 1.0  # only button index 2 pressed
    cls, multi = scoring.buttons_to_class(b)
    assert cls.tolist() == [0, 3, 0]  # none=0, idx2 -> class 3
    assert multi.tolist() == [False, False, False]


def test_buttons_to_class_priority_and_multipress_flag():
    b = torch.zeros(1, 8)
    b[0, 1] = 1.0
    b[0, 5] = 1.0  # two pressed; priority = ACTION_CHANNELS order -> lower index wins
    cls, multi = scoring.buttons_to_class(b)
    assert cls.item() == 2  # button index 1 -> class 2
    assert multi.item() is True


def test_class_to_onehot_inverts_single_button():
    b = torch.zeros(4, 8)
    b[0, 0] = 1.0
    b[2, 7] = 1.0
    cls, _ = scoring.buttons_to_class(b)
    assert torch.equal(scoring.class_to_onehot(cls, n_buttons=8), b)


# --- proper scoring rules ----------------------------------------------------
def test_bernoulli_logloss_uniform_is_one_bit():
    logits = torch.zeros(5, 8)  # p = 0.5 everywhere
    target = torch.randint(0, 2, (5, 8)).float()
    assert math.isclose(scoring.bernoulli_logloss_bits(logits, target).item(), 1.0, abs_tol=1e-6)


def test_bernoulli_logloss_perfect_is_near_zero():
    target = torch.tensor([[1.0, 0.0, 1.0]])
    logits = torch.where(target > 0.5, 20.0, -20.0)
    assert scoring.bernoulli_logloss_bits(logits, target).item() < 1e-6


def test_categorical_nll_uniform_is_log2_k_bits():
    logits = torch.zeros(7, 9)  # uniform over 9 classes
    idx = torch.randint(0, 9, (7,))
    assert math.isclose(scoring.categorical_nll_bits(logits, idx).item(), math.log2(9), abs_tol=1e-6)


def test_brier_bernoulli_perfect_is_zero_worst_is_one():
    target = torch.tensor([1.0, 0.0])
    assert scoring.brier_bernoulli(target, target).item() < 1e-9
    assert math.isclose(scoring.brier_bernoulli(1.0 - target, target).item(), 1.0, abs_tol=1e-9)


def test_bernoulli_scores_from_probs_anchors():
    target = torch.tensor([[1.0, 0.0, 1.0]])
    ll, br = scoring.bernoulli_scores_from_probs(target, target)  # perfect
    assert ll.item() < 1e-3 and br.item() < 1e-9
    half = torch.full_like(target, 0.5)  # uniform -> 1 bit, brier 0.25
    ll, br = scoring.bernoulli_scores_from_probs(half, target)
    assert math.isclose(ll.item(), 1.0, abs_tol=1e-4) and math.isclose(br.item(), 0.25, abs_tol=1e-9)


def test_cont_density_bits_subtracts_log2_width():
    # density = mass / width  ->  -log2 density = discrete_nll_bits + log2(width)
    out = scoring.cont_density_bits(torch.tensor(3.0), bin_width=0.1)
    assert math.isclose(out.item(), 3.0 + math.log2(0.1), abs_tol=1e-6)


# --- PF-ODE bits/dim (the correctness gate) ----------------------------------
def _gaussian_nll_bits_per_dim(x: torch.Tensor) -> torch.Tensor:
    """Closed-form N(0,I) NLL in bits/dim for each row of x."""
    D = x[0].numel()
    log_p = -0.5 * (D * math.log(2 * math.pi) + x.flatten(1).pow(2).sum(1))
    return -log_p / (D * math.log(2))


def test_pf_ode_translation_field_matches_shifted_gaussian():
    # v(x,t) = c constant  ->  divergence 0, x0 = x1 - c, p1 = N(c, I)
    c = torch.tensor([0.7, -0.3, 1.1, 0.0])
    x1 = torch.randn(16, 4)

    def velocity_fn(a, t):
        return c.expand_as(a)

    got = scoring.pf_ode_bits_per_dim(velocity_fn, x1, n_steps=64)
    want = _gaussian_nll_bits_per_dim(x1 - c)  # p1(x1) = N(x1; c, I)
    assert torch.allclose(got, want, atol=1e-4)


def test_pf_ode_linear_field_matches_closed_form():
    # v(x,t) = alpha*x  ->  x0 = x1*exp(-alpha), integral div dt = alpha*D,
    # log p1(x1) = log N(x0) - alpha*D.  Hutchinson is exact for a linear field.
    alpha = 0.5
    x1 = torch.randn(8, 3)

    def velocity_fn(a, t):
        return alpha * a

    got = scoring.pf_ode_bits_per_dim(velocity_fn, x1, n_steps=2000, gen=torch.Generator().manual_seed(0))
    D = x1[0].numel()
    x0 = x1 * math.exp(-alpha)
    log_p1 = -0.5 * (D * math.log(2 * math.pi) + x0.pow(2).sum(1)) - alpha * D
    want = -log_p1 / (D * math.log(2))
    assert torch.allclose(got, want, rtol=2e-3, atol=2e-3)
