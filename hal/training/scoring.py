"""Shared discretizers + proper-scoring rules + PF-ODE likelihood for action-chunk
experiments.

One source of truth so a classification experiment's targets/decode and every run's
comparison metrics agree byte-for-byte (CLAUDE.md: shared infra never lives in
``experiments/``). Three groups:

* **discretizers** — uniform per-channel ``BinSpec`` bins and hand-tuned joint-2D stick
  ``cluster`` centers, plus the single-label button-class map. The target path and the
  decode path call the same code, so there is no second quantizer to drift.
* **proper scores** — Bernoulli / categorical NLL in **bits** (nats/ln2) and Brier; a
  ``cont_density_bits`` correction expresses a binned *continuous* dim as a density so it
  lines up with a flow model's PF-ODE bits/dim.
* **PF-ODE bits/dim** — the continuous-NLL of a flow model via the instantaneous
  change-of-variables (Hutchinson trace), so flow and classification runs share a
  likelihood axis.

Stick ranges are ``[-1, 1]`` and trigger ranges ``[0, 1]`` — the action-vector
conventions in ``hal/training/features.py:action_vec_to_controller``.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

_LN2 = math.log(2.0)


# --- discretizers ------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class BinSpec:
    """Uniform bins over ``[lo, hi)`` (last bin closed). ``to_idx`` and ``centers`` are
    exact inverses up to half a bin width; an odd ``n_bins`` over a symmetric range puts
    a center exactly on the midpoint (neutral stick = 0)."""

    lo: float
    hi: float
    n_bins: int

    @property
    def width(self) -> float:
        return (self.hi - self.lo) / self.n_bins

    def to_idx(self, x: Tensor) -> Tensor:
        idx = ((x - self.lo) / self.width).floor().long()
        return idx.clamp(0, self.n_bins - 1)

    def centers(self, device: torch.device | None = None) -> Tensor:
        return self.lo + (torch.arange(self.n_bins, device=device) + 0.5) * self.width


# Continuous action channels in ACTION_CHANNELS order: main_x, main_y, c_x, c_y (sticks,
# [-1,1]), trigger_l, trigger_r ([0,1]). The discretization source of truth.
_CONT_RANGES: tuple[tuple[float, float], ...] = (
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (-1.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
)


def cont_binspecs(n_bins: int) -> tuple[BinSpec, ...]:
    return tuple(BinSpec(lo, hi, n_bins) for lo, hi in _CONT_RANGES)


CONT_BINSPECS: tuple[BinSpec, ...] = cont_binspecs(21)


def bins_to_idx(cont: Tensor, binspecs: tuple[BinSpec, ...]) -> Tensor:
    """``[..., C]`` continuous values → ``[..., C]`` long bin indices, per channel."""
    return torch.stack([spec.to_idx(cont[..., i]) for i, spec in enumerate(binspecs)], dim=-1)


def idx_to_centers(idx: Tensor, binspecs: tuple[BinSpec, ...]) -> Tensor:
    """Inverse of ``bins_to_idx``: ``[..., C]`` indices → their bin centers."""
    return torch.stack([spec.centers(idx.device)[idx[..., i]] for i, spec in enumerate(binspecs)], dim=-1)


# --- joint-2D stick clusters -------------------------------------------------
# Hand-tuned (x, y) stick targets in the action [-1, 1] space: neutral, partial/full
# tilts, cardinals, wavedash & ledgedash angles (17/30/45/60/72.5 deg), shield-drop and
# angled-tilt diagonals. A *joint* categorical over correlated x/y, unlike per-axis bins.
_STICK_CLUSTER_XY: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (0.35, 0.0),
    (-0.35, 0.0),
    (0.0, 0.35),
    (0.0, -0.35),
    (0.675, 0.0),
    (-0.675, 0.0),
    (0.0, 0.675),
    (0.0, -0.675),
    (1.0, 0.0),
    (0.0, 1.0),
    (-1.0, 0.0),
    (0.0, -1.0),
    (0.95, -0.3),
    (-0.95, -0.3),
    (0.95, 0.3),
    (-0.95, 0.3),
    (0.85, -0.5),
    (0.85, 0.5),
    (-0.85, -0.5),
    (-0.85, 0.5),
    (0.7, -0.7),
    (-0.7, -0.7),
    (0.7, 0.7),
    (-0.7, 0.7),
    (0.5, 0.5),
    (-0.5, 0.5),
    (0.5, -0.5),
    (-0.5, -0.5),
    (0.5, 0.85),
    (-0.5, 0.85),
    (0.5, -0.85),
    (-0.5, -0.85),
    (0.3, -0.95),
    (0.3, 0.95),
    (-0.3, -0.95),
    (-0.3, 0.95),
)

STICK_CLUSTER_CENTERS_MAIN: Tensor = torch.tensor(_STICK_CLUSTER_XY, dtype=torch.float32)
# Start identical; the quantization audit may fit a c-stick-specific set (mostly cardinals).
STICK_CLUSTER_CENTERS_C: Tensor = STICK_CLUSTER_CENTERS_MAIN.clone()


def nearest_cluster(xy: Tensor, centers: Tensor) -> Tensor:
    """``[..., 2]`` stick coords → ``[...]`` index of the nearest (L2) cluster center."""
    c = centers.to(xy.device)
    return (xy.unsqueeze(-2) - c).pow(2).sum(-1).argmin(-1)


def cluster_to_xy(idx: Tensor, centers: Tensor) -> Tensor:
    """Inverse of ``nearest_cluster``: ``[...]`` indices → ``[..., 2]`` center coords."""
    return centers.to(idx.device)[idx]


# --- single-label button class map -------------------------------------------
def buttons_to_class(buttons: Tensor) -> tuple[Tensor, Tensor]:
    """``[..., 8]`` button bits {0,1} → (``[...]`` class in 0..8, ``[...]`` multi-press mask).

    Class 0 = no button; class ``k`` (1..8) = button index ``k-1``. Concurrent presses
    resolve to the lowest button index (priority = ACTION_CHANNELS order); ``multi_press``
    flags the frames where that one-button assumption was violated."""
    pressed = buttons > 0.5
    any_pressed = pressed.any(-1)
    first = pressed.float().argmax(-1)  # first True == lowest-index pressed
    cls = torch.where(any_pressed, first + 1, torch.zeros_like(first)).long()
    multi_press = pressed.sum(-1) >= 2
    return cls, multi_press


def class_to_onehot(cls: Tensor, *, n_buttons: int = 8) -> Tensor:
    """Inverse of ``buttons_to_class``: class 0 → all-zero, class ``k`` → one-hot at ``k-1``."""
    oh = torch.zeros(*cls.shape, n_buttons, device=cls.device)
    idx = (cls - 1).clamp(min=0).unsqueeze(-1)
    oh.scatter_(-1, idx, 1.0)
    return oh * (cls > 0).unsqueeze(-1)


# --- proper scoring rules (return bits = nats / ln2) -------------------------
def bernoulli_logloss_bits(logits: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    """Bernoulli log-loss in bits from logits and {0,1} targets (per-channel BCE)."""
    bits = F.binary_cross_entropy_with_logits(logits, target, reduction="none") / _LN2
    return bits.mean() if reduction == "mean" else bits


def categorical_nll_bits(logits: Tensor, idx: Tensor, reduction: str = "mean") -> Tensor:
    """Categorical NLL in bits from class logits ``[..., K]`` and indices ``[...]``."""
    bits = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1), reduction="none") / _LN2
    return bits.mean() if reduction == "mean" else bits.reshape(idx.shape)


def brier_bernoulli(probs: Tensor, target: Tensor) -> Tensor:
    """Mean squared error between predicted probabilities and {0,1} targets."""
    return (probs - target).pow(2).mean()


def bernoulli_scores_from_probs(probs: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """``(logloss_bits, brier)`` from probabilities rather than logits — e.g. Monte-Carlo
    button frequencies from a flow model, or a single-label head's per-button marginals.
    Probabilities are clamped off {0,1} so the log-loss stays finite."""
    p = probs.clamp(1e-6, 1 - 1e-6)
    logloss = -(target * p.log2() + (1 - target) * (1 - p).log2()).mean()
    return logloss, (probs - target).pow(2).mean()


def brier_categorical(probs: Tensor, onehot: Tensor) -> Tensor:
    """Multiclass Brier: mean over elements of the summed squared prob error."""
    return (probs - onehot).pow(2).sum(-1).mean()


def cont_density_bits(discrete_nll_bits: Tensor, bin_width: float) -> Tensor:
    """Convert a binned *continuous* dim's discrete NLL (bits) to a density bits/dim by the
    bin-width Jacobian: density = mass / width ⇒ ``-log2 density = nll_bits + log2(width)``.
    Makes a binned-continuous score comparable to a flow model's PF-ODE bits/dim."""
    return discrete_nll_bits + math.log2(bin_width)


# --- PF-ODE bits/dim (flow-model continuous likelihood) ----------------------
def _hutchinson_divergence(velocity_fn, x: Tensor, t: Tensor, gen: torch.Generator | None, n_probes: int) -> Tensor:
    """E_ε[εᵀ J ε] ≈ tr(J) of ∂v/∂x via Rademacher probes (exact for a linear field with a
    single probe). Returns ``[N]``; 0 for an x-independent field."""
    total = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    with torch.enable_grad():
        x_ = x.detach().requires_grad_(True)
        v = velocity_fn(x_, t)
        if not v.requires_grad:  # x-independent field (e.g. pure translation) → zero divergence
            return total
        for p in range(n_probes):
            eps = torch.randint(0, 2, x_.shape, generator=gen, device=x.device, dtype=torch.float32) * 2 - 1
            # allow_unused: ``v`` may carry grad through model params yet not depend on ``x_``;
            # autograd then returns None for the input grad, which is a zero contribution.
            vjp = torch.autograd.grad(v, x_, grad_outputs=eps, retain_graph=(p < n_probes - 1), allow_unused=True)[0]
            if vjp is not None:
                total = total + (vjp * eps).flatten(1).sum(1)
    return (total / n_probes).detach()


def pf_ode_bits_per_dim(
    velocity_fn, x1: Tensor, *, n_steps: int, gen: torch.Generator | None = None, n_probes: int = 1
) -> Tensor:
    """Continuous NLL (bits/dim) of data ``x1`` under the flow ``dx/dt = velocity_fn(x, t)``.

    Integrates the augmented PF-ODE backward t=1→0 from the data with the instantaneous
    change of variables ``log p₁(x₁) = log p₀(x₀) - ∫₀¹ div(v) dt`` (N(0,I) base). Run in
    fp32 (the caller disables autocast); ``velocity_fn(a, t)`` closes over the experiment's
    own bound velocity so this stays architecture-agnostic. Returns ``[N]``."""
    x = x1.to(torch.float32).clone()
    n, d = x.shape[0], x[0].numel()
    div_integral = torch.zeros(n, device=x.device, dtype=torch.float32)
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((n,), 1.0 - k * dt, device=x.device, dtype=torch.float32)
        div_integral = div_integral + _hutchinson_divergence(velocity_fn, x, t, gen, n_probes) * dt
        with torch.no_grad():
            x = x - dt * velocity_fn(x, t).to(torch.float32)
    log_p0 = -0.5 * (d * math.log(2 * math.pi) + x.flatten(1).pow(2).sum(1))
    log_p1 = log_p0 - div_integral
    return -log_p1 / (d * _LN2)
