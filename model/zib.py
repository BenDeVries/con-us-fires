"""Zero-inflated Beta likelihood and evaluation metrics.

The target burned fraction has a point mass at 0 (~92% of county-months) and a
continuous part on (0,1) (observed max ~0.56, no exact 1s -> no one-inflation).
We model it as:  P(y=0) = pi ;  y | y>0 ~ Beta(alpha, beta)  with
alpha = mu*phi, beta = (1-mu)*phi, mu in (0,1) the conditional mean, phi>0 precision.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:  # sklearn optional at import time
    roc_auc_score = average_precision_score = None

_LOG2 = 0.6931471805599453


def _log1mexp(a: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(-a)) for a > 0 (Mächler 2012).

    Both torch.where branches are kept finite so the unused branch does not poison
    gradients with NaNs: the small-a branch uses log(-expm1(-a)) and the large-a
    branch uses log1p(-exp(-a)), each evaluated on a clamped copy of `a`.
    """
    a = a.clamp(min=1e-12)
    small = torch.log(-torch.expm1(-a.clamp(max=_LOG2)))
    large = torch.log1p(-torch.exp(-a.clamp(min=_LOG2)))
    return torch.where(a <= _LOG2, small, large)


def gate_log_probs(pi_logit: torch.Tensor, link: str = "logit"):
    """Return (log pi, log(1-pi)) for the zero gate under the chosen link.

    The occurrence predictor is eta = -pi_logit, so P(fire) = inverse_link(eta) and
    pi = P(y=0) = 1 - inverse_link(eta). The negation keeps the sign convention identical
    across links: a larger occurrence predictor always means more fire.
    """
    if link == "logit":
        return F.logsigmoid(pi_logit), F.logsigmoid(-pi_logit)
    if link == "cloglog":
        # eta = -pi_logit; P(fire) = 1 - exp(-exp(eta)); pi = exp(-exp(eta)).
        z = torch.exp((-pi_logit).clamp(max=30.0))     # = exp(eta), the Poisson rate
        return -z, _log1mexp(z)                         # log pi = -z ; log(1-pi) = log1mexp(z)
    raise ValueError(f"unknown link {link!r}")


def gate_fire_prob(pi_logit: torch.Tensor, link: str = "logit") -> torch.Tensor:
    """P(fire) = 1 - pi under the chosen link."""
    if link == "logit":
        return torch.sigmoid(-pi_logit)
    if link == "cloglog":
        return -torch.expm1(-torch.exp((-pi_logit).clamp(max=30.0)))
    raise ValueError(f"unknown link {link!r}")


def sample_zib(pi_logit: torch.Tensor, mu: torch.Tensor, phi: torch.Tensor,
               link: str = "logit", eps: float = 1e-6) -> torch.Tensor:
    """One ancestral draw from the zero-inflated Beta.

    occ ~ Bernoulli(P(fire));  if fire, y ~ Beta(mu*phi, (1-mu)*phi);  y = occ * y_beta.
    Draws from the global torch RNG (torch.distributions.Beta has no per-call generator),
    so seed once with torch.manual_seed for reproducible rollouts. Returns y in [0, 1-eps]."""
    p_fire = gate_fire_prob(pi_logit, link)
    occ = torch.bernoulli(p_fire)
    alpha = mu * phi
    beta = (1 - mu) * phi
    y_beta = torch.distributions.Beta(alpha, beta).sample()
    return (occ * y_beta).clamp(0.0, 1 - eps)


def _sens_spec_sweep(label: np.ndarray, p_occ: np.ndarray):
    """Sensitivity and specificity at every cutpoint, plus the sorted scores.

    Predictions are sorted descending; entry k is the operating point that calls the
    top (k+1) scores positive (i.e. threshold p_occ >= p_sorted[k])."""
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    order = np.argsort(p_occ)[::-1]
    p_sorted = p_occ[order]
    lab = label[order].astype(float)
    tp = np.cumsum(lab)
    fp = np.arange(1, len(lab) + 1) - tp
    sens = tp / n_pos
    spec = (n_neg - fp) / n_neg
    return p_sorted, sens, spec, n_pos, n_neg


def _balanced_accuracy(label: np.ndarray, p_occ: np.ndarray) -> float:
    """Balanced accuracy at the threshold that maximises sensitivity + specificity."""
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    _, sens, spec, _, _ = _sens_spec_sweep(label, p_occ)
    return float(((sens + spec) / 2).max())


def optimal_gate_threshold(label: np.ndarray, p_occ: np.ndarray) -> float:
    """p_occ cutpoint that maximises sensitivity + specificity (Youden's J).

    Exact O(n log n) sweep over all cuts -- the operating point implied by
    `_balanced_accuracy`, which reports the score at this cut. A diagnostic /
    visualisation utility only (training and HPO optimise the ZIB NLL, not
    balanced accuracy). Returned as the midpoint between the two scores
    straddling the optimum so `p_occ > thr` reproduces the optimal positive set."""
    n_pos = int(label.sum())
    n_neg = len(label) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    p_sorted, sens, spec, _, _ = _sens_spec_sweep(label, p_occ)
    k = int(np.argmax(sens + spec))
    hi = float(p_sorted[k])
    lo = float(p_sorted[k + 1]) if k + 1 < len(p_sorted) else hi
    return (hi + lo) / 2


def zib_nll(pi_logit: torch.Tensor, mu: torch.Tensor, phi: torch.Tensor,
            y: torch.Tensor, reduction: str = "mean",
            link: str = "logit") -> torch.Tensor:
    """Negative log-likelihood of the zero-inflated Beta. All tensors broadcast-compatible.

    y enters the Beta term exactly as observed. Zero rows are masked out of that term
    rather than clamped into (0,1): the panel's smallest positive burned fraction is
    7.96e-08, so any clamp big enough to guard the logarithm also censors real data.
    """
    is_pos = y > 0
    # log P(zero gate) and log P(fire), under the chosen gate link.
    log_pi, log_1mpi = gate_log_probs(pi_logit, link)

    # The Beta term is discarded on zero rows, but torch.where still differentiates it:
    # a log(0) there backs up as 0 * inf = NaN. The dummy keeps that dead branch finite
    # without touching any y the likelihood actually reads.
    yc = torch.where(is_pos, y, torch.full_like(y, 0.5))
    alpha = mu * phi
    beta = (1 - mu) * phi
    log_beta_pdf = ((alpha - 1) * torch.log(yc)
                    + (beta - 1) * torch.log1p(-yc)
                    + torch.lgamma(alpha + beta)
                    - torch.lgamma(alpha) - torch.lgamma(beta))

    nll_pos = -(log_1mpi + log_beta_pdf)
    nll_zero = -log_pi
    nll = torch.where(is_pos, nll_pos, nll_zero)

    if reduction == "mean":
        return nll.mean()
    if reduction == "none":
        return nll
    return nll.sum()


@torch.no_grad()
def compute_metrics(pi_logit: torch.Tensor, mu: torch.Tensor, phi: torch.Tensor,
                    y: torch.Tensor, link: str = "logit") -> dict:
    """Aggregate metrics over a [.., H, ..] prediction tensor (any shape, flattened)."""
    nll = zib_nll(pi_logit, mu, phi, y, link=link).item()
    p_fire = gate_fire_prob(pi_logit, link)
    p_occ = p_fire.flatten().cpu().numpy()
    mu_f = mu.flatten().cpu().numpy()
    ey = (p_fire * mu).flatten().cpu().numpy()
    yf = y.flatten().cpu().numpy()
    label = (yf > 0).astype("int8")

    out = {"nll": nll, "mae_full": float(abs(ey - yf).mean()),
           "frac_pos": float(label.mean())}
    if label.any() and label.mean() < 1.0:
        out["balanced_accuracy"] = _balanced_accuracy(label, p_occ)
        if roc_auc_score is not None:
            out["gate_auc"] = float(roc_auc_score(label, p_occ))
            out["gate_ap"] = float(average_precision_score(label, p_occ))
    pos = label.astype(bool)
    if pos.any():
        out["mae_pos"] = float(abs(mu_f[pos] - yf[pos]).mean())
    return out


@torch.no_grad()
def per_horizon_nll(pi_logit: torch.Tensor, mu: torch.Tensor, phi: torch.Tensor,
                    y: torch.Tensor, link: str = "logit") -> list[float]:
    """Mean NLL for each horizon step; expects tensors shaped [B, H, N]."""
    nll = zib_nll(pi_logit, mu, phi, y, reduction="none", link=link)   # [B,H,N]
    return nll.mean(dim=(0, 2)).cpu().tolist()
