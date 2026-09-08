"""Zero-adjusted Beta fitted as a *true hurdle*, not a zero-inflated mixture.

xgboostlss ships `ZeroAdjustedBeta`, whose `log_prob` scores an exact zero through the
mixture form

    log( gate + (1-gate) * Beta.pdf(epsilon) )

with the support clamped to epsilon. That term lets a Beta with alpha < 1 spike at the
origin and explain the point mass itself, which drives the gate to its floor -- a dead
occurrence gate whose signal leaks into the Beta mean. The historical defence was to floor
alpha at 2 (`softplus_fn_df`), forcing Beta.pdf(0) -> 0. It worked, and it cost the model
the severity distribution: alpha saturated the floor on **every** row (measured 2.000032 to
2.000033 over all 559,440 test rows), so mu was pinned to 2/phi and the severity head was a
one-parameter family Beta(2, b) -- every member unimodal, density vanishing at y = 0. The
burned-fraction positives are J-shaped: their unconstrained marginal MLE is
alpha = 0.39, beta = 107, and 14% of them fall below 1e-4.

`ZABetaHurdleDist` removes the cause instead of the symptom. Under the hurdle factorisation

    y = 0     ->  log(gate)
    y > 0     ->  log(1 - gate) + log Beta(y; alpha, beta)

zeros carry no Beta term at all, so no boundary spike can explain them however small alpha
gets, and the gate must carry P(y=0) on its own. That makes the alpha floor unnecessary:
`concentration1` is a plain softplus and the severity head is free to be J-shaped.

This is also exactly `model.zib.zib_nll`, the score every other model class in the repo is
ranked by, so the training loss, the early-stopping metric and the reported NLL are now one
function rather than three.

`concentration0 <= C0_CAP` is kept. It guards a different failure: left unbounded, boosting
inflates beta (~900 -> ~20,000 over many rounds) to sharpen the Beta onto the many tiny
positive burned fractions, and a Beta that sharp is numerically brittle at the small y those
rows carry. C0_CAP=3000 leaves full headroom over the ~1730 a healthy model uses.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import torch

from xgboostlss.distributions.zero_inflated import ZeroAdjustedBeta as _ZAB
from xgboostlss.distributions.distribution_utils import DistributionClass
from xgboostlss.utils import softplus_fn, sigmoid_fn, nan_to_num

from ..zib import zib_nll

C0_CAP = 3000.0

def spec(c0_cap: float = C0_CAP) -> str:
    return f"ZABetaHurdle(true-hurdle log_prob, c1 free, c0<={c0_cap:g})"


def _capped_softplus(predt: torch.Tensor, cap: float = C0_CAP) -> torch.Tensor:
    """concentration0 in (0, cap): a smooth ceiling on how sharp the Beta can get.

    The cap is a sigmoid, so a row sitting near it is also sitting where the response
    function's slope has collapsed -- check the fitted beta histogram against it rather
    than assuming it is slack. Bound with `functools.partial`, never a closure:
    `param_dict` is pickled by `XGBoostLSS.save_model`, and a closure is not picklable."""
    return cap * torch.sigmoid(nan_to_num(predt)) + torch.tensor(1e-06, dtype=predt.dtype)


class ZABetaHurdleDist(_ZAB):
    """ZeroAdjustedBeta with the hurdle log-density in place of the mixture one.

    The parent evaluates the Beta at every row, including the zeros, and adds its density
    into the zero's probability. Here zeros score `log(gate)` alone; the Beta is evaluated
    only where it applies. `torch.where` still differentiates the unselected branch, so the
    zeros get a dummy in place of their 0 -- that keeps their gradient contribution to alpha
    and beta an exact zero rather than a NaN, and leaves every positive y as observed.
    """

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        gate = self.gate
        is_zero = value == 0
        y = torch.where(is_zero, torch.full_like(value, 0.5), value)
        return torch.where(is_zero,
                           torch.log(gate),
                           torch.log1p(-gate) + self.base_dist.log_prob(y))


class ZABetaHurdle(DistributionClass):
    def __init__(self, stabilization: str = "L2", loss_fn: str = "nll",
                 c0_cap: float = C0_CAP):
        if stabilization not in ("None", "MAD", "L2"):
            raise ValueError("stabilization must be 'None', 'MAD' or 'L2'")
        if loss_fn != "nll":
            raise ValueError("loss_fn must be 'nll'")
        self.c0_cap = c0_cap
        param_dict = {
            "concentration1": softplus_fn,       # softplus(.)      -> alpha > 0, unfloored
            "concentration0": partial(_capped_softplus, cap=c0_cap),  # 0 < beta <= cap
            "gate": sigmoid_fn,                  # P(y == 0)
        }
        torch.distributions.Distribution.set_default_validate_args(False)
        super().__init__(
            distribution=ZABetaHurdleDist,
            univariate=True,
            discrete=False,
            n_dist_param=len(param_dict),
            stabilization=stabilization,
            param_dict=param_dict,
            distribution_arg_names=list(param_dict.keys()),
            loss_fn=loss_fn,
        )

    def metric_fn(self, predt: np.ndarray, data) -> tuple[str, float]:
        """Early-stopping signal = mean `zib_nll`, which the training loss now equals.

        The parent would report the same quantity as a float32 sum; this reports the
        per-row mean in float64, on the scale every metrics.json in the repo uses."""
        target = torch.tensor(data.get_label().reshape(-1, 1), dtype=torch.float64)
        start_values = data.get_base_margin().reshape(-1, self.n_dist_param)[0, :].tolist()
        predt = np.array(predt, dtype=np.float64).reshape(-1, self.n_dist_param)
        mask = np.isnan(predt) | np.isinf(predt)
        predt[mask] = np.take(start_values, np.where(mask)[1])
        cols = [torch.tensor(predt[:, i].reshape(-1, 1)) for i in range(self.n_dist_param)]
        alpha, beta, gate = (fn(cols[i]) for i, fn in enumerate(self.param_dict.values()))
        gate = gate.clamp(1e-6, 1 - 1e-6)
        pi_logit = torch.log(gate) - torch.log1p(-gate)   # logit of P(y==0)
        phi = alpha + beta
        mu = (alpha / phi).clamp(1e-6, 1 - 1e-6)
        loss = zib_nll(pi_logit, mu, phi, target, link="logit")
        return "true_nll", float(loss.detach())
