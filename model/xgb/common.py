"""Shared paths and the ZABeta <-> (pi, mu, phi) parameter mapping.

xgboostlss' ZABeta emits (concentration1=alpha, concentration0=beta, gate=P(y=0)).
The NN speaks in (pi_logit, mu, phi); we convert so model/zib.py metrics apply
unchanged for an apples-to-apples comparison:
    p_occ = 1 - gate ;  mu = alpha/(alpha+beta) ;  phi = alpha+beta
    pi_logit = logit(gate)         (so gate_fire_prob(pi_logit)=sigmoid(-pi_logit)=p_occ)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

OUT_DIR = Path("output/xgb")
EPS = 1e-6


def make_dmatrix(X: pd.DataFrame, y: np.ndarray | None = None) -> xgb.DMatrix:
    # y is passed through as observed. ZABetaHurdle masks the zeros out of the Beta term,
    # and the panel has no exact 1s (observed max 0.60), so nothing needs clamping.
    return xgb.DMatrix(X, label=y, enable_categorical=True)


def params_from_predt(predt: pd.DataFrame) -> dict[str, np.ndarray]:
    """Map a ZABeta parameter DataFrame to the NN's (pi_logit, mu, phi) + p_occ."""
    alpha = predt["concentration1"].to_numpy()
    beta = predt["concentration0"].to_numpy()
    gate = np.clip(predt["gate"].to_numpy(), EPS, 1.0 - EPS)
    phi = alpha + beta
    mu = alpha / phi
    p_occ = 1.0 - gate
    pi_logit = np.log(gate / (1.0 - gate))
    return {"p_occ": p_occ, "mu": mu, "phi": phi, "pi_logit": pi_logit, "e_y": p_occ * mu}
