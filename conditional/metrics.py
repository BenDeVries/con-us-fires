"""One reporting/selection score for saved hurdle parameters."""
import numpy as np
from scipy.special import betaln


def row_nll(y, p, mu, phi):
    y, p, mu, phi = (np.asarray(v, dtype='float64') for v in (y, p, mu, phi))
    if not all(np.isfinite(v).all() for v in (y, p, mu, phi)):
        raise FloatingPointError('Nonfinite response or parameter')
    if not (((y >= 0) & (y < 1)).all() and ((p > 0) & (p < 1)).all()
            and ((mu > 0) & (mu < 1)).all() and (phi > 0).all()):
        raise FloatingPointError('Invalid hurdle distribution')
    pos = y > 0
    score = -np.log1p(-p)
    a, b = (mu * phi)[pos], ((1 - mu) * phi)[pos]
    score[pos] = -np.log(p[pos]) - (a - 1) * np.log(y[pos]) - (b - 1) * np.log1p(-y[pos]) + betaln(a, b)
    return score
