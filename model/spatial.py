"""Spatial-autocorrelation statistics on the county queen graph.

Global Moran's I with a permutation null, over the row-standardized queen adjacency,
plus the exclusive (Pfeifer-Deutsch) lag operators that `model/starima.py` selects over
and `model/diagnostics.py` measures residual space-time autocorrelation on.
"""
from __future__ import annotations

import numpy as np


def undirected_edges(edge_index) -> np.ndarray:
    """Unique undirected edges (i<j) from a possibly-directed edge_index [2, E]."""
    ei = np.asarray(edge_index)
    src, dst = ei[0], ei[1]
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]            # [E_u, 2]


def row_norm_W(edge_index, n_nodes: int):
    """Row-standardized queen adjacency W and the binary adjacency A."""
    pairs = undirected_edges(edge_index)
    A = np.zeros((n_nodes, n_nodes))
    A[pairs[:, 0], pairs[:, 1]] = 1.0
    A = A + A.T
    deg = A.sum(1)
    W = A / np.maximum(deg[:, None], 1.0)
    return W, A


def exclusive_orders(edge_index, n: int, max_order: int, sparse: bool = False):
    """Row-standardised exclusive spatial lag operators W_0..W_L (Pfeifer-Deutsch).

    Order l holds counties at graph distance *exactly* l, so the orders partition the
    neighbourhood instead of nesting it and a retained order-2 term cannot be an echo of
    order 1. W_0 is the identity (the county's own history).
    """
    ei = np.asarray(edge_index)
    A = np.zeros((n, n), dtype=bool)
    A[ei[0], ei[1]] = True
    A |= A.T
    np.fill_diagonal(A, False)

    W = [np.eye(n, dtype=np.float64)]
    prev = np.eye(n, dtype=bool)               # reachable within l-1 steps
    reach = prev | A                           # reachable within l steps
    for _ in range(max_order):
        excl = reach & ~prev
        rs = excl.sum(axis=1, keepdims=True).astype(np.float64)
        W.append(np.where(rs > 0, excl / np.where(rs > 0, rs, 1.0), 0.0))
        prev = reach
        reach = prev | ((prev.astype(np.float32) @ A.astype(np.float32)) > 0)
    if sparse:
        from scipy.sparse import csr_matrix
        return [csr_matrix(w) for w in W]
    return W


def moran_i(x: np.ndarray, W: np.ndarray) -> float:
    """Global Moran's I of vector x under spatial-weight matrix W."""
    z = x - x.mean()
    den = float((z * z).sum())
    if den == 0.0:
        return 0.0
    num = float(z @ (W @ z))
    s0 = float(W.sum())
    n = len(x)
    return (n / s0) * (num / den)


def morans_permutation(x: np.ndarray, W: np.ndarray, n_perm: int = 999, seed: int = 0):
    """Observed Moran's I, a permutation p-value, and the null distribution.

    Two-sided p-value relative to the permutation mean (analytic null E[I] = -1/(n-1))."""
    obs = moran_i(x, W)
    rng = np.random.default_rng(seed)
    perms = np.array([moran_i(rng.permutation(x), W) for _ in range(n_perm)])
    center = perms.mean()
    p = (1 + int((np.abs(perms - center) >= abs(obs - center)).sum())) / (1 + n_perm)
    return obs, float(p), perms
