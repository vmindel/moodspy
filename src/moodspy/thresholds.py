"""Score thresholds from p-values.

A port of MOODS' ``threshold_from_p_with_precision``: the score matrix is
rounded to an integer grid (``precision`` steps per score unit), the exact
null distribution of the rounded score is computed by dynamic programming,
and the smallest grid score whose upper tail probability is <= ``p`` is
returned. With the default precision this reproduces MOODS / motifmatchr
thresholds exactly.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numba
import numpy as np

from .motifs import check_background

DEFAULT_PRECISION = 2000.0


@numba.njit(nogil=True, cache=True)
def _min_delta(pssm):
    best = np.inf
    for i in range(pssm.shape[0]):
        mx = -np.inf
        second = -np.inf
        for j in range(pssm.shape[1]):
            v = pssm[i, j]
            if v > mx:
                second = mx
                mx = v
            elif v < mx and v > second:
                second = v
        best = min(best, mx - second)
    return best


@numba.njit(nogil=True, cache=True)
def _threshold_kernel(pssm, bg, p, precision):
    """``pssm`` is ``(length, 4)``; returns the score threshold."""
    n = pssm.shape[0]
    a = pssm.shape[1]
    mat = np.empty((n, a), dtype=np.int64)
    for i in range(n):
        for j in range(a):
            if pssm[i, j] > 0.0:
                mat[i, j] = np.int64(precision * pssm[i, j] + 0.5)
            else:
                mat[i, j] = np.int64(precision * pssm[i, j] - 0.5)

    colmax = np.empty(n, dtype=np.int64)
    min_v = mat[0, 0]
    max_t = 0
    for i in range(n):
        mx = mat[i, 0]
        for j in range(a):
            mx = max(mx, mat[i, j])
            min_v = min(min_v, mat[i, j])
        colmax[i] = mx
        max_t += mx

    r_max = max_t - n * min_v
    table0 = np.zeros(r_max + 1)
    table1 = np.zeros(r_max + 1)
    for j in range(a):
        table0[mat[0, j] - min_v] += bg[j]
    reach = colmax[0] - min_v
    for i in range(1, n):
        for j in range(a):
            s = mat[i, j] - min_v
            # table0 is zero above `reach`, so this matches MOODS' full loop.
            for r in range(s, s + reach + 1):
                table1[r] += bg[j] * table0[r - s]
        reach += colmax[i] - min_v
        for r in range(reach + 1):
            table0[r] = table1[r]
            table1[r] = 0.0

    total = table0[r_max]
    if total > p:
        best = 0.0
        for i in range(n):
            best += pssm[i].max()
        return best - _min_delta(pssm) / 2.0
    for r in range(r_max - 1, -1, -1):
        total += table0[r]
        if total > p:
            return (r + n * min_v + 1) / precision
    worst = 0.0
    for i in range(n):
        worst += pssm[i].min()
    return worst - 1.0


def threshold_from_p(
    pwm: Any,
    p: float,
    background: Any = None,
    precision: float = DEFAULT_PRECISION,
) -> float:
    """Score threshold of a log-odds matrix for a p-value cutoff.

    Parameters
    ----------
    pwm
        ``(4, length)`` log-odds score matrix (rows A, C, G, T).
    p
        Upper-tail probability under the background model, in (0, 1].
        ``p=1`` disables the threshold (it is set 1 below the matrix's lowest
        possible score, as in MOODS), so every placement is a hit.
    background
        Nucleotide frequencies the p-value refers to (default uniform).
    precision
        Grid steps per score unit used for the dynamic programme.
    """
    mat = np.ascontiguousarray(np.asarray(pwm, dtype=np.float64).T)
    if mat.ndim != 2 or mat.shape[1] != 4:
        raise ValueError("pwm must be a (4, length) matrix")
    bg = np.full(4, 0.25) if background is None else check_background(background)
    _check_p(p)
    return float(_threshold(mat, bg, float(p), float(precision)))


def thresholds_from_p(
    pwms: list[np.ndarray],
    p: float,
    background: Any = None,
    precision: float = DEFAULT_PRECISION,
    n_jobs: int | None = 1,
) -> np.ndarray:
    """Vectorised [`threshold_from_p`][moodspy.threshold_from_p] over many matrices (multithreaded)."""
    bg = np.full(4, 0.25) if background is None else check_background(background)
    _check_p(p)
    mats = [np.ascontiguousarray(np.asarray(m, dtype=np.float64).T) for m in pwms]

    def one(m):
        return _threshold(m, bg, float(p), float(precision))

    if n_jobs == 1 or len(mats) < 2:
        return np.array([one(m) for m in mats], dtype=np.float64)
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        return np.array(list(pool.map(one, mats)), dtype=np.float64)


def _threshold(mat: np.ndarray, bg: np.ndarray, p: float, precision: float) -> float:
    if p >= 1.0:
        # No cutoff. The DP would compare a total mass of ~1 against p and could
        # drop the worst word to rounding error, so handle it exactly.
        return float(mat.min(axis=1).sum()) - 1.0
    return _threshold_kernel(mat, bg, p, precision)


def _check_p(p: float) -> None:
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p-value cutoff must be in (0, 1], got {p}")
