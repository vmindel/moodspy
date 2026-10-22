"""Motif comparison, merging, clustering and alignment.

A port of universalmotif's ``compare_motifs``, ``compare_motifs_lite``,
``merge_motifs``, ``merge_similar``, ``motif_tree`` and the alignment step
of ``view_motifs``. The numba kernels follow ``compare_motifs.cpp`` and
``compare_motifs_lite.cpp`` line by line where it changes results,
including padding rules, IC filters and score aggregation.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numba
import numpy as np
import pandas as pd

from .manipulate import UM_PSEUDOCOUNT, motif_nsites, ppm
from .motifs import Motif, as_motif_list, motif_labels

Method = Literal["PCC", "EUCL", "SW", "KL", "WEUCL", "ALLR", "BHAT", "HELL", "WPCC", "SEUCL", "MAN", "ALLR_LL"]
ScoreStrat = Literal["sum", "a.mean", "g.mean", "median", "wa.mean", "wg.mean", "fzt"]

_METHODS = {"EUCL": 1, "KL": 2, "HELL": 3, "SEUCL": 5, "MAN": 6, "PCC": 7, "SW": 8,
            "ALLR": 9, "BHAT": 10, "ALLR_LL": 11, "WEUCL": 12, "WPCC": 13}
_SIMILARITIES = {"PCC", "WPCC", "SW", "ALLR", "BHAT", "ALLR_LL"}
_STRATS = {"sum": 1, "a.mean": 2, "g.mean": 3, "median": 4, "wa.mean": 5, "wg.mean": 6, "fzt": 7}
# Metrics that get +0.01 on every matrix entry to avoid log(0) (universalmotif's klfix).
_PSEUDO_METHODS = {"KL", "ALLR", "ALLR_LL"}
_DBL_MAX = np.finfo(np.float64).max


# ====================================================================== column metrics
@numba.njit(nogil=True, cache=True, error_model="numpy")
def _col_score(method, a, b, bkg1, bkg2, n1, n2):
    r = a.shape[0]
    s = 0.0
    if method == 1 or method == 5 or method == 12:  # EUCL, SEUCL, WEUCL
        for j in range(r):
            d = a[j] - b[j]
            s += d * d * ((bkg1[j] + bkg2[j]) if method == 12 else 1.0)
        return s if method == 5 else np.sqrt(s)
    if method == 2:  # KL (symmetrised)
        for j in range(r):
            s += a[j] * np.log(a[j] / b[j]) + b[j] * np.log(b[j] / a[j])
        return 0.5 * s
    if method == 3:  # HELL
        for j in range(r):
            d = np.sqrt(a[j]) - np.sqrt(b[j])
            s += d * d
        return np.sqrt(s) / np.sqrt(2.0)
    if method == 6:  # MAN
        for j in range(r):
            s += abs(a[j] - b[j])
        return s
    if method == 10:  # BHAT
        for j in range(r):
            s += np.sqrt(a[j] * b[j])
        return s
    if method == 8:  # SW
        for j in range(r):
            d = a[j] - b[j]
            s += d * d
        return 2.0 - s
    if method == 7:  # PCC
        sab = 0.0
        sa = 0.0
        sb = 0.0
        saa = 0.0
        sbb = 0.0
        for j in range(r):
            sab += a[j] * b[j]
            sa += a[j]
            sb += b[j]
            saa += a[j] * a[j]
            sbb += b[j] * b[j]
        top = r * sab - sa * sb
        bot = np.sqrt((r * saa - sa * sa) * (r * sbb - sb * sb))
        return 1.0 if bot == 0 else top / bot
    if method == 13:  # WPCC
        m1 = 0.0
        m2 = 0.0
        for j in range(r):
            w = bkg1[j] + bkg2[j]
            m1 += w * a[j]
            m2 += w * b[j]
        v1 = 0.0
        v2 = 0.0
        cov = 0.0
        for j in range(r):
            w = bkg1[j] + bkg2[j]
            v1 += w * (a[j] - m1) ** 2
            v2 += w * (b[j] - m2) ** 2
            cov += w * (a[j] - m1) * (b[j] - m2)
        bot = np.sqrt(v1 * v2)
        return 1.0 if bot == 0 else cov / bot
    # ALLR (9), ALLR_LL (11)
    left = 0.0
    right = 0.0
    for j in range(r):
        left += b[j] * n2 * np.log(a[j] / bkg1[j])
        right += a[j] * n1 * np.log(b[j] / bkg2[j])
    s = (left + right) / (n1 + n2)
    if method == 11 and s < -2.0:
        s = -2.0
    return s


@numba.njit(nogil=True, cache=True, error_model="numpy")
def _combine(vals, w, n, strat):
    """universalmotif's calc_final_score() over the ``n`` aligned (non-gap) columns."""
    if strat == 1:
        return vals[:n].sum()
    if strat == 2:
        return vals[:n].sum() / n
    if strat == 3:  # g.mean: non-positive scores add nothing but still count in n
        s = 0.0
        for i in range(n):
            if vals[i] > 0:
                s += np.log(vals[i])
        return 0.0 if s == 0.0 else np.exp(s / n)
    if strat == 4:
        if n == 1:
            return vals[0]
        v = np.sort(vals[:n])
        return (v[n // 2 - 1] + v[n // 2]) / 2.0 if n % 2 == 0 else v[n // 2]
    if strat == 5 or strat == 6:
        tot = w[:n].sum()
        out = 0.0
        if strat == 5:
            for i in range(n):
                out += vals[i] * w[i] / tot
            return out
        chk = 0.0
        for i in range(n):
            x = np.log(vals[i]) if vals[i] > 0 else vals[i]
            chk += x
            out += x * w[i] / tot
        return 0.0 if chk == 0.0 else np.exp(out)
    # fzt
    s = 0.0
    for i in range(n):
        s += np.tanh(vals[i])
    return np.arctanh(s / n)


# ====================================================================== pairwise alignment
@numba.njit(nogil=True, cache=True)
def _equalize(m1, m2, ic1, ic2, overlap):
    """Pad the shorter motif with gap columns (-1) so overhangs down to ``overlap`` are tried."""
    n1 = m1.shape[0]
    n2 = m2.shape[0]
    if overlap < 1:
        o1 = int(overlap * n1)
        o2 = int(overlap * n2)
    else:
        o1 = int(overlap)
        o2 = int(overlap)
    add1 = 0 if o1 > n2 else n2 - o1
    add2 = 0 if o2 > n1 else n1 - o2
    if add1 == 0 or add2 == 0:
        return m1, m2, ic1, ic2
    r = m1.shape[1]
    if n2 > n1:
        p = np.full((2 * add1 + n1, r), -1.0)
        pic = np.zeros(2 * add1 + n1)
        p[add1 : add1 + n1] = m1
        pic[add1 : add1 + n1] = ic1
        return p, m2, pic, ic2
    p = np.full((2 * add2 + n2, r), -1.0)
    pic = np.zeros(2 * add2 + n2)
    p[add2 : add2 + n2] = m2
    pic[add2 : add2 + n2] = ic2
    return m1, p, ic1, pic


@numba.njit(nogil=True, cache=True, error_model="numpy")
def _best_offset(m1, m2, ic1, ic2, method, strat, overlap, minic, norm, posic,
                 bkg1, bkg2, n1, n2, similarity):
    """Best score over all offsets of ``m2`` against ``m1`` (no reverse complement).

    Returns ``(score, index)`` where ``index`` enumerates offsets as
    universalmotif does (the shorter padded motif slides along the longer).
    ``score`` is +-DBL_MAX when every offset failed the mean IC filter.
    """
    tlen = max(m1.shape[0], m2.shape[0])
    a, b, ia, ib = _equalize(m1, m2, ic1, ic2, overlap)
    minw = min(a.shape[0], b.shape[0])
    fori = 1 + a.shape[0] - minw
    forj = 1 + b.shape[0] - minw
    dmax = 1.7976931348623157e308
    vals = np.empty(minw)
    wts = np.empty(minw)
    best = 0.0
    best_i = -1
    counter = 0
    for i in range(fori):
        for j in range(forj):
            s1 = 0.0
            c1 = 0
            s2 = 0.0
            c2 = 0
            n = 0
            for k in range(minw):
                t1 = ia[k + i]
                t2 = ib[k + j]
                g1 = a[k + i, 0] >= 0
                g2 = b[k + j, 0] >= 0
                if posic > 0:
                    if t1 < posic:
                        t1 = -1.0
                        g1 = False
                    if t2 < posic:
                        t2 = -1.0
                        g2 = False
                if t1 >= 0:
                    s1 += t1
                    c1 += 1
                if t2 >= 0:
                    s2 += t2
                    c2 += 1
                if g1 and g2:
                    vals[n] = _col_score(method, a[k + i], b[k + j], bkg1, bkg2, n1, n2)
                    wts[n] = t1 + t2
                    n += 1
            lowic = s1 / c1 < minic or s2 / c2 < minic
            alignlen = n if norm else tlen
            if lowic:
                ans = -dmax if similarity else dmax
            else:
                ans = _combine(vals, wts, n, strat)
                if method == 1 or method == 2 or method == 12:
                    ans *= tlen / alignlen
                else:
                    ans *= alignlen / tlen
            if best_i < 0 or (similarity and best < ans) or (not similarity and ans < best):
                best = ans
                best_i = counter
            counter += 1
    return best, best_i


@numba.njit(nogil=True, cache=True, error_model="numpy")
def _compare_batch(flat, offs, ics, bkgs, nsites, idx1, idx2, method, strat, overlap,
                   rc, minic, norm, posic, similarity, out_score, out_rc, out_off):
    for p in range(idx1.shape[0]):
        i = idx1[p]
        j = idx2[p]
        m1 = flat[offs[i] : offs[i + 1]]
        m2 = flat[offs[j] : offs[j + 1]]
        ic1 = ics[offs[i] : offs[i + 1]]
        ic2 = ics[offs[j] : offs[j + 1]]
        s, o = _best_offset(m1, m2, ic1, ic2, method, strat, overlap, minic, norm, posic,
                            bkgs[i], bkgs[j], nsites[i], nsites[j], similarity)
        out_score[p] = s
        out_off[p] = o
        out_rc[p] = False
        if rc:
            m2r = m2[::-1, ::-1].copy()
            ic2r = ic2[::-1].copy()
            sr, orc = _best_offset(m1, m2r, ic1, ic2r, method, strat, overlap, minic, norm, posic,
                                   bkgs[i], bkgs[j], nsites[i], nsites[j], similarity)
            if (similarity and sr > s) or (not similarity and sr < s):
                out_score[p] = sr
                out_off[p] = orc
                out_rc[p] = True


# ====================================================================== preparation
@dataclass
class _Prepared:
    motifs: list[Motif]
    labels: list[str]
    raw: list[np.ndarray]  # (L, 4) matrices before the +0.01 fix
    flat: np.ndarray       # (sum L, 4) matrices used for scoring
    offs: np.ndarray
    ics: np.ndarray
    bkgs: np.ndarray
    nsites: np.ndarray


def _position_ic(mat: np.ndarray, bkg: np.ndarray, use_type: str, relative: bool) -> np.ndarray:
    """universalmotif's internal_posIC() on (L, 4) rows."""
    if use_type == "ICM":
        return mat.sum(axis=1)
    if relative:
        with np.errstate(divide="ignore", invalid="ignore"):
            h = mat * np.log2(mat / bkg)
        h[~np.isfinite(h)] = 0.0
        return np.maximum(h, 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.where(mat > 0, -mat * np.log2(mat), 0.0).sum(axis=1)
    return np.log2(mat.shape[1]) - ent


def _prepare(motifs: Any, method: str, use_type: str, relative_entropy: bool,
             pseudocount: float) -> _Prepared:
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}")
    if use_type not in ("PPM", "ICM"):
        raise ValueError("use_type must be 'PPM' or 'ICM'")
    if use_type == "ICM" and method in ("ALLR", "ALLR_LL"):
        raise ValueError("use_type='ICM' is not allowed with ALLR / ALLR_LL")
    mots = as_motif_list(motifs)
    raw, mats, ics, bkgs = [], [], [], []
    for m in mots:
        p = ppm(m, pseudocount)
        bkg = m.background.copy()
        if use_type == "ICM":
            from .manipulate import _ic_columns

            p = _ic_columns(p, bkg, relative_entropy)
        mat = np.ascontiguousarray(p.T)
        raw.append(mat)
        if method in _PSEUDO_METHODS:
            mat = mat + 0.01
            if (bkg == 0).any():
                bkg = bkg + 0.0025
        mats.append(mat)
        bkgs.append(bkg)
        ics.append(_position_ic(mat, bkg, use_type, relative_entropy))
    offs = np.zeros(len(mats) + 1, dtype=np.int64)
    np.cumsum([x.shape[0] for x in mats], out=offs[1:])
    return _Prepared(
        motifs=mots, labels=motif_labels(mots), raw=raw,
        flat=np.ascontiguousarray(np.concatenate(mats)), offs=offs,
        ics=np.concatenate(ics), bkgs=np.array(bkgs),
        nsites=np.array([motif_nsites(m) for m in mots]),
    )


def _check_strat(method: str, score_strat: str) -> None:
    if score_strat not in _STRATS:
        raise ValueError(f"score_strat must be one of {list(_STRATS)}")
    if score_strat in ("g.mean", "wg.mean") and method in ("ALLR", "ALLR_LL", "PCC"):
        raise ValueError("g.mean / wg.mean are not allowed for methods that can be negative "
                         "(ALLR, ALLR_LL, PCC)")


def _n_threads(n_jobs: int | None) -> int:
    if n_jobs is None:
        return os.cpu_count() or 1
    return max(1, int(n_jobs))


def _run_pairs(q: _Prepared, t: _Prepared | None, idx1, idx2, method, score_strat, min_overlap,
               try_rc, min_mean_ic, normalise_scores, min_position_ic, n_jobs):
    """Score pairs (q[idx1], t[idx2]); returns score, used-RC flag and offset index."""
    if min_mean_ic < 0 or min_position_ic < 0:
        raise ValueError("min_mean_ic and min_position_ic must be >= 0")
    if t is not None:
        # put both sets into one flat array so the kernel sees a single motif list
        nq = len(q.raw)
        flat = np.ascontiguousarray(np.concatenate([q.flat, t.flat]))
        offs = np.concatenate([q.offs, t.offs[1:] + q.offs[-1]])
        ics = np.concatenate([q.ics, t.ics])
        bkgs = np.concatenate([q.bkgs, t.bkgs])
        nsites = np.concatenate([q.nsites, t.nsites])
        idx2 = idx2 + nq
    else:
        flat, offs, ics, bkgs, nsites = q.flat, q.offs, q.ics, q.bkgs, q.nsites
    n = len(idx1)
    score = np.empty(n)
    rc = np.zeros(n, dtype=np.bool_)
    off = np.zeros(n, dtype=np.int64)
    overlap = 1.0 if min_overlap < 0 else float(min_overlap)
    args = (flat, offs, ics, bkgs, nsites)
    kw = (_METHODS[method], _STRATS[score_strat], overlap, bool(try_rc), float(min_mean_ic),
          bool(normalise_scores), float(min_position_ic), method in _SIMILARITIES)
    threads = min(_n_threads(n_jobs), max(1, n // 256))
    bounds = np.linspace(0, n, threads + 1).astype(np.int64)

    def work(c):
        a, b = bounds[c], bounds[c + 1]
        _compare_batch(*args, idx1[a:b], idx2[a:b], *kw, score[a:b], rc[a:b], off[a:b])

    if threads == 1:
        work(0)
    else:
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(work, range(threads)))
    return score, rc, off


# ====================================================================== compare_motifs
def compare_motifs(
    motifs: Any,
    targets: Any = None,
    *,
    method: Method = "PCC",
    use_type: Literal["PPM", "ICM"] = "PPM",
    try_rc: bool = True,
    min_overlap: float = 6,
    min_mean_ic: float = 0.25,
    min_position_ic: float = 0.0,
    relative_entropy: bool = False,
    normalise_scores: bool = False,
    score_strat: ScoreStrat = "a.mean",
    pseudocount: float = UM_PSEUDOCOUNT,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Similarity or distance between motifs, scored at their best alignment.

    A port of universalmotif's ``compare_motifs``. Every offset between two
    motifs that overlaps by at least ``min_overlap`` columns is scored
    column by column, the column scores are combined with ``score_strat``,
    and the best alignment (optionally including the reverse complement of
    the second motif) is reported.

    Parameters
    ----------
    motifs
        Motifs in any form [`match_motifs`][moodspy.match_motifs] accepts.
    targets
        If given, compare every motif in ``motifs`` with every motif in
        ``targets`` (rectangular result). Otherwise all against all.
    method
        Similarities (higher = more alike): ``PCC`` (Pearson correlation),
        ``WPCC`` (background-weighted PCC), ``SW`` (Sandelin-Wasserman),
        ``ALLR`` / ``ALLR_LL`` (average log-likelihood ratio), ``BHAT``
        (Bhattacharyya). Distances (lower = more alike): ``EUCL``,
        ``WEUCL``, ``SEUCL``, ``MAN``, ``HELL``, ``KL``.
    use_type
        Compare probability (``"PPM"``) or information content (``"ICM"``)
        matrices.
    try_rc
        Also align the reverse complement and keep the better score.
    min_overlap
        Minimum number of overlapping columns. A value below 1 is a
        fraction of each motif's length. Values above the motif lengths
        forbid overhangs.
    min_mean_ic
        An alignment is skipped when the mean information content of either
        motif's window (gap columns count as 0) is below this. Pairs with
        no passing alignment are NaN.
    min_position_ic
        Columns below this information content are left out of the score.
    relative_entropy
        Use background-relative information content for the IC filters.
    normalise_scores
        Multiply similarities (divide distances) by the fraction of the
        longer motif that is aligned.
    score_strat
        How column scores combine: ``"sum"``, ``"a.mean"``, ``"g.mean"``,
        ``"median"``, ``"wa.mean"``, ``"wg.mean"`` (weighted by the summed
        IC of the aligned columns) or ``"fzt"`` (Fisher z-transform).
    pseudocount
        Pseudocount for count matrices (universalmotif's default is 1).
        Probability matrices are used as they are. ``KL`` and ``ALLR``
        additionally add 0.01 to every entry, as universalmotif does.
    n_jobs
        Threads (default: all cores).

    Returns
    -------
    pandas.DataFrame
        Motifs x motifs (or motifs x targets), labelled with ``Motif.label``.

    Notes
    -----
    Scores reproduce universalmotif 1.20 on the same matrices; see the
    parity tests. P-values (universalmotif's ``db.scores``) are not ported;
    use [`compare_motifs_lite`][moodspy.compare_motifs_lite] for
    Tomtom-style p- and q-values.
    """
    _check_strat(method, score_strat)
    q = _prepare(motifs, method, use_type, relative_entropy, pseudocount)
    common = (method, score_strat, min_overlap, try_rc, min_mean_ic, normalise_scores,
              min_position_ic, n_jobs)
    if targets is None:
        n = len(q.raw)
        idx1, idx2 = np.triu_indices(n)
        score, _, _ = _run_pairs(q, None, idx1.astype(np.int64), idx2.astype(np.int64), *common)
        mat = np.empty((n, n))
        mat[idx1, idx2] = score
        mat[idx2, idx1] = score
        cols = q.labels
    else:
        t = _prepare(targets, method, use_type, relative_entropy, pseudocount)
        nq, nt = len(q.raw), len(t.raw)
        idx1 = np.repeat(np.arange(nq, dtype=np.int64), nt)
        idx2 = np.tile(np.arange(nt, dtype=np.int64), nq)
        score, _, _ = _run_pairs(q, t, idx1, idx2, *common)
        mat = score.reshape(nq, nt)
        cols = t.labels
    mat[np.abs(mat) == _DBL_MAX] = np.nan
    if np.isnan(mat).any():
        import warnings

        warnings.warn("some comparisons failed due to low motif IC (NaN in the result); "
                      "lower min_mean_ic or drop low-IC motifs (see average_ic)", stacklevel=2)
    return pd.DataFrame(mat, index=pd.Index(q.labels, name="motif"), columns=pd.Index(cols))


# ====================================================================== merging
def _place(shorter: np.ndarray, length: int, at: int) -> np.ndarray:
    out = np.full((length, shorter.shape[1]), -1.0)
    out[at : at + shorter.shape[0]] = shorter
    return out


def _frame_pair(m1, m2, offset, min_overlap):
    """Put two (L, 4) matrices into one frame, gap columns = -1, trimmed at the edges."""
    a, b, _, _ = _equalize(m1, m2, np.zeros(len(m1)), np.zeros(len(m2)),
                           1.0 if min_overlap < 0 else float(min_overlap))
    if len(a) > len(b):
        b = _place(b, len(a), offset)
    elif len(b) > len(a):
        a = _place(a, len(b), offset)
    empty = (a[:, 0] < 0) & (b[:, 0] < 0)
    keep = np.flatnonzero(~empty)
    if len(keep):
        a, b = a[keep[0] : keep[-1] + 1], b[keep[0] : keep[-1] + 1]
    return a, b


def merge_motifs(
    motifs: Any,
    *,
    method: Method = "ALLR",
    min_overlap: float = 6,
    min_mean_ic: float = 0.25,
    try_rc: bool = True,
    relative_entropy: bool = False,
    normalise_scores: bool = False,
    min_position_ic: float = 0.0,
    score_strat: ScoreStrat = "sum",
    name: str | None = None,
    pseudocount: float = UM_PSEUDOCOUNT,
) -> Motif:
    """Merge motifs into one by aligning and averaging their probability columns.

    A port of universalmotif's ``merge_motifs``. Motifs are added one at a
    time in the order given: each is aligned to the running merge (best
    offset and strand under ``method``) and averaged in with weight 1
    against the ``k`` motifs already merged. Columns covered by only one
    side are copied. The alignment parameters are those of
    [`compare_motifs`][moodspy.compare_motifs].

    Order matters. [`merge_similar`][moodspy.merge_similar] sorts each
    cluster by decreasing information content first, which usually gives the
    best alignments.

    Returns
    -------
    Motif
        A probability motif named ``"a/b/c"`` (or ``name``), with ``nsites``
        the largest of the inputs and the averaged background.
    """
    _check_strat(method, score_strat)
    mots = as_motif_list(motifs)
    if len(mots) == 1:
        return mots[0].copy()
    prep = _prepare(mots, method, "PPM", relative_entropy, pseudocount)
    pseudo = method in _PSEUDO_METHODS
    sim = method in _SIMILARITIES
    kw = dict(method=_METHODS[method], strat=_STRATS[score_strat],
              overlap=1.0 if min_overlap < 0 else float(min_overlap), minic=float(min_mean_ic),
              norm=bool(normalise_scores), posic=float(min_position_ic), similarity=sim)

    def fixed(mat):
        return mat + 0.01 if pseudo else mat

    def bkgfix(bkg):
        return bkg + 0.0025 if pseudo and (bkg == 0).any() else bkg

    merged = prep.raw[0]
    merged_bkg = mots[0].background.copy()
    nsites_merged = prep.nsites[0]
    for i in range(1, len(mots)):
        weight = i
        m2 = prep.raw[i]
        mp1, mp2 = fixed(merged), fixed(m2)
        b1, b2 = bkgfix(merged_bkg), bkgfix(mots[i].background.copy())
        ic1 = _position_ic(mp1, b1, "PPM", relative_entropy)
        ic2 = _position_ic(mp2, b2, "PPM", relative_entropy)
        n1 = prep.nsites[0] if i == 1 else nsites_merged
        s, off = _best_offset(mp1, mp2, ic1, ic2, kw["method"], kw["strat"], kw["overlap"],
                              kw["minic"], kw["norm"], kw["posic"], b1, b2, n1, prep.nsites[i], sim)
        if try_rc:
            sr, offr = _best_offset(mp1, mp2[::-1, ::-1].copy(), ic1, ic2[::-1].copy(),
                                    kw["method"], kw["strat"], kw["overlap"], kw["minic"],
                                    kw["norm"], kw["posic"], b1, b2, n1, prep.nsites[i], sim)
            if (sim and sr > s) or (not sim and sr < s):
                off, m2 = offr, m2[::-1, ::-1].copy()
        a, b = _frame_pair(merged, m2, off, min_overlap)
        out = []
        for ca, cb in zip(a, b):
            if ca[0] < 0 and cb[0] >= 0:
                out.append(cb)
            elif cb[0] < 0 and ca[0] >= 0:
                out.append(ca)
            elif ca[0] >= 0 and cb[0] >= 0:
                out.append((ca * weight + cb) / (weight + 1))
        merged = np.array(out)
        merged_bkg = (merged_bkg * weight + mots[i].background) / (weight + 1)
        nsites_merged = prep.nsites[0] + prep.nsites[1] if i == 1 else nsites_merged + prep.nsites[i]
    names = list(dict.fromkeys(m.name or m.label for m in mots))
    return Motif.from_probabilities(
        merged.T, name=name if name is not None else "/".join(names),
        nsites=float(max(prep.nsites)), background=None,
        meta={"merged_background": merged_bkg},
    )


def merge_similar(
    motifs: Any,
    threshold: float = 0.95,
    *,
    method: Method = "PCC",
    min_overlap: float = 6,
    min_mean_ic: float = 0.0,
    try_rc: bool = True,
    relative_entropy: bool = False,
    normalise_scores: bool = False,
    min_position_ic: float = 0.0,
    score_strat_compare: ScoreStrat = "a.mean",
    score_strat_merge: ScoreStrat = "sum",
    return_clusters: bool = False,
    pseudocount: float = UM_PSEUDOCOUNT,
    n_jobs: int | None = None,
) -> list[Motif] | list[list[Motif]]:
    """Cluster motifs and merge each cluster into one motif.

    A port of universalmotif's ``merge_similar``: all-vs-all
    [`compare_motifs`][moodspy.compare_motifs], complete-linkage
    hierarchical clustering, a cut at ``threshold`` and
    [`merge_motifs`][moodspy.merge_motifs] within each cluster (motifs
    sorted by decreasing information content). For similarity metrics
    ``threshold`` is a similarity (e.g. 0.95 PCC); for distances it is a
    distance. Clusters are numbered by first appearance, as R's ``cutree``
    does.

    Returns
    -------
    list
        Merged motifs, or the clusters themselves with ``return_clusters=True``.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if score_strat_compare == "sum":
        raise ValueError("score_strat_compare cannot be 'sum'")
    if method in ("ALLR", "ALLR_LL"):
        raise ValueError("ALLR / ALLR_LL scores cannot be turned into distances")
    mots = as_motif_list(motifs)
    if len(mots) == 1:
        return [[mots[0]]] if return_clusters else [mots[0].copy()]
    comp = compare_motifs(mots, method=method, min_overlap=min_overlap, min_mean_ic=min_mean_ic,
                          try_rc=try_rc, relative_entropy=relative_entropy,
                          normalise_scores=normalise_scores, min_position_ic=min_position_ic,
                          score_strat=score_strat_compare, pseudocount=pseudocount,
                          n_jobs=n_jobs).to_numpy()
    if np.isnan(comp).any():
        raise ValueError("NaN in the comparison matrix: lower min_mean_ic / min_position_ic "
                         "or drop the low-IC motifs")
    if method in ("PCC", "SW", "WPCC"):
        threshold = comp[0, 0] - threshold
        comp = comp[0, 0] - comp
    dist = squareform(comp, checks=False)
    groups = fcluster(linkage(dist, "complete"), t=threshold, criterion="distance")
    _, first = np.unique(groups, return_index=True)
    order = groups[np.sort(first)]
    clusters = [[mots[i] for i in np.flatnonzero(groups == g)] for g in order]
    if return_clusters:
        return clusters
    from .manipulate import motif_ic

    out = []
    for cl in clusters:
        ics = np.array([motif_ic(m, pseudocount).sum() for m in cl])
        cl = [cl[i] for i in np.argsort(-ics, kind="stable")]
        out.append(merge_motifs(cl, method=method, min_overlap=min_overlap, min_mean_ic=min_mean_ic,
                                try_rc=try_rc, relative_entropy=relative_entropy,
                                normalise_scores=normalise_scores, min_position_ic=min_position_ic,
                                score_strat=score_strat_merge, pseudocount=pseudocount))
    return out


# ====================================================================== alignment
@dataclass
class MotifAlignment:
    """Motifs placed on a shared column frame (output of [`align_motifs`][moodspy.align_motifs]).

    Attributes
    ----------
    motifs
        The motifs, reverse-complemented where that aligned better, in
        alignment order (anchor first).
    table
        One row per motif: ``offset`` (first frame column of the motif),
        ``strand`` (``"-"`` if reverse-complemented), ``length`` and
        ``score`` against the anchor (NaN for the anchor itself).
    width
        Number of frame columns.
    """

    motifs: list[Motif]
    table: pd.DataFrame
    width: int

    def padded(self, kind: Literal["ICM", "PPM"] = "ICM", pseudocount: float = UM_PSEUDOCOUNT) -> list[np.ndarray]:
        """``(4, width)`` matrices with zeros outside each motif."""
        from .manipulate import icm

        out = []
        for m, off in zip(self.motifs, self.table["offset"]):
            mat = icm(m, pseudocount) if kind == "ICM" else ppm(m, pseudocount)
            full = np.zeros((4, self.width))
            full[:, off : off + m.length] = mat
            out.append(full)
        return out


def align_motifs(
    motifs: Any,
    *,
    method: Method = "ALLR",
    try_rc: bool = True,
    min_overlap: float = 6,
    min_mean_ic: float = 0.25,
    relative_entropy: bool = False,
    normalise_scores: bool = False,
    min_position_ic: float = 0.0,
    score_strat: ScoreStrat = "sum",
    anchor: int | str | None = None,
    pseudocount: float = UM_PSEUDOCOUNT,
) -> MotifAlignment:
    """Align motifs to one anchor motif for plotting or inspection.

    The alignment step of universalmotif's ``view_motifs``: each motif is
    aligned to the anchor (by default the motif with the highest total
    information content) at its best offset and strand, then all are
    shifted onto one frame. Motifs whose every alignment fails the IC
    filters are placed at offset 0 on the ``+`` strand.

    Unlike ``view_motifs``, which aligns the information content matrices,
    this aligns probability matrices, the same input as
    [`merge_motifs`][moodspy.merge_motifs].
    """
    from .manipulate import motif_ic

    _check_strat(method, score_strat)
    mots = as_motif_list(motifs)
    labels = motif_labels(mots)
    if anchor is None:
        a = int(np.argmax([motif_ic(m, pseudocount).sum() for m in mots]))
    elif isinstance(anchor, str):
        a = labels.index(anchor)
    else:
        a = int(anchor)
    order = [a] + [i for i in range(len(mots)) if i != a]
    prep = _prepare([mots[i] for i in order], method, "PPM", relative_entropy, pseudocount)
    n = len(order)
    if n == 1:
        table = pd.DataFrame({"offset": [0], "strand": ["+"], "length": [mots[a].length],
                              "score": [np.nan]}, index=pd.Index([labels[a]], name="motif"))
        return MotifAlignment([mots[a].copy()], table, mots[a].length)
    idx1 = np.zeros(n - 1, dtype=np.int64)
    idx2 = np.arange(1, n, dtype=np.int64)
    score, rc, off = _run_pairs(prep, None, idx1, idx2, method, score_strat, min_overlap, try_rc,
                                min_mean_ic, normalise_scores, min_position_ic, 1)
    starts = [0]
    out = [prep.motifs[0].copy()]
    for k in range(n - 1):
        m = prep.motifs[k + 1]
        failed = abs(score[k]) == _DBL_MAX
        use_rc = bool(rc[k]) and not failed
        m2 = m.reverse_complement() if use_rc else m.copy()
        mat2 = prep.raw[k + 1][::-1, ::-1] if use_rc else prep.raw[k + 1]
        a_frame, b_frame = _frame_pair(prep.raw[0], mat2, 0 if failed else int(off[k]), min_overlap)
        # position of the motif relative to the anchor's first column
        first_a = int(np.flatnonzero(a_frame[:, 0] >= 0)[0])
        first_b = int(np.flatnonzero(b_frame[:, 0] >= 0)[0])
        starts.append(first_b - first_a)
        out.append(m2)
        if failed:
            score[k] = np.nan
    starts = np.array(starts)
    shift = -starts.min()
    starts = starts + shift
    lengths = np.array([m.length for m in out])
    table = pd.DataFrame({
        "offset": starts, "strand": ["+"] + ["-" if (rc[k] and not np.isnan(score[k])) else "+" for k in range(n - 1)],
        "length": lengths, "score": np.concatenate([[np.nan], score]),
    }, index=pd.Index([labels[i] for i in order], name="motif"))
    return MotifAlignment(out, table, int((starts + lengths).max()))


# ====================================================================== tree / clustering
@dataclass
class MotifTree:
    """Hierarchical clustering of motifs (output of [`motif_tree`][moodspy.motif_tree]).

    Attributes
    ----------
    linkage
        SciPy linkage matrix (pass to ``scipy.cluster.hierarchy`` functions).
    distances
        The motifs x motifs distance matrix that was clustered.
    motifs
        The input motifs.
    """

    linkage: np.ndarray
    distances: pd.DataFrame
    motifs: list[Motif]

    @property
    def labels(self) -> list[str]:
        return list(self.distances.index)

    def clusters(self, threshold: float | None = None, n_clusters: int | None = None) -> pd.Series:
        """Cut the tree at a distance ``threshold`` or into ``n_clusters`` groups.

        Clusters are numbered 1, 2, ... by first appearance in the input.
        """
        from scipy.cluster.hierarchy import fcluster

        if (threshold is None) == (n_clusters is None):
            raise ValueError("give exactly one of threshold or n_clusters")
        if threshold is not None:
            g = fcluster(self.linkage, t=threshold, criterion="distance")
        else:
            g = fcluster(self.linkage, t=n_clusters, criterion="maxclust")
        _, first = np.unique(g, return_index=True)
        renum = {old: new for new, old in enumerate(g[np.sort(first)], start=1)}
        return pd.Series([renum[x] for x in g], index=pd.Index(self.labels, name="motif"), name="cluster")

    def leaves(self) -> list[str]:
        """Motif labels in dendrogram order."""
        from scipy.cluster.hierarchy import leaves_list

        return [self.labels[i] for i in leaves_list(self.linkage)]

    def plot(self, **kwargs):
        """Dendrogram with logos at the tips; see [`plot_motif_tree`][moodspy.plotting.plot_motif_tree]."""
        from .plotting import plot_motif_tree

        return plot_motif_tree(self, **kwargs)


def motif_tree(
    motifs: Any,
    *,
    method: Method = "EUCL",
    linkage: str = "complete",
    use_type: Literal["PPM", "ICM"] = "PPM",
    min_overlap: float = 6,
    min_position_ic: float = 0.0,
    try_rc: bool = True,
    min_mean_ic: float = 0.0,
    relative_entropy: bool = False,
    score_strat: ScoreStrat = "a.mean",
    pseudocount: float = UM_PSEUDOCOUNT,
    n_jobs: int | None = None,
) -> MotifTree:
    """Cluster motifs hierarchically from their pairwise distances.

    universalmotif's ``motif_tree`` builds the tree with R's ``hclust``
    (complete linkage) on [`compare_motifs`][moodspy.compare_motifs]
    distances; this returns the SciPy equivalent. Similarity metrics are
    turned into distances as ``self-similarity - similarity`` (e.g.
    ``1 - PCC``), which universalmotif does not allow for ``motif_tree``
    but does in ``merge_similar``. ``ALLR`` cannot be used.
    """
    from scipy.cluster.hierarchy import linkage as _linkage
    from scipy.spatial.distance import squareform

    if method in ("ALLR", "ALLR_LL"):
        raise ValueError("ALLR / ALLR_LL scores cannot be turned into distances")
    mots = as_motif_list(motifs)
    comp = compare_motifs(mots, method=method, use_type=use_type, min_overlap=min_overlap,
                          min_position_ic=min_position_ic, try_rc=try_rc, min_mean_ic=min_mean_ic,
                          relative_entropy=relative_entropy, score_strat=score_strat,
                          pseudocount=pseudocount, n_jobs=n_jobs)
    if comp.isna().to_numpy().any():
        raise ValueError("NaN in the comparison matrix: lower min_mean_ic / min_position_ic")
    d = comp.to_numpy()
    if method in _SIMILARITIES:
        d = np.diag(d)[:, None] - d
        d = (d + d.T) / 2
    d = np.maximum(d, 0.0)
    np.fill_diagonal(d, 0.0)
    dist = pd.DataFrame(d, index=comp.index, columns=comp.columns)
    return MotifTree(_linkage(squareform(d, checks=False), linkage), dist, mots)


def _average(x: np.ndarray, how: str) -> float:
    x = np.asarray(x, dtype=np.float64)
    if how not in ("a.mean", "g.mean", "median", "fzt"):
        raise ValueError("average must be 'a.mean', 'g.mean', 'median' or 'fzt'")
    return float(_combine(x, x, len(x), _STRATS[how]))


# ====================================================================== compare_motifs_lite
_PCC_BINS = 51


def _pcc_cols(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """PCC of every column of ``q`` (n, 4) with every column of ``t`` (m, 4), centred on 0.25."""
    qd = q - 0.25
    td = t - 0.25
    sqq = (qd * qd).sum(axis=1)
    stt = (td * td).sum(axis=1)
    num = qd @ td.T
    den = np.sqrt(np.outer(sqq, stt))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[(sqq < 1e-12)[:, None] | (stt < 1e-12)[None, :]] = 0.0
    return out


@numba.njit(nogil=True, cache=True)
def _lite_align(qf, qoff, tf, toff, qi, ti, min_overlap, rc, out_score, out_L, out_off,
                out_strand, out_qstart, out_ntested):
    for p in range(qi.shape[0]):
        q = qf[qoff[qi[p]] : qoff[qi[p] + 1]]
        t = tf[toff[ti[p]] : toff[ti[p] + 1]]
        wq = q.shape[0]
        wt = t.shape[0]
        mo = min(min_overlap, wq, wt)
        best = -np.inf
        nt = 0
        for orient in range(2 if rc else 1):
            qq = q if orient == 0 else q[::-1, ::-1]
            for d in range(mo - wt, wq - mo + 1):
                t_start = 0 if d >= 0 else -d
                t_end = min(wq - d, wt)
                L = t_end - t_start
                if L < mo:
                    continue
                q_start = d + t_start
                s = 0.0
                for j in range(L):
                    sqq = 0.0
                    stt = 0.0
                    sqt = 0.0
                    for b in range(4):
                        x = qq[q_start + j, b] - 0.25
                        y = t[t_start + j, b] - 0.25
                        sqq += x * x
                        stt += y * y
                        sqt += x * y
                    if sqq >= 1e-12 and stt >= 1e-12:
                        s += sqt / np.sqrt(sqq * stt)
                nt += 1
                if s > best or nt == 1:
                    best = s
                    out_L[p] = L
                    out_off[p] = d
                    out_strand[p] = orient
                    out_qstart[p] = q_start
        out_score[p] = best
        out_ntested[p] = nt


def _bin_pcc(v: np.ndarray) -> np.ndarray:
    t = np.clip((v + 1.0) / 2.0, 0.0, 1.0)
    return np.clip(np.floor(t * (_PCC_BINS - 1) + 0.5).astype(np.int64), 0, _PCC_BINS - 1)


def _enum_cols(bkg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 56 columns of the K=5 simplex grid with Dirichlet-multinomial weights (alpha = 4 * bkg)."""
    from scipy.special import gammaln

    K = 5
    alpha = np.maximum(4.0 * bkg, 0.1)
    cols, lw = [], []
    for a in range(K + 1):
        for b in range(K + 1 - a):
            for c in range(K + 1 - a - b):
                cnt = np.array([a, b, c, K - a - b - c], dtype=np.float64)
                cols.append(cnt / K)
                w = gammaln(K + 1.0) - gammaln(cnt + 1).sum() + (gammaln(cnt + alpha) - gammaln(alpha)).sum()
                w -= gammaln(K + alpha.sum()) - gammaln(alpha.sum())
                lw.append(w)
    lw = np.array(lw)
    w = np.exp(lw - lw.max())
    return np.array(cols), w / w.sum()


def _lite_pvalues(q_mat, strand, qstart, L, score, ntested, null_cols, null_w):
    """Bonferroni-corrected p-values of one query's best alignments (Tomtom-style null)."""
    out = np.ones(len(score))
    cache = {}
    for orient in (0, 1):
        sel = np.flatnonzero((strand == orient) & (L > 0) & (ntested > 0))
        if not len(sel):
            continue
        qq = q_mat if orient == 0 else q_mat[::-1, ::-1]
        bins = _bin_pcc(_pcc_cols(qq, null_cols))
        pmfs = np.zeros((len(qq), _PCC_BINS))
        for i in range(len(qq)):
            pmfs[i] = np.bincount(bins[i], weights=null_w, minlength=_PCC_BINS)
        for k in sel:
            key = (orient, int(qstart[k]), int(L[k]))
            if key not in cache:
                pmf = pmfs[qstart[k]]
                for c in range(qstart[k] + 1, qstart[k] + L[k]):
                    pmf = np.convolve(pmf, pmfs[c])
                cache[key] = np.cumsum(pmf[::-1])[::-1]  # upper tail
            tail = cache[key]
            per = np.clip((score[k] / L[k] + 1.0) / 2.0, 0.0, 1.0)
            b = int(np.clip(np.floor(per * (_PCC_BINS - 1) * L[k] + 0.5), 0, L[k] * (_PCC_BINS - 1)))
            p = float(np.clip(tail[b], 0.0, 1.0)) if b < len(tail) else 0.0
            out[k] = min(1.0, p * ntested[k])
    return out


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment, as R's ``p.adjust(method = "BH")``."""
    n = len(p)
    order = np.argsort(p)[::-1]
    adj = np.minimum.accumulate(p[order] * n / np.arange(n, 0, -1))
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def _iupac_range(mat: np.ndarray, start: int, length: int) -> str:
    """compare_motifs_lite's consensus: smallest letter set reaching 0.75 probability."""
    codes = {1: "A", 2: "C", 4: "G", 8: "T", 3: "M", 5: "R", 9: "W", 6: "S", 10: "Y", 12: "K",
             7: "V", 11: "H", 13: "D", 14: "B"}
    out = []
    for col in mat[start : start + length]:
        order = sorted(range(4), key=lambda i: -col[i])  # stable: ties keep A < C < G < T
        cum, mask = 0.0, 0
        for i in order:
            cum += col[i]
            mask |= 1 << i
            if cum >= 0.75:
                break
        out.append(codes.get(mask, "N"))
    return "".join(out)


def compare_motifs_lite(
    motifs: Any,
    targets: Any = None,
    *,
    qvalue: float = 0.1,
    min_overlap: int = 5,
    rc: bool = True,
    null: Literal["empirical", "parametric"] = "empirical",
    background: Any = None,
    matrix_out: Literal["score", "pvalue", "qvalue"] = "score",
    pseudocount: float = UM_PSEUDOCOUNT,
) -> pd.DataFrame:
    """Fast PCC motif comparison with Tomtom-style p- and q-values.

    A port of universalmotif's ``compare_motifs_lite``. Each pair is aligned
    at every offset with at least ``min_overlap`` columns (and on the
    reverse complement with ``rc``); the alignment score is the sum of
    column Pearson correlations (centred on 0.25), reported as the mean
    over the overlap. Its p-value comes from the distribution of that sum
    under a null built per query column, Bonferroni-corrected for the
    number of alignments tried; q-values are Benjamini-Hochberg over all
    targets of a query.

    Parameters
    ----------
    motifs
        Query motifs.
    targets
        Database to search. ``None`` compares the motifs all against all and
        returns a square matrix of ``matrix_out``. With targets, a long
        table of matches is returned (universalmotif's ``compare.to`` mode,
        with the database being ``targets``).
    qvalue
        Keep matches with q-value at most this (long format only).
    min_overlap
        Minimum number of overlapping columns.
    rc
        Also align the reverse complement of each query.
    null
        ``"empirical"``: column PCCs of the query against every column of
        the target database. ``"parametric"``: a Dirichlet-multinomial
        weighting of the 56 columns of a 5-count simplex grid, with
        concentration ``4 * background``; better for small databases.
    background
        A, C, G, T frequencies for the parametric null (default uniform).
    matrix_out
        In all-vs-all mode: ``"score"`` (mean PCC), ``"pvalue"`` or ``"qvalue"``.
    pseudocount
        Pseudocount for count matrices; probability matrices are used as
        they are.

    Returns
    -------
    pandas.DataFrame
        All-vs-all: query x target matrix. Otherwise one row per match with
        ``query``, ``query_i``, ``target``, ``target_i`` (0-based),
        ``offset`` (query column, in the query's aligned orientation, that
        target column 0 sits on; negative when the target overhangs on the
        left), ``strand``, ``overlap``, ``score``, ``query_consensus``,
        ``target_consensus``, ``pvalue`` and ``qvalue``, sorted by q-value.

    Notes
    -----
    Mapping from R: ``compare_motifs_lite(c(q, db), compare.to = 1)`` is
    ``compare_motifs_lite([q], targets=[q] + db)``, because R searches the
    query against the whole input including itself.
    """
    from .motifs import check_background

    if not 0 < qvalue <= 1:
        raise ValueError("qvalue must be in (0, 1]")
    if min_overlap < 1:
        raise ValueError("min_overlap must be a positive integer")
    if null not in ("empirical", "parametric"):
        raise ValueError("null must be 'empirical' or 'parametric'")
    bkg = np.full(4, 0.25) if background is None else check_background(background)
    qm = as_motif_list(motifs)
    matrix_mode = targets is None
    tm = qm if matrix_mode else as_motif_list(targets)
    if matrix_mode and len(qm) < 2:
        raise ValueError("need at least 2 motifs for all-vs-all mode")
    qmats = [np.ascontiguousarray(ppm(m, pseudocount).T) for m in qm]
    tmats = qmats if matrix_mode else [np.ascontiguousarray(ppm(m, pseudocount).T) for m in tm]
    qlab = motif_labels(qm)
    tlab = qlab if matrix_mode else motif_labels(tm)

    def flat(mats):
        offs = np.zeros(len(mats) + 1, dtype=np.int64)
        np.cumsum([len(x) for x in mats], out=offs[1:])
        return np.ascontiguousarray(np.concatenate(mats)), offs

    qf, qoff = flat(qmats)
    tf, toff = flat(tmats)
    nq, nt = len(qmats), len(tmats)
    qi = np.repeat(np.arange(nq, dtype=np.int64), nt)
    ti = np.tile(np.arange(nt, dtype=np.int64), nq)
    n = len(qi)
    score = np.empty(n)
    L = np.zeros(n, dtype=np.int64)
    off = np.zeros(n, dtype=np.int64)
    strand = np.zeros(n, dtype=np.int64)
    qstart = np.zeros(n, dtype=np.int64)
    ntested = np.zeros(n, dtype=np.int64)
    _lite_align(qf, qoff, tf, toff, qi, ti, int(min_overlap), bool(rc), score, L, off, strand,
                qstart, ntested)

    need_p = not matrix_mode or matrix_out != "score"
    pvals = np.full(n, np.nan)
    qvals = np.full(n, np.nan)
    if need_p:
        if null == "empirical":
            null_cols, null_w = tf, np.full(len(tf), 1.0 / len(tf))
        else:
            null_cols, null_w = _enum_cols(bkg)
        for k in range(nq):
            sl = slice(k * nt, (k + 1) * nt)
            pvals[sl] = _lite_pvalues(qmats[k], strand[sl], qstart[sl], L[sl], score[sl],
                                      ntested[sl], null_cols, null_w)
            qvals[sl] = _bh(pvals[sl])
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_pcc = np.where(L > 0, score / np.maximum(L, 1), np.nan)

    if matrix_mode:
        val = {"score": mean_pcc, "pvalue": pvals, "qvalue": qvals}[matrix_out]
        return pd.DataFrame(val.reshape(nq, nt), index=pd.Index(qlab, name="motif"), columns=pd.Index(tlab))

    keep = np.flatnonzero((qvals <= qvalue) & (L > 0))
    rows = []
    for k in keep:
        wq = len(qmats[qi[k]])
        q_orig = qstart[k] if strand[k] == 0 else wq - 1 - (qstart[k] + L[k] - 1)
        rows.append({
            "query": qlab[qi[k]], "query_i": int(qi[k]), "target": tlab[ti[k]], "target_i": int(ti[k]),
            "offset": int(off[k]), "strand": "-" if strand[k] == 1 else "+", "overlap": int(L[k]),
            "score": mean_pcc[k],
            "query_consensus": _iupac_range(qmats[qi[k]], int(q_orig), int(L[k])),
            "target_consensus": _iupac_range(tmats[ti[k]], max(0, -int(off[k])), int(L[k])),
            "pvalue": pvals[k], "qvalue": qvals[k],
        })
    cols = ["query", "query_i", "target", "target_i", "offset", "strand", "overlap", "score",
            "query_consensus", "target_consensus", "pvalue", "qvalue"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = df.sort_values(["qvalue", "pvalue", "target_i"], kind="mergesort", ignore_index=True)
    return df
