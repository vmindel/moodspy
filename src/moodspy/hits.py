"""Post-processing of motif hit tables."""

from __future__ import annotations

from typing import Literal

import numba
import numpy as np
import pandas as pd


@numba.njit(nogil=True, cache=True)
def _greedy_keep(group, start, end, reverse, order_by_score, order_by_pos, rank, opposite_only, min_overlap):
    """Greedy non-maximum suppression within groups.

    Hits are visited best score first. A hit is dropped when it overlaps an
    already kept hit of the same group by at least ``min_overlap`` bases
    (and, with ``opposite_only``, lies on the other strand). Neighbours are
    found by walking outwards in position order, which stays cheap because
    every hit in a group has the same motif, so overlapping hits are close.
    """
    n = start.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    max_len = 0
    for i in range(n):
        max_len = max(max_len, end[i] - start[i])
    for t in range(n):
        i = order_by_score[t]
        r = rank[i]
        ok = True
        # left neighbours (smaller starts)
        q = r - 1
        while ok and q >= 0:
            j = order_by_pos[q]
            if group[j] != group[i] or start[i] - start[j] >= max_len:
                break
            if keep[j] and (not opposite_only or reverse[j] != reverse[i]):
                if min(end[i], end[j]) - max(start[i], start[j]) >= min_overlap:
                    ok = False
            q -= 1
        # right neighbours (larger or equal starts)
        q = r + 1
        while ok and q < n:
            j = order_by_pos[q]
            if group[j] != group[i] or start[j] - start[i] >= max_len:
                break
            if keep[j] and (not opposite_only or reverse[j] != reverse[i]):
                if min(end[i], end[j]) - max(start[i], start[j]) >= min_overlap:
                    ok = False
            q += 1
        keep[i] = ok
    return keep


def remove_overlaps(
    hits: pd.DataFrame,
    strands: Literal["opposite", "any"] = "opposite",
    min_overlap: int = 1,
) -> pd.DataFrame:
    """Keep the best-scoring hit where hits of the same motif overlap.

    Palindromic and near-palindromic motifs (e.g. AP-1 ``TGASTCA``, or GRE
    and other inverted repeats) match one site on both strands, usually
    offset by a base or two. This collapses such pairs into one hit.

    Parameters
    ----------
    hits
        Output of ``match_motifs(..., out="positions")`` or
        [`Scanner.scan`][moodspy.Scanner.scan]: needs ``seq``, ``motif``,
        ``start``, ``end``, ``strand`` and ``score`` columns.
    strands
        ``"opposite"`` (default) only removes overlaps between a ``+`` and a
        ``-`` hit, i.e. the same site seen on both strands. ``"any"`` also
        removes overlapping hits on the same strand, so no two kept hits of a
        motif overlap (classic non-maximum suppression).
    min_overlap
        Minimum number of shared bases for two hits to count as overlapping.

    Returns
    -------
    pandas.DataFrame
        The kept rows of ``hits``, in their original order and with their
        original index.

    Notes
    -----
    The selection is greedy: hits are visited from the highest score down,
    and a hit is kept unless it overlaps one already kept. Ties are broken by
    position, then ``+`` before ``-``, so the result is deterministic. Hits
    of different motifs never suppress each other.
    """
    if strands not in ("opposite", "any"):
        raise ValueError("strands must be 'opposite' or 'any'")
    if min_overlap < 1:
        raise ValueError("min_overlap must be at least 1")
    missing = {"seq", "motif", "start", "end", "strand", "score"} - set(hits.columns)
    if missing:
        raise ValueError(f"hits is missing columns {sorted(missing)}")
    if len(hits) == 0:
        return hits.copy()

    group = pd.MultiIndex.from_arrays([hits["seq"], hits["motif"]]).codes
    group = np.ravel_multi_index(tuple(np.asarray(c, dtype=np.int64) for c in group),
                                 tuple(int(c.max()) + 1 for c in group))
    start = hits["start"].to_numpy(dtype=np.int64)
    end = hits["end"].to_numpy(dtype=np.int64)
    reverse = (hits["strand"].astype(str).to_numpy() == "-")
    score = hits["score"].to_numpy(dtype=np.float64)

    order_by_pos = np.lexsort((reverse, start, group))
    rank = np.empty_like(order_by_pos)
    rank[order_by_pos] = np.arange(len(order_by_pos))
    order_by_score = np.lexsort((reverse, start, -score))
    keep = _greedy_keep(group, start, end, reverse, order_by_score, order_by_pos, rank,
                        strands == "opposite", int(min_overlap))
    return hits[keep]
