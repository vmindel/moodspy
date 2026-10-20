"""Numba kernels for multi-matrix scanning.

The algorithm is MOODS' lookahead filtration scanner. Every matrix gets a
"window" of ``w`` consecutive, most informative columns. For each of the
``4**w`` possible window contents we precompute which matrices could still
reach their threshold (window score + best possible score of the remaining
columns). Scanning then only needs one table lookup per sequence position;
the few candidates are verified column by column, most informative first,
abandoning as soon as the threshold is out of reach.

Units of work are ``(seq_index, scan_start, report_end, seq_end)`` rows in
absolute coordinates of the concatenated, encoded sequences. A unit reports
hits whose start lies in ``[scan_start, report_end)``, which lets long
sequences be split across threads without losing or duplicating hits.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(nogil=True, cache=True)
def valid_runs(codes, offsets):
    """``run[i]`` = number of consecutive ACGT characters starting at ``i``,
    not crossing the end of the sequence that contains ``i``."""
    run = np.zeros(codes.shape[0], dtype=np.int64)
    for s in range(offsets.shape[0] - 1):
        nxt = 0
        for i in range(offsets[s + 1] - 1, offsets[s] - 1, -1):
            if codes[i] < 4:
                nxt += 1
            else:
                nxt = 0
            run[i] = nxt
    return run


@numba.njit(nogil=True, cache=True, inline="always")
def _verify(codes, start, k, score, thr, mat_flat, mat_off, la_off, la_order, la_bound):
    """Add the non-window columns to ``score``; -inf if the threshold is missed."""
    base = la_off[k]
    moff = mat_off[k]
    for t in range(la_off[k + 1] - base):
        if score + la_bound[base + t] < thr:
            return -np.inf
        col = la_order[base + t]
        score += mat_flat[moff + col * 4 + codes[start + col]]
    if score >= thr:
        return score
    return -np.inf


@numba.njit(nogil=True, cache=True, inline="always")
def _code_at(codes, t, seq_end):
    if t >= seq_end:
        return 0
    c = codes[t]
    if c > 3:
        return 0
    return c


@numba.njit(nogil=True, cache=True)
def scan_positions(
    codes, run, units, w, max_wp,
    tab_ptr, tab_mat, tab_score,
    mat_flat, mat_off, mat_len, wp, full, thr,
    la_off, la_order, la_bound,
):
    """Return every hit as parallel arrays (seq, matrix, start, score)."""
    cap = 1024
    o_seq = np.empty(cap, dtype=np.int64)
    o_mat = np.empty(cap, dtype=np.int32)
    o_pos = np.empty(cap, dtype=np.int64)
    o_score = np.empty(cap, dtype=np.float64)
    n = 0
    mask = (1 << (2 * w)) - 1
    for u in range(units.shape[0]):
        sidx = units[u, 0]
        a = units[u, 1]
        rend = units[u, 2]
        e = units[u, 3]
        stop = min(rend + max_wp, e)
        code = 0
        for t in range(a, a + w - 1):
            code = (code << 2) | _code_at(codes, t, e)
        for i in range(a, stop):
            code = ((code << 2) | _code_at(codes, i + w - 1, e)) & mask
            for ent in range(tab_ptr[code], tab_ptr[code + 1]):
                k = tab_mat[ent]
                start = i - wp[k]
                if start < a or start >= rend or run[start] < mat_len[k]:
                    continue
                if full[k]:
                    score = tab_score[ent]
                else:
                    score = _verify(codes, start, k, tab_score[ent], thr[k],
                                    mat_flat, mat_off, la_off, la_order, la_bound)
                    if score == -np.inf:
                        continue
                if n == cap:
                    cap *= 2
                    o_seq = _grow_i64(o_seq, cap)
                    o_mat = _grow_i32(o_mat, cap)
                    o_pos = _grow_i64(o_pos, cap)
                    o_score = _grow_f64(o_score, cap)
                o_seq[n] = sidx
                o_mat[n] = k
                o_pos[n] = start
                o_score[n] = score
                n += 1
    return o_seq[:n].copy(), o_mat[:n].copy(), o_pos[:n].copy(), o_score[:n].copy()


@numba.njit(nogil=True, cache=True)
def scan_summary(
    codes, run, units, w, max_wp, n_motifs,
    tab_ptr, tab_mat, tab_score,
    mat_flat, mat_off, mat_len, wp, full, thr,
    la_off, la_order, la_bound,
):
    """Per (unit, motif) hit count and best score, both strands combined.

    Matrix ``k`` belongs to motif ``k % n_motifs`` (forward matrices come
    first, reverse complements second).
    """
    cnt = np.zeros(n_motifs, dtype=np.int64)
    best = np.full(n_motifs, -np.inf)
    touched = np.empty(n_motifs, dtype=np.int64)
    cap = 1024
    o_seq = np.empty(cap, dtype=np.int64)
    o_mot = np.empty(cap, dtype=np.int32)
    o_cnt = np.empty(cap, dtype=np.int64)
    o_best = np.empty(cap, dtype=np.float64)
    n = 0
    mask = (1 << (2 * w)) - 1
    for u in range(units.shape[0]):
        sidx = units[u, 0]
        a = units[u, 1]
        rend = units[u, 2]
        e = units[u, 3]
        stop = min(rend + max_wp, e)
        nt = 0
        code = 0
        for t in range(a, a + w - 1):
            code = (code << 2) | _code_at(codes, t, e)
        for i in range(a, stop):
            code = ((code << 2) | _code_at(codes, i + w - 1, e)) & mask
            for ent in range(tab_ptr[code], tab_ptr[code + 1]):
                k = tab_mat[ent]
                start = i - wp[k]
                if start < a or start >= rend or run[start] < mat_len[k]:
                    continue
                if full[k]:
                    score = tab_score[ent]
                else:
                    score = _verify(codes, start, k, tab_score[ent], thr[k],
                                    mat_flat, mat_off, la_off, la_order, la_bound)
                    if score == -np.inf:
                        continue
                j = k % n_motifs
                if cnt[j] == 0:
                    touched[nt] = j
                    nt += 1
                cnt[j] += 1
                if score > best[j]:
                    best[j] = score
        if n + nt > cap:
            while n + nt > cap:
                cap *= 2
            o_seq = _grow_i64(o_seq, cap)
            o_mot = _grow_i32(o_mot, cap)
            o_cnt = _grow_i64(o_cnt, cap)
            o_best = _grow_f64(o_best, cap)
        for t in range(nt):
            j = touched[t]
            o_seq[n] = sidx
            o_mot[n] = j
            o_cnt[n] = cnt[j]
            o_best[n] = best[j]
            n += 1
            cnt[j] = 0
            best[j] = -np.inf
    return o_seq[:n].copy(), o_mot[:n].copy(), o_cnt[:n].copy(), o_best[:n].copy()


@numba.njit(nogil=True, cache=True)
def _grow_i64(arr, cap):
    out = np.empty(cap, dtype=np.int64)
    out[: arr.shape[0]] = arr
    return out


@numba.njit(nogil=True, cache=True)
def _grow_i32(arr, cap):
    out = np.empty(cap, dtype=np.int32)
    out[: arr.shape[0]] = arr
    return out


@numba.njit(nogil=True, cache=True)
def _grow_f64(arr, cap):
    out = np.empty(cap, dtype=np.float64)
    out[: arr.shape[0]] = arr
    return out
