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
def valid_runs(codes, offsets, run):
    """Fill ``run[i]`` with the number of consecutive ACGT characters starting
    at ``i``, not crossing the end of the sequence that contains ``i``.

    Counts saturate at the largest value ``run``'s dtype can hold, so a small
    dtype is enough as long as it can represent the longest motif.
    """
    cap = np.iinfo(run.dtype).max
    for s in range(offsets.shape[0] - 1):
        nxt = 0
        for i in range(offsets[s + 1] - 1, offsets[s] - 1, -1):
            if codes[i] < 4:
                if nxt < cap:
                    nxt += 1
            else:
                nxt = 0
            run[i] = nxt
    return run


@numba.njit(nogil=True, cache=True)
def window_candidates(k_lo, k_hi, w, mat_flat, mat_off, mat_len, wp, thr, la0, store):
    """Lookup-table entries of matrices ``k_lo .. k_hi - 1``.

    For each matrix, every ``w``-mer code whose window score plus the best
    score of the remaining columns (``la0``) reaches the threshold. The codes
    are enumerated depth first, most significant base first, and a prefix is
    abandoned once even the best completion falls short, so the cost follows
    the number of entries rather than ``4**w``. Entries come out ordered by
    matrix, then code. Window scores are summed left to right.

    Returns ``(code, matrix, window score, number of entries)``. With
    ``store=False`` the entries are only counted and the arrays are empty.
    """
    cap = 4096
    o_code = np.empty(cap, dtype=np.int64)
    o_mat = np.empty(cap, dtype=np.int32)
    o_score = np.empty(cap, dtype=np.float64)
    n = 0
    digit = np.zeros(w + 1, dtype=np.int64)
    prefix = np.zeros(w + 1)
    sufmax = np.zeros(w + 1)
    for k in range(k_lo, k_hi):
        mw = min(mat_len[k], w)
        if mw == 0:
            continue
        base = mat_off[k] + wp[k] * 4
        free = 2 * (w - mw)
        n_free = 1 << free
        scale = max(1.0, abs(thr[k]), abs(la0[k]))
        sufmax[mw] = 0.0
        for d in range(mw - 1, -1, -1):
            mx = mat_flat[base + d * 4]
            for c in range(4):
                v = mat_flat[base + d * 4 + c]
                mx = max(mx, v)
                scale = max(scale, abs(v) * mw)
            sufmax[d] = sufmax[d + 1] + mx
        # pruning bound, loosened so rounding can never drop a real entry
        limit = thr[k] - la0[k] - 1e-9 * scale
        depth = 0
        digit[0] = 0
        prefix[0] = 0.0
        while depth >= 0:
            c = digit[depth]
            if c == 4:
                depth -= 1
                if depth >= 0:
                    digit[depth] += 1
                continue
            s = prefix[depth] + mat_flat[base + depth * 4 + c]
            if depth == mw - 1:
                if s + la0[k] >= thr[k]:
                    hi = 0
                    for t in range(mw):
                        hi = (hi << 2) | digit[t]
                    hi <<= free
                    if not store:
                        n += n_free
                        digit[depth] += 1
                        continue
                    if n + n_free > cap:
                        while n + n_free > cap:
                            cap *= 2
                        o_code = _grow_i64(o_code, cap)
                        o_mat = _grow_i32(o_mat, cap)
                        o_score = _grow_f64(o_score, cap)
                    for low in range(n_free):
                        o_code[n] = hi | low
                        o_mat[n] = k
                        o_score[n] = s
                        n += 1
                digit[depth] += 1
            elif s + sufmax[depth + 1] < limit:
                digit[depth] += 1
            else:
                prefix[depth + 1] = s
                depth += 1
                digit[depth] = 0
    if not store:
        return o_code[:0].copy(), o_mat[:0].copy(), o_score[:0].copy(), n
    return o_code[:n].copy(), o_mat[:n].copy(), o_score[:n].copy(), n


@numba.njit(nogil=True, cache=True)
def bucket_table(n_codes, code, mat, score):
    """Stable counting sort of table entries by code.

    Returns ``(ptr, mat, score)``; the entries of code ``c`` are
    ``ptr[c] .. ptr[c + 1] - 1``, in their input order.
    """
    ptr = np.zeros(n_codes + 1, dtype=np.int64)
    for i in range(code.shape[0]):
        ptr[code[i] + 1] += 1
    for c in range(n_codes):
        ptr[c + 1] += ptr[c]
    fill = ptr[:n_codes].copy()
    s_mat = np.empty(mat.shape[0], dtype=np.int32)
    s_score = np.empty(score.shape[0], dtype=np.float64)
    for i in range(code.shape[0]):
        j = fill[code[i]]
        fill[code[i]] = j + 1
        s_mat[j] = mat[i]
        s_score[j] = score[i]
    return ptr, s_mat, s_score


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
def dense_scores(codes, run, lo, hi, sel, mat_flat, mat_off, mat_len, out):
    """Score every placement starting in ``[lo, hi)`` for matrices ``sel``.

    ``out[j, i]`` gets the score of matrix ``sel[j]`` at absolute position
    ``i``, or NaN when the window runs past the sequence end or contains a
    non-ACGT character. Columns are summed left to right.
    """
    for j in range(sel.shape[0]):
        k = sel[j]
        m = mat_len[k]
        off = mat_off[k]
        for i in range(lo, hi):
            if run[i] < m:
                out[j, i] = np.nan
                continue
            s = 0.0
            for c in range(m):
                s += mat_flat[off + c * 4 + codes[i + c]]
            out[j, i] = s


@numba.njit(nogil=True, cache=True)
def collect_tracks(dense, offsets, n_motifs, row_rev, best):
    """Flatten dense placement scores into rows ordered by (seq, motif, strand, start).

    ``dense`` rows are ``strand_block * n_motifs + motif``; ``row_rev[r]``
    says whether row ``r`` is the reverse strand. With ``best``, ``dense``
    holds a forward block then a reverse block and each placement gives one
    row with the higher score (ties go to ``+``). NaN placements are skipped.
    """
    n_blocks = dense.shape[0] // n_motifs
    out_blocks = 1 if best else n_blocks
    n = 0
    for r in range(out_blocks * n_motifs):
        for i in range(dense.shape[1]):
            if not np.isnan(dense[r, i]):
                n += 1
    o_seq = np.empty(n, dtype=np.int64)
    o_mot = np.empty(n, dtype=np.int64)
    o_pos = np.empty(n, dtype=np.int64)
    o_rev = np.empty(n, dtype=np.int8)
    o_score = np.empty(n, dtype=np.float64)
    t = 0
    for s in range(offsets.shape[0] - 1):
        a = offsets[s]
        for j in range(n_motifs):
            for b in range(out_blocks):
                r = b * n_motifs + j
                for i in range(a, offsets[s + 1]):
                    v = dense[r, i]
                    if np.isnan(v):
                        continue
                    rev = row_rev[r]
                    if best:
                        w = dense[n_motifs + j, i]
                        if w > v:
                            v = w
                            rev = True
                    o_seq[t] = s
                    o_mot[t] = j
                    o_pos[t] = i - a
                    o_rev[t] = 1 if rev else 0
                    o_score[t] = v
                    t += 1
    return o_seq, o_mot, o_pos, o_rev, o_score


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
