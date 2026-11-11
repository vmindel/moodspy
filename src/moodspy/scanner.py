"""Reusable multi-motif scanner."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import _kernels
from ._alphabet import encode_many
from .motifs import DEFAULT_PSEUDOCOUNT, Motif, as_motif_list, check_background, motif_labels
from .thresholds import DEFAULT_PRECISION, thresholds_from_p

#: Sequences longer than this are split into pieces scanned in parallel.
SPLIT_LENGTH = 1_000_000

#: Automatic window size: ``(total bases below, w)`` tiers, then the last ``w``.
AUTO_W_TIERS = ((1_000_000, 7), (100_000_000, 9), (None, 10))
#: An automatically chosen window is reduced (not below 7) until its lookup
#: table fits in this many bytes.
AUTO_TABLE_BYTES = 256 * 2**20
_ENTRY_BYTES = 12  # int32 matrix index + float64 window score


def _resolve_jobs(n_jobs: int | None) -> int:
    if n_jobs is None or n_jobs <= 0:
        return os.cpu_count() or 1
    return int(n_jobs)


class Scanner:
    """Scan many sequences for many motifs on both strands.

    Building a scanner computes thresholds and lookup tables once; it can
    then be reused on any number of sequence sets. Most users should call
    [`moodspy.match_motifs`][moodspy.match_motifs], which wraps this class.

    Parameters
    ----------
    motifs
        Motifs, matrices, a mapping of label -> matrix, or a motif file path.
    background
        A/C/G/T frequencies used both for log-odds scores and for p-values.
        Default uniform.
    p_cutoff
        P-value cutoff used to derive each motif's score threshold.
    thresholds
        Explicit score thresholds (one per motif); overrides ``p_cutoff``.
    w
        Window size of the lookup table (``4**w`` entries), 1 to 12. Affects
        speed and memory, never which hits are found (scores may differ by
        ~1e-15 because columns are summed in a different order). The default
        ``None`` picks it for each call from the amount of sequence: 7 below
        1 Mb, 9 below 100 Mb, 10 above, reduced if the table would exceed
        256 MB. Tables are built on first use and kept.
    pseudocount
        Pseudocount for count / probability matrices.
    log_base
        Base of the log-odds scores reported (2, as in motifmatchr).
    motifmatchr_compat
        Score matrices exactly as motifmatchr does, including its inverted
        background correction (see [`Motif.log_odds`][moodspy.Motif.log_odds]). Only useful to
        reproduce motifmatchr results.
    n_jobs
        Threads to use (``None`` or ``-1`` for all cores).
    """

    def __init__(
        self,
        motifs: Any,
        background: Any = None,
        p_cutoff: float = 5e-5,
        *,
        thresholds: Sequence[float] | None = None,
        w: int | None = None,
        pseudocount: float = DEFAULT_PSEUDOCOUNT,
        log_base: float = 2.0,
        precision: float = DEFAULT_PRECISION,
        motifmatchr_compat: bool = False,
        n_jobs: int | None = None,
    ):
        if w is not None and not 1 <= w <= 12:
            raise ValueError("window size w must be between 1 and 12")
        self.motifs: list[Motif] = as_motif_list(motifs)
        self.labels = motif_labels(self.motifs)
        self.background = np.full(4, 0.25) if background is None else check_background(background)
        self.p_cutoff = p_cutoff
        self.w = None if w is None else int(w)
        self.n_jobs = _resolve_jobs(n_jobs)
        self.pwms = [
            m.log_odds(self.background, pseudocount, log_base, motifmatchr_compat) for m in self.motifs
        ]
        if thresholds is None:
            thr = thresholds_from_p(self.pwms, p_cutoff, self.background, precision, self.n_jobs)
        else:
            thr = np.asarray(thresholds, dtype=np.float64).ravel()
            if thr.shape != (len(self.pwms),):
                raise ValueError(f"need {len(self.pwms)} thresholds, got {thr.size}")
            self.p_cutoff = None
        self.thresholds = thr
        # forward matrices first, reverse complements second; (length, 4) layout
        mats = [np.ascontiguousarray(p.T) for p in self.pwms]
        mats += [np.ascontiguousarray(p[::-1, ::-1].T) for p in self.pwms]
        self._mats = mats
        self._thr = np.concatenate([thr, thr])
        self._mat_len = np.array([m.shape[0] for m in mats], dtype=np.int64)
        self._mat_off = np.zeros(len(mats) + 1, dtype=np.int64)
        np.cumsum(self._mat_len * 4, out=self._mat_off[1:])
        self._mat_flat = np.concatenate([m.ravel() for m in mats])
        self._layouts: dict[int, tuple] = {}
        self._table_cache: dict[int, tuple] = {}
        self._auto_w: dict[int, int] = {}
        if self.w is not None:
            self._table(self.w)

    @property
    def n_motifs(self) -> int:
        return len(self.motifs)

    def __repr__(self) -> str:
        w = "auto" if self.w is None else self.w
        return f"Scanner({self.n_motifs} motifs, p_cutoff={self.p_cutoff}, w={w})"

    # ------------------------------------------------------------------ setup
    def _layout(self, w: int) -> tuple:
        """Window position and verification order of every matrix for ``w``."""
        if w in self._layouts:
            return self._layouts[w]
        bg, n_mat = self.background, len(self._mats)
        wp = np.zeros(n_mat, dtype=np.int64)
        la_order, la_bound, la_off = [], [], [0]
        la0 = np.zeros(n_mat)
        for k, mat in enumerate(self._mats):
            m = mat.shape[0]
            colmax = mat.max(axis=1)
            ed = colmax - mat @ bg
            if m <= w:
                pos, order = 0, np.zeros(0, dtype=np.int64)
            else:
                pos = _window_position(ed, w)
                rest = np.r_[0:pos, pos + w : m]
                order = rest[np.argsort(-ed[rest], kind="stable")]
            wp[k] = pos
            bound = np.cumsum(colmax[order][::-1])[::-1] if order.size else np.zeros(0)
            la_order.append(order)
            la_bound.append(bound)
            la_off.append(la_off[-1] + order.size)
            la0[k] = bound[0] if bound.size else 0.0
        self._layouts[w] = (
            wp, self._mat_len <= w, np.asarray(la_off, dtype=np.int64),
            np.concatenate(la_order).astype(np.int64), np.concatenate(la_bound).astype(np.float64), la0,
        )
        return self._layouts[w]

    def _candidates(self, w: int, store: bool) -> list[tuple]:
        """Run the table kernel over blocks of matrices, in threads."""
        wp, la0 = self._layout(w)[0], self._layout(w)[5]
        n_mat = len(self._mats)
        n_blocks = 1 if self.n_jobs == 1 else min(n_mat, self.n_jobs * 4)
        edges = np.linspace(0, n_mat, n_blocks + 1).astype(np.int64)
        args = (w, self._mat_flat, self._mat_off, self._mat_len, wp, self._thr, la0, store)
        if n_blocks == 1:
            return [_kernels.window_candidates(0, n_mat, *args)]
        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            futures = [
                pool.submit(_kernels.window_candidates, edges[b], edges[b + 1], *args)
                for b in range(n_blocks)
            ]
            return [f.result() for f in futures]

    def _table(self, w: int) -> tuple:
        """``(max_wp, *kernel tables)`` for window size ``w`` (built once)."""
        if w not in self._table_cache:
            wp, full, la_off, la_order, la_bound, _ = self._layout(w)
            parts = self._candidates(w, True)
            tab_ptr, tab_mat, tab_score = _kernels.bucket_table(
                1 << (2 * w), *(np.concatenate([p[i] for p in parts]) for i in range(3))
            )
            self._table_cache[w] = (
                int(wp.max()),
                tab_ptr, tab_mat, tab_score,
                self._mat_flat, self._mat_off, self._mat_len, wp, full, self._thr,
                la_off, la_order, la_bound,
            )
        return self._table_cache[w]

    def _pick_w(self, n_bases: int) -> int:
        """The window size to use for ``n_bases`` of sequence."""
        if self.w is not None:
            return self.w
        tier = next(w for limit, w in AUTO_W_TIERS if limit is None or n_bases < limit)
        if tier not in self._auto_w:
            w = tier
            while w > 7 and w not in self._table_cache:
                n_entries = sum(p[3] for p in self._candidates(w, False))
                if n_entries * _ENTRY_BYTES + 8 * 4**w <= AUTO_TABLE_BYTES:
                    break
                w -= 1
            self._auto_w[tier] = w
        return self._auto_w[tier]

    # ------------------------------------------------------------------ scanning
    def _prepare(self, seqs: Sequence[str], split_length: int):
        codes, offsets = encode_many(seqs)
        # saturating counts: only "at least the motif length" is ever asked
        longest = int(self._mat_len.max())
        dtype = np.uint8 if longest <= 255 else np.uint16 if longest <= 65535 else np.int64
        run = _kernels.valid_runs(codes, offsets, np.empty(codes.shape[0], dtype=dtype))
        units = []
        for s in range(len(seqs)):
            a, e = int(offsets[s]), int(offsets[s + 1])
            for start in range(a, e, split_length):
                units.append((s, start, min(start + split_length, e), e))
        units = np.array(units, dtype=np.int64).reshape(-1, 4)
        return codes, offsets, run, units

    def _chunks(self, units: np.ndarray) -> list[np.ndarray]:
        if self.n_jobs == 1 or len(units) < 2:
            return [units]
        work = units[:, 2] - units[:, 1]
        n_chunks = min(len(units), self.n_jobs * 8)
        bounds = np.searchsorted(np.cumsum(work), np.linspace(0, work.sum(), n_chunks + 1)[1:-1])
        return [c for c in np.split(units, np.unique(bounds)) if len(c)]

    def _run(self, kernel, codes, run, units, *extra):
        """Call ``kernel(codes, run, chunk, w, max_wp, *extra, *tables)`` per chunk."""
        chunks = self._chunks(units)
        w = self._pick_w(codes.shape[0])
        max_wp, *tables = self._table(w)
        tail = (w, max_wp, *extra, *tables)
        if len(chunks) == 1:
            return [kernel(codes, run, chunks[0], *tail)]
        with ThreadPoolExecutor(max_workers=self.n_jobs) as pool:
            futures = [pool.submit(kernel, codes, run, c, *tail) for c in chunks]
            return [f.result() for f in futures]

    def scan(self, seqs: Sequence[str], *, split_length: int = SPLIT_LENGTH) -> pd.DataFrame:
        """All hits as a DataFrame.

        Columns: ``seq`` (sequence index), ``motif`` (label), ``start``,
        ``end`` (0-based, half-open, on the forward strand), ``strand`` and
        ``score``. Rows are sorted by sequence, motif, strand and start.
        """
        if isinstance(seqs, str):
            seqs = [seqs]
        codes, offsets, run, units = self._prepare(seqs, split_length)
        parts = self._run(_kernels.scan_positions, codes, run, units)
        s_idx = np.concatenate([p[0] for p in parts])
        k = np.concatenate([p[1] for p in parts]).astype(np.int64)
        start = np.concatenate([p[2] for p in parts])
        start -= offsets[s_idx]
        score = np.concatenate([p[3] for p in parts])
        n_m = self.n_motifs
        fwd_len = self._mat_len[:n_m]
        span = int(np.diff(offsets).max()) + 1
        if len(seqs) * 2 * n_m * span < 2**62:
            # one integer per hit, ordered like (seq, motif, strand, start)
            key = (s_idx * (2 * n_m) + (k % n_m) * 2 + (k >= n_m)) * span + start
            order = np.argsort(key)
            group, start = np.divmod(key[order], span)
            reverse = group & 1
            s_idx, motif = np.divmod(group >> 1, n_m)
        else:
            motif = k % n_m
            reverse = (k >= n_m).astype(np.int64)
            order = np.lexsort((start, reverse, motif, s_idx))
            s_idx, motif, reverse, start = (x[order] for x in (s_idx, motif, reverse, start))
        return pd.DataFrame(
            {
                "seq": s_idx,
                "motif": pd.Categorical.from_codes(motif, categories=self.labels),
                "start": start,
                "end": start + fwd_len[motif],
                "strand": pd.Categorical.from_codes(reverse.astype(np.int8), categories=["+", "-"]),
                "score": score[order],
            }
        )

    def summarize(
        self, seqs: Sequence[str], *, split_length: int = SPLIT_LENGTH
    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Hit counts and best hit score per (sequence, motif), both strands.

        Returns two ``(n_seqs, n_motifs)`` CSR matrices: integer counts and
        float max scores (absent entries mean no hit).
        """
        if isinstance(seqs, str):
            seqs = [seqs]
        codes, _, run, units = self._prepare(seqs, split_length)
        parts = self._run(_kernels.scan_summary, codes, run, units, self.n_motifs)
        s_idx = np.concatenate([p[0] for p in parts])
        mot = np.concatenate([p[1] for p in parts]).astype(np.int64)
        cnt = np.concatenate([p[2] for p in parts])
        best = np.concatenate([p[3] for p in parts])
        shape = (len(seqs), self.n_motifs)
        key = s_idx * self.n_motifs + mot
        if key.size and (np.diff(key) <= 0).any():  # split sequences: merge pieces
            order = np.argsort(key, kind="stable")
            key, cnt, best = key[order], cnt[order], best[order]
            first = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
            key = key[first]
            cnt = np.add.reduceat(cnt, first)
            best = np.maximum.reduceat(best, first)
        rows, cols = np.divmod(key, self.n_motifs)
        counts = sp.csr_matrix((cnt.astype(np.int32), (rows, cols)), shape=shape)
        scores = sp.csr_matrix((best, (rows, cols)), shape=shape)
        return counts, scores


def _window_position(ed: np.ndarray, w: int) -> int:
    """Start of the length-``w`` window with the largest expected score gain."""
    current = float(ed[:w].sum())
    best, pos = current, 0
    for i in range(len(ed) - w):
        current -= ed[i]
        current += ed[i + w]
        if current > best:
            best, pos = current, i + 1
    return pos
