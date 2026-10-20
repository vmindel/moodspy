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
        Window size of the lookup table (``4**w`` entries). Affects speed
        and memory, never which hits are found (scores may differ by ~1e-15
        because columns are summed in a different order).
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
        w: int = 7,
        pseudocount: float = DEFAULT_PSEUDOCOUNT,
        log_base: float = 2.0,
        precision: float = DEFAULT_PRECISION,
        motifmatchr_compat: bool = False,
        n_jobs: int | None = None,
    ):
        if not 1 <= w <= 12:
            raise ValueError("window size w must be between 1 and 12")
        self.motifs: list[Motif] = as_motif_list(motifs)
        self.labels = motif_labels(self.motifs)
        self.background = np.full(4, 0.25) if background is None else check_background(background)
        self.p_cutoff = p_cutoff
        self.w = int(w)
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
        self._build()

    @property
    def n_motifs(self) -> int:
        return len(self.motifs)

    def __repr__(self) -> str:
        return f"Scanner({self.n_motifs} motifs, p_cutoff={self.p_cutoff}, w={self.w})"

    # ------------------------------------------------------------------ setup
    def _build(self) -> None:
        w, bg = self.w, self.background
        # forward matrices first, reverse complements second; (length, 4) layout
        mats = [np.ascontiguousarray(p.T) for p in self.pwms]
        mats += [np.ascontiguousarray(p[::-1, ::-1].T) for p in self.pwms]
        thr = np.concatenate([self.thresholds, self.thresholds])
        n_mat = len(mats)

        lengths = np.array([m.shape[0] for m in mats], dtype=np.int64)
        mat_off = np.zeros(n_mat + 1, dtype=np.int64)
        np.cumsum(lengths * 4, out=mat_off[1:])
        wp = np.zeros(n_mat, dtype=np.int64)
        full = lengths <= w
        la_order, la_bound, la_off = [], [], [0]

        n_codes = 1 << (2 * w)
        all_codes = np.arange(n_codes, dtype=np.int64)
        digits = [(all_codes >> (2 * (w - 1 - i))) & 3 for i in range(w)]
        tab_code, tab_mat, tab_score = [], [], []

        for k, mat in enumerate(mats):
            m = mat.shape[0]
            ed = mat.max(axis=1) - mat @ bg
            if m <= w:
                pos, order = 0, np.zeros(0, dtype=np.int64)
            else:
                pos = _window_position(ed, w)
                rest = np.r_[0:pos, pos + w : m]
                order = rest[np.argsort(-ed[rest], kind="stable")]
            wp[k] = pos
            colmax = mat.max(axis=1)
            bound = np.cumsum(colmax[order][::-1])[::-1] if order.size else np.zeros(0)
            la_order.append(order)
            la_bound.append(bound)
            la_off.append(la_off[-1] + order.size)

            score = np.zeros(n_codes)
            for i in range(min(m, w)):
                score = score + mat[pos + i][digits[i]]
            lookahead = bound[0] if bound.size else 0.0
            hit = np.flatnonzero(score + lookahead >= thr[k])
            tab_code.append(hit)
            tab_mat.append(np.full(hit.size, k, dtype=np.int32))
            tab_score.append(score[hit])

        tab_code = np.concatenate(tab_code)
        order = np.argsort(tab_code, kind="stable")
        self._tab_ptr = np.searchsorted(tab_code[order], np.arange(n_codes + 1)).astype(np.int64)
        self._tab_mat = np.ascontiguousarray(np.concatenate(tab_mat)[order])
        self._tab_score = np.ascontiguousarray(np.concatenate(tab_score)[order])
        self._mat_flat = np.concatenate([m.ravel() for m in mats])
        self._mat_off = mat_off
        self._mat_len = lengths
        self._wp = wp
        self._full = full
        self._thr = thr
        self._la_off = np.asarray(la_off, dtype=np.int64)
        self._la_order = np.concatenate(la_order).astype(np.int64)
        self._la_bound = np.concatenate(la_bound).astype(np.float64)
        self._max_wp = int(wp.max())

    def _tables(self):
        return (
            self._tab_ptr, self._tab_mat, self._tab_score,
            self._mat_flat, self._mat_off, self._mat_len, self._wp, self._full, self._thr,
            self._la_off, self._la_order, self._la_bound,
        )

    # ------------------------------------------------------------------ scanning
    def _prepare(self, seqs: Sequence[str], split_length: int):
        codes, offsets = encode_many(seqs)
        run = _kernels.valid_runs(codes, offsets)
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
        tail = (self.w, self._max_wp, *extra, *self._tables())
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
        pos = np.concatenate([p[2] for p in parts])
        score = np.concatenate([p[3] for p in parts])
        motif = k % self.n_motifs
        reverse = k >= self.n_motifs
        start = pos - offsets[s_idx]
        order = np.lexsort((start, reverse, motif, s_idx))
        s_idx, motif, reverse, start, score = (x[order] for x in (s_idx, motif, reverse, start, score))
        return pd.DataFrame(
            {
                "seq": s_idx,
                "motif": pd.Categorical.from_codes(motif, categories=self.labels),
                "start": start,
                "end": start + self._mat_len[k[order]],
                "strand": pd.Categorical.from_codes(reverse.astype(np.int8), categories=["+", "-"]),
                "score": score,
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
