"""Sequence manipulation: k-let backgrounds, shuffling, random sequences,
masking and motif implanting.

A port of universalmotif's sequence utilities (``get_bkg``,
``shuffle_sequences``, ``create_sequences``, ``mask_seqs``, ``mask_ranges``
and ``implant_motifs``). Every function takes sequences the way
[`match_motifs`][moodspy.match_motifs] does (a string, a list, a dict of
name -> sequence, a pandas Series or a FASTA path) and returns the same
container type (a FASTA path gives a dict).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from itertools import product
from typing import Any, Literal

import numba
import numpy as np
import pandas as pd

from ._alphabet import _LUT, ALPHABET
from .manipulate import RngLike, as_rng, ppm
from .motifs import as_motif_list, motif_labels

_COMP = str.maketrans("ACGTacgtNnRYMKSWHBVDrymkswhbvd", "TGCAtgcaNnYRKMSWDVBHyrkmswdvbh")


def revcomp(seq: str) -> str:
    """Reverse complement of a DNA string (IUPAC codes and case are kept)."""
    return seq.translate(_COMP)[::-1]


# ---------------------------------------------------------------------- containers
def _unpack(seqs: Any) -> tuple[list[str], Any]:
    """Return (list of sequences, function that packs a list back into the input type)."""
    if isinstance(seqs, os.PathLike) or (isinstance(seqs, str) and os.path.isfile(seqs)):
        from .genome import read_fasta

        seqs = read_fasta(seqs)
    if isinstance(seqs, str):
        return [seqs], lambda out: out[0]
    if isinstance(seqs, pd.Series):
        index = seqs.index
        return [str(s) for s in seqs], lambda out: pd.Series(out, index=index, name=seqs.name)
    if isinstance(seqs, Mapping):
        names = list(seqs)
        return [str(seqs[k]) for k in names], lambda out: dict(zip(names, out))
    lst = [str(s) for s in seqs]
    return lst, lambda out: list(out)


def _names(seqs: Any, n: int) -> list[str]:
    if isinstance(seqs, Mapping):
        return [str(k) for k in seqs]
    if isinstance(seqs, pd.Series):
        return [str(k) for k in seqs.index]
    return [f"seq{i}" for i in range(n)]


def write_fasta(seqs: Any, path: str | os.PathLike, width: int = 80) -> None:
    """Write sequences to a FASTA file (unnamed sequences become ``seq0``, ``seq1``...)."""
    if isinstance(seqs, str):
        seqs = [seqs]
    lst, _ = _unpack(seqs)
    names = _names(seqs, len(lst))
    with open(path, "w") as fh:
        for name, s in zip(names, lst):
            fh.write(f">{name}\n")
            fh.writelines(s[i : i + width] + "\n" for i in range(0, max(len(s), 1), width))


# ---------------------------------------------------------------------- k-let counting
def _klets(k: int) -> list[str]:
    return ["".join(p) for p in product(ALPHABET, repeat=k)]


@numba.njit(nogil=True, cache=True)
def _count_klets(codes, k):
    out = np.zeros(4**k, dtype=np.int64)
    idx = 0
    valid = 0
    mask = 4 ** (k - 1)
    for c in codes:
        if c >= 4:
            valid = 0
            idx = 0
            continue
        idx = (idx % mask) * 4 + c if k > 1 else c
        valid += 1
        if valid >= k:
            out[idx] += 1
    return out


def _codes(seq: str) -> np.ndarray:
    return _LUT[np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)]


def get_bkg(
    seqs: Any,
    k: int | list[int] | range = (1, 2, 3),
    *,
    rc: bool = False,
    pseudocount: float = 0,
    merge: bool = True,
    meme_file: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """Count k-lets (k-mers) and their frequencies.

    universalmotif's ``get_bkg``. k-lets containing anything other than
    ``ACGT`` (case-insensitive, so soft-masked bases count) are skipped.

    Parameters
    ----------
    seqs
        Sequences (string, list, dict, Series or FASTA path).
    k
        One or several k-let sizes.
    rc
        Also count the reverse complement, giving strand-symmetric frequencies.
    pseudocount
        Added to every count before computing probabilities.
    merge
        ``True``: one table summed over all sequences. ``False``: one block
        of rows per sequence, with a ``seq`` column.
    meme_file
        Also write a MEME background model (requires ``k`` = 1, 2, ..., n
        and no zero or one probabilities). Pass this to
        ``fimo --bfile`` or ``meme -bfile``.

    Returns
    -------
    pandas.DataFrame
        Columns ``klet``, ``count``, ``probability`` (and ``seq`` if
        ``merge=False``); probabilities sum to 1 within each k.
    """
    ks = [k] if isinstance(k, (int, np.integer)) else sorted({int(x) for x in k})
    if not ks or min(ks) < 1:
        raise ValueError("k must be at least 1")
    lst, _ = _unpack(seqs)
    if rc:
        lst = lst + [revcomp(s) for s in lst]
    names = _names(seqs, len(lst) // (2 if rc else 1))
    if rc:
        names = names + names
    if lst and min(len(s) for s in lst) < max(ks):
        raise ValueError("k must not exceed the length of the shortest sequence")
    codes = [_codes(s) for s in lst]
    frames = []
    for kk in ks:
        counts = np.array([_count_klets(c, kk) for c in codes], dtype=np.float64)
        if merge:
            c = counts.sum(axis=0) + pseudocount
            tot = c.sum()
            frames.append(pd.DataFrame({"klet": _klets(kk), "count": c,
                                        "probability": c / tot if tot > 0 else 0.0}))
        else:
            df = pd.DataFrame(counts, columns=_klets(kk))
            df["seq"] = names
            df = df.groupby("seq", sort=False).sum()
            df = df + pseudocount
            prob = df.div(df.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
            klets = np.tile(np.array(df.columns), len(df))
            frames.append(pd.DataFrame({
                "seq": np.repeat(np.array(df.index), df.shape[1]), "klet": klets,
                "count": df.to_numpy().ravel(), "probability": prob.to_numpy().ravel(),
            }))
    out = pd.concat(frames, ignore_index=True)
    if not merge:
        order = {n: i for i, n in enumerate(dict.fromkeys(names))}
        key = out["seq"].map(order) * (max(ks) + 1) + out["klet"].str.len()
        out = out.iloc[np.argsort(key.to_numpy(), kind="stable")].reset_index(drop=True)
        out = out[["seq", "klet", "count", "probability"]]
    if meme_file is not None:
        if not merge or ks != list(range(1, max(ks) + 1)):
            raise ValueError("a MEME background needs merge=True and k = 1, 2, ..., n")
        p = out["probability"].to_numpy()
        if (p == 0).any() or (p == 1).any():
            raise ValueError("MEME backgrounds cannot contain 0 or 1; use a pseudocount")
        with open(meme_file, "w") as fh:
            for kk in ks:
                fh.write(f"#   order {kk - 1}\n")
                fh.writelines(f"{row.klet:<8}{row.probability:.3e} \n" for row in out[out["klet"].str.len() == kk].itertuples())
    return out


# ---------------------------------------------------------------------- shuffling
@numba.njit(nogil=True, cache=True)
def _seed(seed):
    np.random.seed(seed)


@numba.njit(nogil=True, cache=True)
def _pick(weights):
    tot = weights.sum()
    u = np.random.random() * tot
    acc = 0.0
    for i in range(weights.shape[0]):
        acc += weights[i]
        if u < acc:
            return i
    for i in range(weights.shape[0] - 1, -1, -1):
        if weights[i] > 0:
            return i
    return weights.shape[0] - 1


@numba.njit(nogil=True, cache=True)
def _euler(x, a, k):
    """Uniformly random sequence with exactly the same k-let counts (Altschul & Erickson 1985).

    Port of universalmotif's shuffle_euler: a random spanning arborescence
    of the (k-1)-let de Bruijn multigraph rooted at the last (k-1)-let
    (Wilson's loop-erased random walks), last-exit edges fixed, other edges
    shuffled, then an Eulerian walk from the first (k-1)-let.
    """
    n = x.shape[0]
    m = a ** (k - 1)
    edges = np.zeros((m, a), dtype=np.int64)
    v = 0
    for i in range(k - 1):
        v = v * a + x[i]
    first = v
    for i in range(k - 1, n):
        edges[v, x[i]] += 1
        v = (v * a + x[i]) % m
    last = v
    w = edges.astype(np.float64)
    in_tree = np.zeros(m, dtype=np.bool_)
    nxt = np.zeros(m, dtype=np.int64)
    for u in range(m):
        if edges[u].sum() == 0:
            in_tree[u] = True
    in_tree[last] = True
    for i in range(m):
        u = i
        while not in_tree[u]:
            nxt[u] = _pick(w[u])
            u = (u * a + nxt[u]) % m
        u = i
        while not in_tree[u]:
            in_tree[u] = True
            u = (u * a + nxt[u]) % m
    # edge lists: shuffled non-tree edges, then the tree (last-exit) edge
    start = np.zeros(m + 1, dtype=np.int64)
    for u in range(m):
        start[u + 1] = start[u] + edges[u].sum()
    lists = np.empty(start[m], dtype=np.int64)
    for u in range(m):
        cnt = edges[u].copy()
        if u != last and cnt.sum() > 0:
            cnt[nxt[u]] -= 1
        p = start[u]
        for c in range(a):
            for _ in range(cnt[c]):
                lists[p] = c
                p += 1
        np.random.shuffle(lists[start[u] : p])
        if u != last and edges[u].sum() > 0:
            lists[p] = nxt[u]
    out = np.empty(n, dtype=np.int64)
    out[: k - 1] = x[: k - 1]
    used = np.zeros(m, dtype=np.int64)
    v = first
    for i in range(k - 1, n):
        c = lists[start[v] + used[v]]
        used[v] += 1
        out[i] = c
        v = (v * a + c) % m
    return out


@numba.njit(nogil=True, cache=True)
def _markov(x, a, k):
    """Random sequence from the order-(k-1) Markov chain fitted to ``x``."""
    n = x.shape[0]
    m = a ** (k - 1)
    counts = np.zeros(m * a, dtype=np.float64)
    v = 0
    for i in range(n):
        v = (v * a + x[i]) % (m * a)
        if i >= k - 1:
            counts[v] += 1
    letters = np.zeros(a)
    for i in range(n):
        letters[x[i]] += 1
    out = np.empty(n, dtype=np.int64)
    w = _pick(counts)
    for i in range(k - 1, -1, -1):
        out[i] = w % a
        w //= a
    for i in range(k, n):
        prev = 0
        for j in range(i - k + 1, i):
            prev = prev * a + out[j]
        row = counts[prev * a : prev * a + a]
        out[i] = _pick(row) if row.sum() > 0 else _pick(letters)
    return out


def _shuffle_one(seq: str, k: int, method: str, gen: np.random.Generator) -> str:
    if len(seq) < 2:
        return seq
    raw = np.frombuffer(seq.encode("latin-1", errors="replace"), dtype=np.uint8)
    alph, x = np.unique(raw, return_inverse=True)
    x = x.astype(np.int64)
    if k == 1:
        y = gen.permutation(x)
    elif method == "linear":
        nblocks = len(x) // k
        blocks = x[: nblocks * k].reshape(nblocks, k)[gen.permutation(nblocks)].ravel()
        y = np.concatenate([blocks, x[nblocks * k :]])
    else:
        _seed(int(gen.integers(2**31 - 1)))
        y = _euler(x, len(alph), k) if method == "euler" else _markov(x, len(alph), k)
    return alph[y].tobytes().decode("latin-1")


def shuffle_sequences(
    seqs: Any,
    k: int = 1,
    method: Literal["euler", "markov", "linear"] = "euler",
    rng: RngLike = None,
) -> Any:
    """Shuffle sequences while keeping their k-let composition.

    universalmotif's ``shuffle_sequences``. Each sequence is shuffled on its
    own, over the characters it contains (``N`` and lowercase are ordinary
    letters here).

    Parameters
    ----------
    k
        k-let size to preserve. ``k=1`` is a plain permutation; ``k=2``
        keeps dinucleotide counts (CpG content), the usual choice for
        background sequences.
    method
        ``"euler"``: exact k-let counts, uniformly sampled among all such
        sequences (Altschul-Erickson / uShuffle algorithm); keeps the first
        k-1 letters. ``"markov"``: a new sequence from the order-(k-1)
        Markov chain of the original, so k-let counts are only preserved
        on average. ``"linear"``: cut into consecutive k-mers and shuffle
        those (fast, but only keeps the counts of those k-mers).
    rng
        Seed or ``numpy.random.Generator``.
    """
    if method not in ("euler", "markov", "linear"):
        raise ValueError("method must be 'euler', 'markov' or 'linear'")
    if k < 1:
        raise ValueError("k must be at least 1")
    lst, pack = _unpack(seqs)
    if lst and min(len(s) for s in lst) < k:
        raise ValueError("k must not exceed the length of the shortest sequence")
    gen = as_rng(rng)
    return pack([_shuffle_one(s, int(k), method, gen) for s in lst])


# ---------------------------------------------------------------------- random sequences
def create_sequences(
    n: int = 100,
    length: int = 100,
    freqs: Any = None,
    rng: RngLike = None,
) -> list[str]:
    """Random DNA sequences from a background model.

    universalmotif's ``create_sequences``. With k-let frequencies the
    sequences come from the order-(k-1) Markov chain they define: the first
    k letters are drawn from the k-let frequencies, then each next letter
    from ``P(next | previous k-1)``.

    Parameters
    ----------
    freqs
        ``None`` (uniform), 4 base frequencies, a mapping ``k-let ->
        frequency`` covering all ``4**k`` k-lets, or a
        [`get_bkg`][moodspy.get_bkg] table (its largest k is used).
    """
    gen = as_rng(rng)
    if freqs is None:
        freqs = np.full(4, 0.25)
    if isinstance(freqs, pd.DataFrame):
        kmax = freqs["klet"].str.len().max()
        sub = freqs[freqs["klet"].str.len() == kmax]
        freqs = dict(zip(sub["klet"], sub["probability"]))
    if isinstance(freqs, (Mapping, pd.Series)):
        d = {str(key).upper(): float(v) for key, v in dict(freqs).items()}
        k = len(next(iter(d)))
        missing = set(_klets(k)) - set(d)
        if missing:
            raise ValueError(f"freqs is missing {len(missing)} of the {4**k} {k}-lets")
        p = np.array([d[x] for x in _klets(k)])
    else:
        p = np.asarray(freqs, dtype=np.float64).ravel()
        k = round(np.log(len(p)) / np.log(4))
        if 4**k != len(p):
            raise ValueError("freqs must have 4**k entries in ACGT order")
    if (p < 0).any() or p.sum() <= 0:
        raise ValueError("freqs must be non-negative and not all zero")
    p = p / p.sum()
    letters = np.frombuffer(ALPHABET.encode(), dtype=np.uint8)
    if k == 1:
        idx = gen.choice(4, size=(n, length), p=p)
        return [letters[row].tobytes().decode() for row in idx]
    trans = p.reshape(4 ** (k - 1), 4)
    rowsum = trans.sum(axis=1, keepdims=True)
    trans = np.where(rowsum > 0, trans / np.where(rowsum > 0, rowsum, 1), 0.25)
    cum = np.cumsum(trans, axis=1)
    out = []
    m = 4 ** (k - 1)
    for _ in range(n):
        first = gen.choice(len(p), p=p)
        s = [(first // 4 ** (k - 1 - i)) % 4 for i in range(k)][:length]
        prev = first % m
        u = gen.random(max(length - k, 0))
        for x in u:
            c = int(np.searchsorted(cum[prev], x * cum[prev, -1], side="right"))
            c = min(c, 3)
            s.append(c)
            prev = (prev * 4 + c) % m
        out.append(letters[np.array(s, dtype=np.int64)].tobytes().decode())
    return out


# ---------------------------------------------------------------------- masking
def mask_seqs(seqs: Any, pattern: str, rc: bool = False, letter: str = "N") -> Any:
    """Replace every occurrence of ``pattern`` (overlapping, case-insensitive) with ``letter``.

    universalmotif's ``mask_seqs``, which masks with ``"-"`` by default; here
    the default is ``"N"`` so the output stays a valid DNA FASTA. Masked
    bases never match a motif. With ``rc=True`` the reverse complement of
    the pattern is masked too.
    """
    if len(letter) != 1:
        raise ValueError("letter must be a single character")
    if not pattern:
        raise ValueError("empty pattern")
    lst, pack = _unpack(seqs)
    pats = {pattern.upper()}
    if rc:
        pats.add(revcomp(pattern.upper()))
    out = []
    for s in lst:
        up = s.upper()
        hit = np.zeros(len(s) + 1, dtype=np.int64)
        for pat in pats:
            i = up.find(pat)
            while i >= 0:
                hit[i] += 1
                hit[i + len(pat)] -= 1
                i = up.find(pat, i + 1)
        masked = np.cumsum(hit[:-1]) > 0
        out.append(_apply_mask(s, masked, letter))
    return pack(out)


def _apply_mask(s: str, masked: np.ndarray, letter: str) -> str:
    if not masked.any():
        return s
    arr = np.frombuffer(s.encode("latin-1", errors="replace"), dtype=np.uint8).copy()
    arr[masked] = ord(letter)
    return arr.tobytes().decode("latin-1")


def mask_ranges(seqs: Any, ranges: pd.DataFrame, letter: str = "N") -> Any:
    """Mask the ``[start, end)`` ranges in ``ranges`` with ``letter``.

    ``ranges`` needs ``seq``, ``start`` and ``end`` columns with 0-based,
    half-open coordinates *within* each sequence, e.g. the output of
    [`Scanner.scan`][moodspy.Scanner.scan] or of
    ``match_motifs(..., out="positions")`` on plain sequences. ``seq`` holds
    sequence names (dict/Series input) or positions in the list. Use it to
    blank out hits of known motifs before rescanning or before motif
    discovery.
    """
    if len(letter) != 1:
        raise ValueError("letter must be a single character")
    missing = {"seq", "start", "end"} - set(ranges.columns)
    if missing:
        raise ValueError(f"ranges is missing columns {sorted(missing)}")
    lst, pack = _unpack(seqs)
    names = _names(seqs, len(lst))
    pos = {n: i for i, n in enumerate(names)}
    masks = [np.zeros(len(s) + 1, dtype=np.int64) for s in lst]
    for seq, start, end in zip(ranges["seq"], ranges["start"], ranges["end"]):
        i = pos.get(str(seq)) if str(seq) in pos else int(seq)
        start, end = max(int(start), 0), min(int(end), len(lst[i]))
        if end > start:
            masks[i][start] += 1
            masks[i][end] -= 1
    return pack([_apply_mask(s, np.cumsum(m[:-1]) > 0, letter) for s, m in zip(lst, masks)])


# ---------------------------------------------------------------------- implanting
def implant_motifs(
    motifs: Any,
    seqs: Any,
    *,
    n_per_seq: int = 1,
    rate: float | None = None,
    positions: list | None = None,
    strand: Literal["both", "+"] = "both",
    centre_bias: int = 1,
    min_spacing: int = 0,
    max_retries: int = 100,
    rng: RngLike = None,
) -> tuple[Any, pd.DataFrame]:
    """Overwrite parts of sequences with sites sampled from motifs, recording where.

    universalmotif's ``implant_motifs``: ground-truth positives for
    benchmarking scans, enrichment or motif discovery. For each implant a
    motif is picked uniformly, a strand is chosen, a site is sampled column
    by column from the motif's probabilities and written over the sequence.

    Parameters
    ----------
    n_per_seq
        Implants per sequence (the default mode).
    rate
        Instead: implants per base; the count per sequence is
        ``Poisson(rate * length)``.
    positions
        Instead: a list with one list of 0-based start positions per
        sequence. Out-of-bounds positions are dropped with a warning.
    strand
        ``"both"`` (random strand) or ``"+"``.
    centre_bias
        1 for uniform positions; ``n > 1`` averages ``n`` uniforms
        (Irwin-Hall), concentrating implants near the centre, like ChIP or
        ATAC summits.
    min_spacing
        Minimum number of bases between implants in one sequence.
    max_retries
        Placement attempts per implant before giving up (with a warning).

    Returns
    -------
    (sequences, implants)
        The modified sequences (same container as ``seqs``) and a DataFrame
        with ``seq``, ``motif``, ``start``, ``end`` (0-based, half-open),
        ``strand`` and the implanted ``site`` (as it reads on the forward
        strand).
    """
    import warnings

    from .manipulate import sample_sites

    if strand not in ("both", "+"):
        raise ValueError("strand must be 'both' or '+'")
    if centre_bias < 1 or min_spacing < 0 or max_retries < 1:
        raise ValueError("centre_bias and max_retries must be >= 1, min_spacing >= 0")
    mots = as_motif_list(motifs)
    labels = motif_labels(mots)
    widths = [m.length for m in mots]
    gen = as_rng(rng)
    lst, pack = _unpack(seqs)
    names = _names(seqs, len(lst))
    if positions is not None and len(positions) != len(lst):
        raise ValueError("positions must have one entry per sequence")
    for m in mots:
        ppm(m)  # validate early
    out, rows, short = [], [], False
    for i, s in enumerate(lst):
        L = len(s)
        if positions is not None:
            n_i = len(positions[i])
        elif rate is not None:
            n_i = int(gen.poisson(rate * L))
        else:
            n_i = int(n_per_seq)
        buf = list(s)
        placed: list[tuple[int, int]] = []
        for j in range(n_i):
            mi = 0 if len(mots) == 1 else int(gen.integers(len(mots)))
            w = widths[mi]
            if w > L:
                continue
            st = "+" if strand == "+" or gen.integers(2) == 0 else "-"
            if positions is not None:
                start = int(positions[i][j])
                if start < 0 or start + w > L:
                    warnings.warn(f"dropping out-of-bounds position {start} in sequence {names[i]}",
                                  stacklevel=2)
                    continue
            else:
                for _ in range(max_retries):
                    u = gen.random() if centre_bias == 1 else gen.random(centre_bias).mean()
                    start = min(int(np.floor(u * (L - w + 1))), L - w)
                    if not any(start <= e + min_spacing - 1 and start + w + min_spacing - 1 >= b
                               for b, e in placed):
                        break
                else:
                    short = True
                    continue
            site = sample_sites(mots[mi], 1, rng=gen)[0]
            if st == "-":
                site = revcomp(site)
            buf[start : start + w] = site
            placed.append((start, start + w))
            rows.append((names[i], labels[mi], start, start + w, st, site))
        out.append("".join(buf))
    if short:
        warnings.warn("some implants could not be placed within max_retries attempts", stacklevel=2)
    table = pd.DataFrame(rows, columns=["seq", "motif", "start", "end", "strand", "site"])
    return pack(out), table
