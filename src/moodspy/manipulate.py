"""Motif manipulation: information content, IUPAC consensus, trimming,
creating, shuffling and sampling motifs.

A port of the motif utilities of the R/Bioconductor package
`universalmotif <https://github.com/bjmt/universalmotif>`_. Where this
module turns a motif into probabilities it follows universalmotif rather
than motifmatchr (see [`ppm`][moodspy.manipulate.ppm]): count matrices get a
pseudocount of 1 by default, probability matrices are used as they are.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from ._alphabet import ALPHABET, encode
from .motifs import Motif, as_motif, as_motif_list, check_background

#: universalmotif's default motif pseudocount (its ``pseudocount`` slot).
UM_PSEUDOCOUNT = 1.0

RngLike = int | np.random.Generator | None

_IUPAC = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "M": "AC", "K": "GT", "S": "CG", "W": "AT",
    "H": "ACT", "B": "CGT", "V": "ACG", "D": "AGT", "N": "ACGT",
}
# universalmotif's consensus_to_ppm(): a single IUPAC letter as a probability column.
_IUPAC_PPM = {
    1: (0.997, 0.001),
    2: (0.499, 0.001),
    3: (0.333, 0.001),
    4: (0.25, 0.25),
}


def as_rng(rng: RngLike) -> np.random.Generator:
    """Turn ``None``, a seed or a Generator into a ``numpy.random.Generator``."""
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


# ---------------------------------------------------------------------- probabilities
def ppm(motif: Any, pseudocount: float = UM_PSEUDOCOUNT) -> np.ndarray:
    """Position probability matrix ``(4, length)`` as universalmotif builds it.

    Counts are smoothed with ``(n + pseudocount / 4) / (N + pseudocount)``
    (the same formula as [`Motif.probabilities`][moodspy.Motif.probabilities],
    but universalmotif's default pseudocount is 1, not 0.8). Probability
    matrices are only renormalised, without a pseudocount. Log-odds matrices
    are converted back with their own background.
    """
    m = as_motif(motif)
    if m.kind == "counts":
        mat = m.matrix
        if pseudocount > 0:
            return (mat + pseudocount / 4) / (mat.sum(axis=0, keepdims=True) + pseudocount)
        colsum = mat.sum(axis=0, keepdims=True)
        colsum[colsum == 0] = 1.0
        return mat / colsum
    if m.kind == "probability":
        # Used as written (rounded MEME columns sum to 1 +- 1e-6), like
        # universalmotif; only clearly unnormalised columns are rescaled.
        colsum = m.matrix.sum(axis=0, keepdims=True)
        off = np.abs(colsum - 1.0) > 1e-3
        colsum[~off | (colsum == 0)] = 1.0
        return m.matrix / colsum
    return m.probabilities()


def motif_nsites(motif: Motif) -> float:
    """Number of sites behind a motif (universalmotif uses 100 when unknown)."""
    if motif.nsites and motif.nsites > 1:
        return float(motif.nsites)
    if motif.kind == "counts":
        n = float(motif.matrix.sum(axis=0).max())
        if n > 1:
            return n
    return 100.0


def _ic_columns(p: np.ndarray, background: np.ndarray, relative_entropy: bool) -> np.ndarray:
    """Per-letter information content heights of a ``(4, L)`` probability matrix."""
    if relative_entropy:
        with np.errstate(divide="ignore", invalid="ignore"):
            h = p * np.log2(p / background[:, None])
        h[~np.isfinite(h)] = 0.0
        return np.maximum(h, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.where(p > 0, -p * np.log2(p), 0.0).sum(axis=0)
    return p * (2.0 - ent)


def icm(
    motif: Any,
    pseudocount: float = UM_PSEUDOCOUNT,
    background: Any = None,
    relative_entropy: bool = False,
) -> np.ndarray:
    """Information content matrix ``(4, length)``: letter heights in bits, as in a logo.

    Each column is the probability column scaled by its information content,
    ``2 - H`` bits with ``H`` the Shannon entropy. With
    ``relative_entropy=True`` the heights are ``p * log2(p / background)``
    (negative heights set to 0), so a column that matches the background
    has no information.
    """
    m = as_motif(motif)
    bg = m.background if background is None else check_background(background)
    return _ic_columns(ppm(m, pseudocount), bg, relative_entropy)


def motif_ic(
    motif: Any,
    pseudocount: float = UM_PSEUDOCOUNT,
    background: Any = None,
    relative_entropy: bool = False,
) -> np.ndarray:
    """Information content of each position, in bits (column sums of [`icm`][moodspy.icm])."""
    return icm(motif, pseudocount, background, relative_entropy).sum(axis=0)


def average_ic(
    motifs: Any,
    average: Literal["a.mean", "g.mean", "median", "fzt"] = "a.mean",
    pseudocount: float = UM_PSEUDOCOUNT,
) -> np.ndarray:
    """Average information content per position of each motif.

    universalmotif's ``average_ic``. Handy for dropping low-information
    motifs before [`compare_motifs`][moodspy.compare_motifs], whose
    ``min_mean_ic`` filter otherwise returns NaN for them.
    """
    from .compare import _average

    return np.array([_average(motif_ic(m, pseudocount), average) for m in as_motif_list(motifs)])


# ---------------------------------------------------------------------- consensus
def _consensus_letter(p: Sequence[float]) -> str:
    a, c, g, t = p
    if a > 0.5 and c <= 0.25 and g <= 0.25 and t <= 0.25:
        return "A"
    if c > 0.5 and a <= 0.25 and g <= 0.25 and t <= 0.25:
        return "C"
    if g > 0.5 and a <= 0.25 and c <= 0.25 and t <= 0.25:
        return "G"
    if t > 0.5 and a <= 0.25 and c <= 0.25 and g <= 0.25:
        return "T"
    for letter, s in (("M", a + c), ("R", a + g), ("W", a + t), ("S", c + g), ("Y", c + t), ("K", g + t)):
        if s > 0.75:
            return letter
    if a > 0.25 and c > 0.25 and t > 0.25:
        return "H"
    if c > 0.25 and g > 0.25 and t > 0.25:
        return "B"
    if a > 0.25 and c > 0.25 and g > 0.25:
        return "V"
    if a > 0.25 and g > 0.25 and t > 0.25:
        return "D"
    return "N"


def iupac_consensus(motif: Any, pseudocount: float = UM_PSEUDOCOUNT) -> str:
    """IUPAC consensus string, with universalmotif's ``get_consensus`` rules.

    A position gets a single letter when its probability is above 0.5 and
    every other letter is at most 0.25; otherwise a two-letter code when a
    pair sums to more than 0.75; otherwise a three-letter code when three
    letters are each above 0.25; otherwise ``N``.
    """
    return "".join(_consensus_letter(col) for col in ppm(motif, pseudocount).T)


# ---------------------------------------------------------------------- trimming
def trim_motifs(
    motifs: Any,
    min_ic: float = 0.25,
    trim_from: Literal["both", "left", "right"] = "both",
    pseudocount: float = UM_PSEUDOCOUNT,
) -> list[Motif]:
    """Remove low-information positions from the edges of motifs.

    Edge positions with information content below ``min_ic`` bits are cut
    until a position at or above it is reached. Internal low-information
    positions are kept. Motifs that are trimmed away completely are dropped
    with a warning.

    Parameters
    ----------
    motifs
        Anything [`match_motifs`][moodspy.match_motifs] accepts as motifs.
    min_ic
        Minimum information content (bits) for an edge position to stay.
    trim_from
        ``"both"``, ``"left"`` or ``"right"``.
    pseudocount
        Pseudocount for count matrices when computing the information content.

    Returns
    -------
    list of Motif
        Trimmed copies with their original kind (counts stay counts).
    """
    if trim_from not in ("both", "left", "right"):
        raise ValueError("trim_from must be 'both', 'left' or 'right'")
    out, dropped = [], []
    for m in as_motif_list(motifs):
        ic = motif_ic(m, pseudocount)
        left, right = 0, len(ic)
        if trim_from in ("both", "left"):
            while left < right and ic[left] < min_ic:
                left += 1
        if trim_from in ("both", "right"):
            while right > left and ic[right - 1] < min_ic:
                right -= 1
        if right <= left:
            dropped.append(m.label or "<unnamed>")
            continue
        new = m.copy()
        new.matrix = np.ascontiguousarray(m.matrix[:, left:right])
        out.append(new)
    if dropped:
        if not out:
            raise ValueError("all motifs were completely trimmed")
        warnings.warn(f"motifs completely trimmed and dropped: {', '.join(dropped)}", stacklevel=2)
    return out


# ---------------------------------------------------------------------- creating
def create_motif(
    consensus: str | Sequence[str] | int | None = None,
    *,
    nsites: float = 100.0,
    background: Any = None,
    name: str = "",
    id: str | None = None,
    rng: RngLike = None,
) -> Motif:
    """Make a motif from a consensus string, a set of sites or at random.

    Parameters
    ----------
    consensus
        * an IUPAC string such as ``"TGASTCA"``: each letter becomes a
          probability column (``A`` -> 0.997/0.001/0.001/0.001, two-letter
          codes -> 0.499 each, three-letter codes -> 0.333 each, ``N`` ->
          uniform), as universalmotif's ``create_motif`` does;
        * a list of equal-length sites: their base counts (non-ACGT
          characters are skipped);
        * an integer: a random motif of that length, each column drawn from
          a Dirichlet distribution with concentration ``background``;
        * ``None``: a random motif of length 10.
    nsites
        Number of sites stored with probability motifs.
    background
        Dirichlet concentration for random motifs (default uniform 0.25),
        i.e. columns are sparse and informative.
    rng
        Seed or ``numpy.random.Generator`` for random motifs.
    """
    if consensus is None:
        consensus = 10
    if isinstance(consensus, (int, np.integer)):
        if consensus < 1:
            raise ValueError("motif length must be at least 1")
        bg = np.full(4, 0.25) if background is None else check_background(background)
        mat = as_rng(rng).dirichlet(bg, size=int(consensus)).T
        return Motif.from_probabilities(mat, name=name, id=id, nsites=nsites)
    if isinstance(consensus, str):
        cols = []
        for ch in consensus.upper():
            if ch not in _IUPAC:
                raise ValueError(f"{ch!r} is not an IUPAC nucleotide code")
            hi, lo = _IUPAC_PPM[len(_IUPAC[ch])]
            cols.append([hi if b in _IUPAC[ch] else lo for b in ALPHABET])
        if not cols:
            raise ValueError("empty consensus")
        return Motif.from_probabilities(np.array(cols).T, name=name, id=id, nsites=nsites)
    sites = [str(s) for s in consensus]
    if not sites:
        raise ValueError("no sites given")
    if len({len(s) for s in sites}) != 1:
        raise ValueError("all sites must have the same length")
    codes = np.array([encode(s) for s in sites])
    counts = np.stack([(codes == i).sum(axis=0) for i in range(4)]).astype(np.float64)
    return Motif.from_counts(counts, name=name, id=id)


# ---------------------------------------------------------------------- randomising
def shuffle_motifs(motifs: Any, k: int = 2, rng: RngLike = None) -> list[Motif]:
    """Shuffle the columns of motifs, keeping blocks of ``k`` adjacent columns together.

    All columns of all motifs are pooled, shuffled and dealt back out, so
    each new motif keeps its length and the pool keeps its composition. This
    is universalmotif's ``shuffle_motifs(method = "linear")``: with ``k > 1``
    the columns are cut into consecutive blocks of ``k`` (plus a remainder),
    and the blocks are shuffled. Use the result as a null set of motifs with
    realistic column statistics.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    mots = as_motif_list(motifs)
    gen = as_rng(rng)
    probs = [ppm(m) for m in mots]
    pool = np.concatenate(probs, axis=1)
    n = pool.shape[1]
    if k == 1:
        order = gen.permutation(n)
    else:
        # The n % k leftover columns stay at the ends, split between left and
        # right as universalmotif's shuffle_linear() does.
        rem = n % k
        left = 1 if rem == 1 else int(np.floor((rem + 0.1) / 2 + 0.5)) if rem else 0
        right = rem - left
        middle = np.arange(left, n - right).reshape(-1, k)
        order = np.concatenate([np.arange(left), middle[gen.permutation(len(middle))].ravel(),
                                np.arange(n - right, n)])
    pool = pool[:, order]
    out, pos = [], 0
    for m, p in zip(mots, probs):
        width = p.shape[1]
        new = Motif.from_probabilities(
            pool[:, pos : pos + width], name=f"{m.name or m.label} [shuffled]",
            nsites=motif_nsites(m), background=None,
        )
        out.append(new)
        pos += width
    return out


def sample_sites(motif: Any, n: int = 100, rng: RngLike = None) -> list[str]:
    """Draw ``n`` sites from a motif, each position independently from its column."""
    p = ppm(as_motif(motif))
    gen = as_rng(rng)
    cum = np.cumsum(p, axis=0)
    cum[-1] = 1.0
    u = gen.random((n, p.shape[1]))
    idx = (u[:, None, :] > cum[None, :, :]).sum(axis=1)
    letters = np.frombuffer(ALPHABET.encode(), dtype=np.uint8)[idx]
    return [row.tobytes().decode() for row in letters]


def round_motif(motif: Any, pct_tolerance: float = 0.05) -> Motif:
    """Set probabilities below ``pct_tolerance`` to zero and renormalise.

    Returns a probability motif (universalmotif converts back to the
    original type; here use ``Motif.log_odds`` etc. as needed).
    """
    if not 0 <= pct_tolerance <= 1:
        raise ValueError("pct_tolerance must be between 0 and 1")
    m = as_motif(motif)
    p = ppm(m, pseudocount=0.0)
    p = np.where(p < pct_tolerance, 0.0, p)
    p = p / p.sum(axis=0, keepdims=True)
    out = Motif.from_probabilities(p, name=m.name, id=m.id, nsites=motif_nsites(m), meta=m.meta)
    out.label = m._label
    return out


def score_match(motif: Any, sites: str | Iterable[str], background: Any = None,
                pseudocount: float = 0.8) -> np.ndarray:
    """Log2-odds score of sites with the motif's length (same scores as the scanner).

    Uses [`Motif.log_odds`][moodspy.Motif.log_odds], so it agrees with the
    scores [`match_motifs`][moodspy.match_motifs] reports for the ``+``
    strand. Sites containing characters other than ``ACGT`` score ``-inf``.
    """
    m = as_motif(motif)
    pwm = m.log_odds(background, pseudocount)
    single = isinstance(sites, str)
    seqs = [sites] if single else list(sites)
    out = np.empty(len(seqs))
    cols = np.arange(m.length)
    for i, s in enumerate(seqs):
        if len(s) != m.length:
            raise ValueError(f"site {s!r} has length {len(s)}, motif has length {m.length}")
        codes = encode(s)
        out[i] = -np.inf if (codes > 3).any() else pwm[codes, cols].sum()
    return out


# ---------------------------------------------------------------------- summary
def motif_table(motifs: Any, pseudocount: float = UM_PSEUDOCOUNT) -> pd.DataFrame:
    """One row per motif: label, name, id, kind, length, nsites, total and mean IC, consensus."""
    from .motifs import motif_labels

    mots = as_motif_list(motifs)
    rows = []
    for m in mots:
        ic = motif_ic(m, pseudocount)
        rows.append({
            "name": m.name, "id": m.id, "kind": m.kind, "length": m.length,
            "nsites": motif_nsites(m), "ic": float(ic.sum()), "mean_ic": float(ic.mean()),
            "consensus": iupac_consensus(m, pseudocount),
        })
    return pd.DataFrame(rows, index=pd.Index(motif_labels(mots), name="motif"))
