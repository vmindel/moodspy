"""Position frequency / weight matrix model.

All public matrices use the ``(4, length)`` orientation with rows in
``A, C, G, T`` order, as in JASPAR files, MOODS and motifmatchr.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import numpy as np

from ._alphabet import ALPHABET

MatrixKind = Literal["counts", "probability", "log-odds"]

#: TFBSTools / motifmatchr default pseudocount for count matrices.
DEFAULT_PSEUDOCOUNT = 0.8
#: Number of sites assumed for probability matrices with unknown ``nsites``
#: (the MEME format default).
DEFAULT_NSITES = 20.0

_UNIFORM = np.full(4, 0.25)


class Motif:
    """A DNA motif stored as counts, probabilities or log-odds scores.

    Parameters
    ----------
    matrix
        ``(4, length)`` array with rows ``A, C, G, T``. A ``(length, 4)``
        array is transposed automatically (a 4x4 matrix is assumed to be
        ``(4, length)`` already).
    name
        Human-readable name, e.g. ``"KLF5"``.
    id
        Database identifier, e.g. ``"MA0599.1"``. Used as the motif's label
        in results when present.
    kind
        ``"counts"``, ``"probability"``, ``"log-odds"`` or ``"auto"`` to
        infer it the way motifmatchr's ``pwmType`` does.
    nsites
        Number of aligned sites behind a probability matrix; used to scale
        the pseudocount. Defaults to `DEFAULT_NSITES`.
    log_base
        Logarithm base of a log-odds matrix.
    background
        Background a log-odds matrix was computed against (default uniform).
    """

    __slots__ = ("matrix", "name", "id", "kind", "nsites", "log_base", "background", "meta", "_label")

    def __init__(
        self,
        matrix: Any,
        name: str = "",
        id: str | None = None,
        *,
        kind: MatrixKind | Literal["auto"] = "auto",
        nsites: float | None = None,
        log_base: float = 2.0,
        background: Any = None,
        meta: dict | None = None,
    ):
        mat = _as_4xl(matrix)
        if kind == "auto":
            kind = infer_kind(mat)
        if kind not in ("counts", "probability", "log-odds"):
            raise ValueError(f"unknown matrix kind {kind!r}")
        if kind != "log-odds" and (mat < 0).any():
            raise ValueError(f"a {kind} matrix cannot contain negative values")
        self.matrix = mat
        self.name = str(name) if name is not None else ""
        self.id = str(id) if id is not None else None
        self.kind: MatrixKind = kind
        self.nsites = None if nsites is None else float(nsites)
        self.log_base = float(log_base)
        self.background = _UNIFORM.copy() if background is None else check_background(background)
        self.meta = dict(meta or {})
        self._label: str | None = None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_counts(cls, counts: Any, name: str = "", id: str | None = None, **kw) -> Motif:
        return cls(counts, name, id, kind="counts", **kw)

    @classmethod
    def from_probabilities(
        cls, probs: Any, name: str = "", id: str | None = None, nsites: float | None = None, **kw
    ) -> Motif:
        return cls(probs, name, id, kind="probability", nsites=nsites, **kw)

    @classmethod
    def from_log_odds(
        cls,
        pwm: Any,
        name: str = "",
        id: str | None = None,
        log_base: float = 2.0,
        background: Any = None,
        **kw,
    ) -> Motif:
        return cls(pwm, name, id, kind="log-odds", log_base=log_base, background=background, **kw)

    @classmethod
    def from_biopython(cls, motif: Any) -> Motif:
        """Convert a ``Bio.motifs.Motif`` (e.g. from pyjaspar) to a Motif."""
        counts = np.array([motif.counts[b] for b in ALPHABET], dtype=np.float64)
        mid = getattr(motif, "matrix_id", None)
        return cls(counts, getattr(motif, "name", "") or "", mid, kind="counts")

    # ------------------------------------------------------------------ properties
    @property
    def length(self) -> int:
        return self.matrix.shape[1]

    def __len__(self) -> int:
        return self.length

    @property
    def label(self) -> str:
        """``"<id>_<name>"`` when both exist, otherwise whichever exists.

        Can be overridden by assignment.
        """
        if self._label:
            return self._label
        if self.id and self.name and self.name != self.id:
            return f"{self.id}_{self.name}"
        return self.id or self.name or ""

    @label.setter
    def label(self, value: str | None) -> None:
        self._label = value

    @property
    def consensus(self) -> str:
        return "".join(ALPHABET[i] for i in self.matrix.argmax(axis=0))

    def __repr__(self) -> str:
        label = self.label or "<unnamed>"
        return f"Motif({label!r}, length={self.length}, kind={self.kind!r}, consensus={self.consensus!r})"

    # ------------------------------------------------------------------ conversions
    def probabilities(self, pseudocount: float = DEFAULT_PSEUDOCOUNT) -> np.ndarray:
        """Column-normalised probabilities with a uniform pseudocount prior.

        For counts this is TFBSTools' ``toPWM(type="prob")``:
        ``(n + 0.25 * pseudocount) / (N + pseudocount)`` per column.
        """
        if self.kind == "counts":
            return _smooth(self.matrix, pseudocount)
        if self.kind == "probability":
            nsites = self.nsites or DEFAULT_NSITES
            colsum = self.matrix.sum(axis=0, keepdims=True)
            colsum[colsum == 0] = 1.0
            return _smooth(self.matrix / colsum * nsites, pseudocount)
        p = self.background[:, None] * np.power(self.log_base, self.matrix)
        return p / p.sum(axis=0, keepdims=True)

    def log_odds(
        self,
        background: Any = None,
        pseudocount: float = DEFAULT_PSEUDOCOUNT,
        log_base: float = 2.0,
        motifmatchr_compat: bool = False,
    ) -> np.ndarray:
        """Log-odds score matrix ``log(p / background)`` (default uniform).

        Counts and probabilities are smoothed with [`probabilities`][moodspy.Motif.probabilities].
        Log-odds input is re-centred on the new background and rescaled to
        ``log_base``.

        ``motifmatchr_compat=True`` reproduces motifmatchr's
        ``convert_pwm``, which applies the background correction with the
        wrong sign (``log2(p * bg / 0.0625)`` instead of ``log2(p / bg)``
        for count matrices). It is identical for a uniform background.
        """
        bg = _UNIFORM if background is None else check_background(background)
        if self.kind == "log-odds":
            shift = (np.log(self.background) - np.log(bg)) / np.log(self.log_base)
            sign = -1.0 if motifmatchr_compat else 1.0
            out = self.matrix + sign * shift[:, None]
            return out * (np.log(self.log_base) / np.log(log_base))
        p = self.probabilities(pseudocount)
        if motifmatchr_compat:
            # Same operations, in the same order, as TFBSTools::toPWM followed
            # by motifmatchr:::convert_pwm, so thresholds round identically.
            out = np.log2(p / 0.25) - (np.log2(0.25) - np.log2(bg))[:, None]
            return out if log_base == 2.0 else out * (np.log(2.0) / np.log(log_base))
        if log_base == 2.0:
            return np.log2(p / bg[:, None])
        return np.log(p / bg[:, None]) / np.log(log_base)

    def reverse_complement(self) -> Motif:
        out = self.copy()
        out.matrix = self.matrix[::-1, ::-1].copy()
        if self.kind == "log-odds":
            out.background = self.background[::-1].copy()
        return out

    def copy(self) -> Motif:
        out = Motif(
            self.matrix.copy(),
            self.name,
            self.id,
            kind=self.kind,
            nsites=self.nsites,
            log_base=self.log_base,
            background=self.background,
            meta=self.meta,
        )
        out._label = self._label
        return out


def _smooth(counts: np.ndarray, pseudocount: float) -> np.ndarray:
    return (counts + 0.25 * pseudocount) / (counts.sum(axis=0, keepdims=True) + pseudocount)


def _as_4xl(matrix: Any) -> np.ndarray:
    if isinstance(matrix, Mapping):
        matrix = [matrix[b] for b in ALPHABET]
    if hasattr(matrix, "loc") and hasattr(matrix, "columns"):  # pandas DataFrame
        cols = [str(c).upper() for c in matrix.columns]
        if set(ALPHABET) <= set(cols):
            matrix = matrix.to_numpy().T[[cols.index(b) for b in ALPHABET]]
        else:
            idx = [str(i).upper() for i in matrix.index]
            if set(ALPHABET) <= set(idx):
                matrix = matrix.to_numpy()[[idx.index(b) for b in ALPHABET]]
            else:
                matrix = matrix.to_numpy()
    mat = np.array(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"motif matrix must be 2-D, got shape {mat.shape}")
    if mat.shape[0] != 4 and mat.shape[1] == 4:
        mat = mat.T
    if mat.shape[0] != 4:
        raise ValueError(f"motif matrix must have 4 rows (A, C, G, T), got shape {mat.shape}")
    if mat.shape[1] == 0:
        raise ValueError("motif matrix has no columns")
    if not np.isfinite(mat).all():
        raise ValueError("motif matrix contains NaN or infinite values")
    return np.ascontiguousarray(mat)


def infer_kind(mat: np.ndarray) -> MatrixKind:
    """Guess whether a matrix holds counts, probabilities or log-odds scores."""
    if (mat < 0).any():
        return "log-odds"
    if np.allclose(mat.sum(axis=0), 1.0, atol=1e-3):
        return "probability"
    return "counts"


def check_background(bg: Any) -> np.ndarray:
    """Validate a background as 4 positive frequencies in A, C, G, T order.

    Accepts a sequence of 4 numbers or a mapping keyed by nucleotide.
    Frequencies summing to within 1% of 1 are renormalised.
    """
    if isinstance(bg, Mapping):
        keys = {str(k).upper(): v for k, v in bg.items()}
        missing = set(ALPHABET) - set(keys)
        if missing:
            raise ValueError(f"background is missing nucleotides {sorted(missing)}")
        bg = [keys[b] for b in ALPHABET]
    arr = np.asarray(bg, dtype=np.float64).ravel()
    if arr.shape != (4,):
        raise ValueError("background must have exactly 4 values (A, C, G, T)")
    if not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError("background frequencies must be positive")
    if abs(arr.sum() - 1.0) > 0.01:
        raise ValueError(f"background frequencies must sum to 1 (got {arr.sum():.4g})")
    return arr / arr.sum()


def as_motif(obj: Any, name: str | None = None) -> Motif:
    """Coerce a Motif, Biopython motif or raw matrix into a [`Motif`][moodspy.Motif]."""
    if isinstance(obj, Motif):
        return obj
    if hasattr(obj, "counts") and hasattr(obj, "alphabet"):  # Bio.motifs.Motif
        return Motif.from_biopython(obj)
    return Motif(obj, name or "", None)


def as_motif_list(motifs: Any) -> list[Motif]:
    """Coerce motif input into a list of [`Motif`][moodspy.Motif].

    Accepts a single Motif / matrix, an iterable of them, a mapping of
    label -> matrix/Motif, or a path to a motif file.
    """
    import os

    if isinstance(motifs, (str, os.PathLike)):
        from .io import read_motifs

        return read_motifs(motifs)
    if isinstance(motifs, Motif) or (hasattr(motifs, "counts") and hasattr(motifs, "alphabet")):
        return [as_motif(motifs)]
    if isinstance(motifs, np.ndarray) and motifs.ndim == 2:
        return [as_motif(motifs, "motif_0")]
    if isinstance(motifs, Mapping):
        out = []
        for key, value in motifs.items():
            m = as_motif(value, str(key)).copy()
            m.label = str(key)
            out.append(m)
        if not out:
            raise ValueError("no motifs given")
        return out
    if isinstance(motifs, Iterable):
        out = [as_motif(m, f"motif_{i}") for i, m in enumerate(motifs)]
        if not out:
            raise ValueError("no motifs given")
        return out
    raise TypeError(f"cannot interpret {type(motifs).__name__} as motifs")


def motif_labels(motifs: list[Motif]) -> list[str]:
    """Unique labels for a list of motifs (duplicates get ``.1``, ``.2``...)."""
    labels, seen = [], {}
    for i, m in enumerate(motifs):
        lab = m.label or f"motif_{i}"
        if lab in seen:
            seen[lab] += 1
            lab = f"{lab}.{seen[lab]}"
        else:
            seen[lab] = 0
        labels.append(lab)
    return labels
