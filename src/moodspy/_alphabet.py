"""DNA alphabet encoding shared by all modules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ALPHABET = "ACGT"
#: Code used for every character outside ``ACGTacgt`` (N, IUPAC codes, gaps...).
INVALID = 4

_LUT = np.full(256, INVALID, dtype=np.uint8)
for _i, _c in enumerate(ALPHABET):
    _LUT[ord(_c)] = _i
    _LUT[ord(_c.lower())] = _i


def encode(seq: str | bytes) -> np.ndarray:
    """Encode one sequence as uint8 codes (A=0, C=1, G=2, T=3, other=4)."""
    if isinstance(seq, str):
        seq = seq.encode("ascii", errors="replace")
    return _LUT[np.frombuffer(seq, dtype=np.uint8)]


def encode_many(seqs: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Encode sequences into one concatenated code array plus offsets.

    Sequence ``i`` occupies ``codes[offsets[i]:offsets[i + 1]]``.
    """
    lengths = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    # "replace" keeps one byte per character, so offsets stay valid.
    codes = encode("".join(seqs))
    return codes, offsets


def base_counts(codes: np.ndarray) -> np.ndarray:
    """Count A, C, G, T in an encoded array (other characters are ignored)."""
    return np.bincount(codes, minlength=INVALID + 1)[:INVALID].astype(np.float64)
