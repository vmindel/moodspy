"""Genome access: indexed FASTA reading and region extraction.

:class:`FastaFile` is a small, dependency-free, memory-mapped reader for
uncompressed FASTA files with a samtools ``.fai`` index (built on the fly
if missing). Other genome objects work too: anything with
``fetch(chrom, start, end)`` (pysam.FastaFile), ``genome[chrom][start:end]``
(pyfaidx.Fasta, or a plain ``dict`` of chromosome -> sequence).
"""

from __future__ import annotations

import mmap
import os
import re
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from ._alphabet import _LUT, INVALID


class FastaFile:
    """Random access to an uncompressed, indexed FASTA file.

    Parameters
    ----------
    path
        FASTA file. ``<path>.fai`` is used if present, otherwise an index
        is built in memory (and written next to the file when possible).
    """

    def __init__(self, path: str | os.PathLike, write_index: bool = True):
        self.path = os.fspath(path)
        if self.path.endswith((".gz", ".bgz", ".bz2", ".xz")):
            raise ValueError(
                f"{self.path} is compressed; decompress it first or pass a pysam.FastaFile / "
                "pyfaidx.Fasta object as the genome"
            )
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        fai = self.path + ".fai"
        if os.path.exists(fai) and os.path.getmtime(fai) >= os.path.getmtime(self.path):
            self.index = _read_fai(fai)
        else:
            self.index = _build_fai(self._mm)
            if write_index:
                try:
                    _write_fai(self.index, fai)
                except OSError:
                    pass
        self._bg: np.ndarray | None = None

    def close(self) -> None:
        self._mm.close()
        self._fh.close()

    def __enter__(self) -> FastaFile:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"FastaFile({self.path!r}, {len(self.index)} sequences)"

    # -- mapping-ish interface
    def keys(self) -> list[str]:
        return list(self.index)

    def __contains__(self, chrom: str) -> bool:
        return chrom in self.index

    def __len__(self) -> int:
        return len(self.index)

    @property
    def lengths(self) -> dict[str, int]:
        return {k: v[0] for k, v in self.index.items()}

    def fetch(self, chrom: str, start: int = 0, end: int | None = None) -> str:
        """Sequence of ``chrom[start:end]`` (0-based, half-open)."""
        try:
            length, offset, line_bases, line_width = self.index[chrom]
        except KeyError:
            raise KeyError(f"sequence {chrom!r} not found in {self.path}") from None
        end = length if end is None else end
        if start < 0 or end > length or start > end:
            raise ValueError(f"region {chrom}:{start}-{end} is outside 0-{length}")

        def at(p: int) -> int:
            return offset + (p // line_bases) * line_width + p % line_bases

        raw = self._mm[at(start) : at(end)]
        if line_width != line_bases:
            raw = raw.replace(b"\n", b"").replace(b"\r", b"")
        return raw.decode("ascii")

    def __getitem__(self, chrom: str) -> str:
        return self.fetch(chrom)

    def base_counts(self) -> np.ndarray:
        """Genome-wide A, C, G, T counts (computed once, then cached)."""
        if self._bg is None:
            counts = np.zeros(INVALID + 1, dtype=np.int64)
            step = 1 << 26
            for chrom, (length, *_) in self.index.items():
                for s in range(0, length, step):
                    seq = self.fetch(chrom, s, min(s + step, length)).encode("ascii")
                    counts += np.bincount(_LUT[np.frombuffer(seq, np.uint8)], minlength=INVALID + 1)
            self._bg = counts[:INVALID].astype(np.float64)
        return self._bg


def _read_fai(path: str) -> dict[str, tuple[int, int, int, int]]:
    index = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                name, length, offset, lb, lw = line.split("\t")[:5]
                index[name] = (int(length), int(offset), int(lb), int(lw))
    return index


def _write_fai(index: dict, path: str) -> None:
    with open(path, "w") as fh:
        for name, (length, offset, lb, lw) in index.items():
            fh.write(f"{name}\t{length}\t{offset}\t{lb}\t{lw}\n")


def _build_fai(mm: mmap.mmap) -> dict[str, tuple[int, int, int, int]]:
    index: dict[str, tuple[int, int, int, int]] = {}
    pos, size = 0, len(mm)
    name = None
    length = offset = lb = lw = 0
    irregular = False
    while pos < size:
        nl = mm.find(b"\n", pos)
        nl = size if nl < 0 else nl
        line = mm[pos:nl]
        nxt = nl + 1
        if line.startswith(b">"):
            if name is not None:
                index[name] = (length, offset, lb, lw)
            name = line[1:].split()[0].decode() if line[1:].split() else ""
            if name in index:
                raise ValueError(f"duplicate sequence name {name!r} in FASTA")
            length, offset, lb, lw, irregular = 0, nxt, 0, 0, False
        elif name is not None:
            bases = len(line.rstrip(b"\r"))
            if (irregular and bases) or (lb and bases > lb):
                raise ValueError(f"sequence {name!r} has lines of unequal length; cannot index")
            if lb == 0:
                lb, lw = bases, nl + 1 - pos
            elif bases != lb:
                irregular = True  # allowed only on the last line
            length += bases
        pos = nxt
    if name is not None:
        index[name] = (length, offset, lb, lw)
    return index


# ---------------------------------------------------------------------- genome adapters
class _Genome:
    """Uniform ``fetch`` / ``base_counts`` over the supported genome types."""

    def __init__(self, genome: Any):
        if isinstance(genome, (str, os.PathLike)):
            genome = FastaFile(genome)
        self.obj = genome

    def fetch(self, chrom: str, start: int, end: int) -> str:
        g = self.obj
        if isinstance(g, FastaFile):
            return g.fetch(chrom, start, end)
        if hasattr(g, "fetch"):  # pysam.FastaFile
            return g.fetch(chrom, start, end)
        seq = g[chrom]
        if start < 0 or end > len(seq):
            raise ValueError(f"region {chrom}:{start}-{end} is outside 0-{len(seq)}")
        return str(seq[start:end])

    def fetch_many(self, chroms, starts, ends) -> list[str]:
        return [self.fetch(c, int(s), int(e)) for c, s, e in zip(chroms, starts, ends)]

    def sequences(self) -> Iterator[str]:
        g = self.obj
        if hasattr(g, "references"):  # pysam
            for name in g.references:
                yield g.fetch(name)
        else:
            for name in g.keys():
                yield str(g[name])

    def base_counts(self) -> np.ndarray:
        if isinstance(self.obj, FastaFile):
            return self.obj.base_counts()
        counts = np.zeros(4)
        for seq in self.sequences():
            counts += np.bincount(_LUT[np.frombuffer(seq.encode("ascii", "replace"), np.uint8)],
                                  minlength=INVALID + 1)[:INVALID]
        return counts


# ---------------------------------------------------------------------- regions
_COLUMN_ALIASES = {
    "chrom": ("chrom", "chr", "chromosome", "seqnames", "seqname", "contig", "#chrom"),
    "start": ("start", "chromstart", "begin"),
    "end": ("end", "chromend", "stop"),
}
_REGION_RE = re.compile(r"^(?P<chrom>.+?)[:_-](?P<start>\d[\d,]*)[-_](?P<end>\d[\d,]*)$")


def regions_frame(regions: Any) -> pd.DataFrame:
    """Normalise genomic regions to a DataFrame with ``chrom, start, end``.

    Accepts a DataFrame with chrom/start/end columns (common aliases and
    pyranges' ``Chromosome/Start/End`` work), a pyranges object, a BED file
    path, or region strings such as ``"chr1:100-200"`` / ``"chr1-100-200"``.
    Coordinates are 0-based, half-open (BED convention).
    """
    if hasattr(regions, "df") and not isinstance(regions, pd.DataFrame):  # pyranges < 1
        regions = regions.df
    if isinstance(regions, (str, os.PathLike)):
        return read_bed(regions)
    if isinstance(regions, pd.DataFrame):
        lower = {str(c).lower(): c for c in regions.columns}
        cols = {}
        for key, aliases in _COLUMN_ALIASES.items():
            found = next((lower[a] for a in aliases if a in lower), None)
            if found is None:
                raise ValueError(f"regions need a {key!r} column (got {list(regions.columns)})")
            cols[key] = found
        out = pd.DataFrame(
            {
                "chrom": regions[cols["chrom"]].astype(str).to_numpy(),
                "start": regions[cols["start"]].to_numpy(dtype=np.int64),
                "end": regions[cols["end"]].to_numpy(dtype=np.int64),
            },
            index=regions.index,
        )
        if isinstance(out.index, pd.RangeIndex):
            out.index = region_names(out)
        return out
    strings = [str(r) for r in regions]
    parsed = [_REGION_RE.match(s) for s in strings]
    bad = [s for s, m in zip(strings, parsed) if m is None]
    if bad:
        raise ValueError(f"cannot parse region strings like {bad[:3]}")
    return pd.DataFrame(
        {
            "chrom": [m["chrom"] for m in parsed],
            "start": [int(m["start"].replace(",", "")) for m in parsed],
            "end": [int(m["end"].replace(",", "")) for m in parsed],
        },
        index=pd.Index(strings),
    )


def region_names(df: pd.DataFrame) -> pd.Index:
    return pd.Index(df["chrom"].astype(str) + ":" + df["start"].astype(str) + "-" + df["end"].astype(str))


def read_bed(path: str | os.PathLike) -> pd.DataFrame:
    """Read the first 3 (and name, if present) columns of a BED file."""
    rows = []
    names = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            rows.append((f[0], int(f[1]), int(f[2])))
            names.append(f[3] if len(f) > 3 and f[3] not in ("", ".") else None)
    df = pd.DataFrame(rows, columns=["chrom", "start", "end"])
    idx = region_names(df)
    if names and all(names) and len(set(names)) == len(names):
        idx = pd.Index(names)
    df.index = idx
    return df


def read_fasta(path: str | os.PathLike) -> dict[str, str]:
    """Read a (optionally gzipped) FASTA file into an ordered dict."""
    import gzip

    path = os.fspath(path)
    opener = gzip.open if path.endswith(".gz") else open
    seqs: dict[str, list[str]] = {}
    name = None
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0] if line[1:].split() else f"seq{len(seqs)}"
                seqs[name] = []
            elif name is not None:
                seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}
