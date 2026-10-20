"""Genome access: indexed FASTA reading and region extraction.

[`FastaFile`][moodspy.FastaFile] is a small, dependency-free, memory-mapped reader for
uncompressed FASTA files with a samtools ``.fai`` index (built on the fly
if missing). Other genome objects work too: anything with
``fetch(chrom, start, end)`` (pysam.FastaFile), ``genome[chrom][start:end]``
(pyfaidx.Fasta, or a plain ``dict`` of chromosome -> sequence).
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import warnings
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
        FASTA file. Its samtools index ``<path>.fai`` is used when present.
        An index older than the FASTA is still used if it is consistent with
        the file. Otherwise an index is built (about 0.2 s per Gb) and
        written next to the FASTA, or to the moodspy cache directory
        (``$MOODSPY_CACHE_DIR``, default ``~/.cache/moodspy``) when that
        directory is read-only.
    write_index
        Save a newly built index (next to the FASTA, else to the cache).
    """

    def __init__(self, path: str | os.PathLike, write_index: bool = True):
        self.path = os.fspath(path)
        if self.path.endswith((".gz", ".bgz", ".bz2", ".xz")):
            raise ValueError(
                f"{self.path} is compressed; decompress it first or pass a pysam.FastaFile / "
                "pyfaidx.Fasta object as the genome"
            )
        self._fh = open(self.path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ) if os.path.getsize(self.path) else b""
        self._key = _file_key(self.path)
        self.index = _load_index(self.path, self._mm, self._key, write_index)
        self._bg: np.ndarray | None = None

    def close(self) -> None:
        if not self._fh.closed:
            if isinstance(self._mm, mmap.mmap):
                self._mm.close()
            self._fh.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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
        if start == end:
            return ""

        def at(p: int) -> int:
            return offset + (p // line_bases) * line_width + p % line_bases

        raw = self._mm[at(start) : at(end)]
        if line_width != line_bases:
            raw = raw.replace(b"\n", b"").replace(b"\r", b"")
        return raw.decode("ascii")

    def __getitem__(self, chrom: str) -> str:
        return self.fetch(chrom)

    def base_counts(self) -> np.ndarray:
        """Genome-wide A, C, G, T counts (case-insensitive).

        Computed once from the raw file (about 0.4 s per Gb), then cached in
        memory and in the moodspy cache directory, so later calls and later
        processes get it instantly.
        """
        if self._bg is None:
            cached = _cache_read(self._key, "counts.json")
            if cached is not None:
                self._bg = np.asarray(json.loads(cached), dtype=np.float64)
            else:
                self._bg = self._count_bases()
                _cache_write(self._key, "counts.json", json.dumps(self._bg.tolist()))
        return self._bg.copy()

    def _count_bases(self) -> np.ndarray:
        counts = np.zeros(4, dtype=np.int64)
        if not self.index:
            return counts.astype(np.float64)
        arr = np.frombuffer(self._mm, dtype=np.uint8)
        try:
            for length, offset, lb, lw in self.index.values():
                if length == 0:
                    continue
                end = offset + (length // lb) * lw + length % lb
                for a in range(offset, end, _CHUNK):
                    folded = arr[a : min(a + _CHUNK, end)] | 0x20  # lower-case
                    for i, base in enumerate(b"acgt"):
                        counts[i] += np.count_nonzero(folded == base)
        finally:
            del arr
        return counts.astype(np.float64)


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
        fh.write(_fai_text(index))


_CHUNK = 1 << 26


def _count_byte(arr: np.ndarray, a: int, b: int, value: int) -> int:
    n = 0
    for c in range(a, b, _CHUNK):
        n += int(np.count_nonzero(arr[c : min(c + _CHUNK, b)] == value))
    return n


def _build_fai(mm) -> dict[str, tuple[int, int, int, int]]:
    """samtools-compatible index of a FASTA held in a buffer (vectorised)."""
    size = len(mm)
    heads = [0] if size and mm[:1] == b">" else []
    pos = 0
    while True:
        i = mm.find(b"\n>", pos)
        if i < 0:
            break
        heads.append(i + 1)
        pos = i + 1
    index: dict[str, tuple[int, int, int, int]] = {}
    if not heads:
        return index
    arr = np.frombuffer(mm, dtype=np.uint8)
    try:
        for k, h in enumerate(heads):
            e = heads[k + 1] if k + 1 < len(heads) else size
            hdr_end = mm.find(b"\n", h, e)
            hdr_end = e if hdr_end < 0 else hdr_end
            words = bytes(mm[h + 1 : hdr_end]).split()
            name = words[0].decode() if words else ""
            if name in index:
                raise ValueError(f"duplicate sequence name {name!r} in FASTA")
            s = min(hdr_end + 1, e)
            n_nl = _count_byte(arr, s, e, 10)
            n_cr = _count_byte(arr, s, e, 13)
            length = (e - s) - n_nl - n_cr
            if length == 0:
                index[name] = (0, s, 0, 0)
                continue
            first_nl = mm.find(b"\n", s, e)
            first_nl = e if first_nl < 0 else first_nl
            lw = first_nl + 1 - s
            lb = len(bytes(mm[s:first_nl]).rstrip(b"\r"))
            ok = lb > 0
            if ok:
                # every line but the last holds exactly lb bases followed by an
                # (lw - lb)-byte terminator; only \n / \r may follow the last base
                breaks = (length - 1) // lb
                last = s + breaks * lw + (length - 1) % lb + 1
                term = lw - lb
                ok = last <= e and term in (1, 2)
            if ok:
                ends = arr[s + lw - 1 : s + breaks * lw : lw]
                ok = len(ends) == breaks and bool((ends == 10).all())
                if ok and term == 2:
                    ok = bool((arr[s + lw - 2 : s + breaks * lw : lw] == 13).all())
                if ok:
                    tail_nl = _count_byte(arr, last, e, 10)
                    tail_cr = _count_byte(arr, last, e, 13)
                    ok = n_nl - tail_nl == breaks and n_cr - tail_cr == breaks * (term - 1)
            if not ok:
                raise ValueError(f"sequence {name!r} has lines of unequal length; cannot index")
            index[name] = (length, s, lb, lw)
    finally:
        del arr
    return index


def _index_consistent(index: dict, mm) -> bool:
    """Cheap check that an existing .fai still describes this file."""
    size = len(mm)
    for length, offset, lb, lw in index.values():
        if offset > size or (offset > 0 and mm[offset - 1 : offset] != b"\n"):
            return False
        if length:
            if lb <= 0 or lw < lb:
                return False
            end = offset + (length // lb) * lw + length % lb
            if end > size or (end < size and mm[end : end + 1] not in (b"\n", b"\r", b">")):
                return False
    return True


def _file_key(path: str) -> str:
    st = os.stat(path)
    raw = f"{os.path.realpath(path)}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(raw.encode()).hexdigest()[:20]


def cache_dir() -> str:
    """Directory for moodspy's caches (FASTA indexes, genome base counts)."""
    env = os.environ.get("MOODSPY_CACHE_DIR")
    if env:
        return env
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "moodspy")


def _cache_read(key: str, suffix: str) -> str | None:
    try:
        with open(os.path.join(cache_dir(), f"{key}.{suffix}")) as fh:
            return fh.read()
    except OSError:
        return None


def _cache_write(key: str, suffix: str, text: str) -> bool:
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        tmp = os.path.join(cache_dir(), f"{key}.{suffix}.{os.getpid()}.tmp")
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, os.path.join(cache_dir(), f"{key}.{suffix}"))
        return True
    except OSError:
        return False


def _fai_text(index: dict) -> str:
    return "".join(f"{n}\t{a}\t{b}\t{c}\t{d}\n" for n, (a, b, c, d) in index.items())


def _parse_fai(text: str) -> dict[str, tuple[int, int, int, int]]:
    index = {}
    for line in text.splitlines():
        if line.strip():
            name, length, offset, lb, lw = line.split("\t")[:5]
            index[name] = (int(length), int(offset), int(lb), int(lw))
    return index


_INDEX_MEMO: dict[str, dict] = {}


def _load_index(path: str, mm, key: str, write: bool) -> dict:
    if key in _INDEX_MEMO:
        return _INDEX_MEMO[key]
    fai = path + ".fai"
    index = None
    if os.path.exists(fai):
        index = _read_fai(fai)
        if os.path.getmtime(fai) < os.path.getmtime(path) and not _index_consistent(index, mm):
            index = None
    if index is None:
        cached = _cache_read(key, "fai")
        if cached is not None:
            index = _parse_fai(cached)
    if index is None:
        index = _build_fai(mm)
        if write:
            try:
                _write_fai(index, fai)
            except OSError:
                if not _cache_write(key, "fai", _fai_text(index)):
                    warnings.warn(
                        f"could not save the index of {path} (directory and {cache_dir()} are not "
                        "writable); it will be rebuilt in every new process. Set MOODSPY_CACHE_DIR "
                        "to a writable directory.",
                        stacklevel=3,
                    )
    _INDEX_MEMO[key] = index
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
        if start < 0 or start > end:
            raise ValueError(f"invalid region {chrom}:{start}-{end}")
        if hasattr(g, "fetch"):  # pysam.FastaFile
            seq = g.fetch(chrom, start, end)
        else:
            seq = str(g[chrom][start:end])
        if len(seq) != end - start:  # pysam / pyfaidx / dict slicing truncate silently
            raise ValueError(f"region {chrom}:{start}-{end} runs past the end of {chrom}")
        return seq

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
    import gzip

    rows = []
    names = []
    opener = gzip.open if os.fspath(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
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
