"""Motif file readers and writers.

Supported formats: JASPAR (``.jaspar``, bracketed or plain), raw JASPAR
PFM (``.pfm``, 4 rows of counts), MEME (minimal and full text output),
HOMER (``.motif``), HOCOMOCO (``.pcm``/``.pfm``/``.pwm``) and TRANSFAC.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Literal

import numpy as np

from .motifs import Motif

MotifFormat = Literal["jaspar", "pfm", "meme", "homer", "hocomoco", "transfac"]

_FLOAT = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def read_motifs(path: str | os.PathLike, format: MotifFormat | None = None) -> list[Motif]:
    """Read motifs from a file, detecting the format if not given."""
    with open(path) as fh:
        text = fh.read()
    fmt = format or _sniff(text, os.fspath(path))
    parser = {
        "jaspar": parse_jaspar,
        "pfm": parse_pfm,
        "meme": parse_meme,
        "homer": parse_homer,
        "hocomoco": parse_hocomoco,
        "transfac": parse_transfac,
    }[fmt]
    motifs = parser(text)
    if fmt == "pfm" and motifs and not motifs[0].label:
        stem = os.path.splitext(os.path.basename(os.fspath(path)))[0]
        motifs[0].id = stem
    if not motifs:
        raise ValueError(f"no motifs found in {path} (format {fmt!r})")
    return motifs


def _sniff(text: str, path: str) -> MotifFormat:
    ext = os.path.splitext(path)[1].lower()
    head = text.lstrip()[:4000]
    if "MEME version" in head:
        return "meme"
    if re.search(r"^(P0|PO)\s", head, re.M) or re.search(r"^ID\s", head, re.M) and "//" in text:
        return "transfac"
    if head.startswith(">"):
        lines = head.splitlines()
        body = lines[1:3]
        if ext == ".jaspar" or any("[" in line or re.match(r"\s*[ACGT]\s", line) for line in body):
            return "jaspar"
        widths = [len(_FLOAT.findall(ln)) for ln in body if ln.strip() and not ln.startswith(">")]
        if widths and widths[0] != 4:
            # >ID name, then 4 rows of L values: JASPAR's "raw PFM" download
            return "jaspar"
        if len(lines[0].split()) == 1:
            # HOMER headers carry consensus, name and threshold; HOCOMOCO only the ID
            return "hocomoco"
        return "homer"
    return "pfm"


# ---------------------------------------------------------------------- JASPAR
def parse_jaspar(text: str) -> list[Motif]:
    """Parse JASPAR format: ``>ID name`` followed by A/C/G/T rows."""
    motifs = []
    blocks = re.split(r"^>", text, flags=re.M)
    for block in blocks:
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 5:
            continue
        header = lines[0].split(None, 1)
        mid = header[0] if header else None
        name = header[1].strip() if len(header) > 1 else ""
        rows = {}
        for ln in lines[1:5]:
            m = re.match(r"\s*([ACGTacgt])\s*\[?(.*?)\]?\s*$", ln)
            if m:
                rows[m.group(1).upper()] = [float(x) for x in _FLOAT.findall(m.group(2))]
            else:
                rows["ACGT"[len(rows)]] = [float(x) for x in _FLOAT.findall(ln)]
        motifs.append(Motif.from_counts([rows[b] for b in "ACGT"], name=name, id=mid))
    return motifs


def parse_pfm(text: str) -> list[Motif]:
    """Parse a raw PFM: 4 whitespace-separated rows (A, C, G, T)."""
    rows = [[float(x) for x in _FLOAT.findall(ln)] for ln in text.splitlines()]
    rows = [r for r in rows if r]
    if len(rows) != 4:
        raise ValueError(f"raw PFM must have 4 numeric rows, found {len(rows)}")
    return [Motif.from_counts(rows)]


def write_jaspar(motifs: Iterable[Motif], path: str | os.PathLike) -> None:
    """Write motifs in JASPAR format (probabilities are scaled to counts)."""
    with open(path, "w") as fh:
        for i, m in enumerate(motifs):
            mat = _as_counts(m)
            mid = m.id or m.label or f"motif_{i}"
            fh.write(f">{mid}\t{m.name}\n" if m.name and m.name != mid else f">{mid}\n")
            for b, row in zip("ACGT", mat):
                fh.write(f"{b}  [ " + " ".join(_fmt(v) for v in row) + " ]\n")


# ---------------------------------------------------------------------- MEME
def parse_meme(text: str) -> list[Motif]:
    """Parse MEME text format (``MOTIF`` blocks with letter-probability matrices)."""
    alph = re.search(r"^ALPHABET\s*=\s*(\S+)", text, re.M)
    if alph and alph.group(1).upper() != "ACGT":
        raise ValueError(f"only the ACGT alphabet is supported, file uses {alph.group(1)}")
    motifs = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("MOTIF"):
            parts = ln.split()
            mid = parts[1] if len(parts) > 1 else f"motif_{len(motifs)}"
            name = parts[2] if len(parts) > 2 else ""
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("letter-probability"):
                if lines[j].startswith("MOTIF"):
                    break
                j += 1
            if j >= len(lines) or lines[j].startswith("MOTIF"):
                i = j
                continue
            header = lines[j]
            width = re.search(r"w\s*=\s*(\d+)", header)
            nsites = re.search(r"nsites\s*=\s*([\d.eE+-]+)", header)
            rows = []
            j += 1
            while j < len(lines) and (width is None or len(rows) < int(width.group(1))):
                vals = _FLOAT.findall(lines[j])
                if len(vals) == 4:
                    rows.append([float(v) for v in vals])
                elif rows or lines[j].startswith(("MOTIF", "URL")):
                    break
                j += 1
            motifs.append(
                Motif.from_probabilities(
                    np.array(rows).T, name=name, id=mid,
                    nsites=float(nsites.group(1)) if nsites else None,
                )
            )
            i = j
        else:
            i += 1
    return motifs


def write_meme(motifs: Iterable[Motif], path: str | os.PathLike, background=None) -> None:
    """Write motifs in minimal MEME format."""
    bg = np.full(4, 0.25) if background is None else np.asarray(background, dtype=float)
    with open(path, "w") as fh:
        fh.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        fh.write("Background letter frequencies\n")
        fh.write(" ".join(f"{b} {v:.4f}" for b, v in zip("ACGT", bg)) + "\n\n")
        for i, m in enumerate(motifs):
            probs = m.probabilities(pseudocount=0.0) if m.kind != "log-odds" else m.probabilities()
            mid = m.id or m.label or f"motif_{i}"
            name = f" {m.name}" if m.name and m.name != mid else ""
            nsites = m.nsites or (m.matrix.sum(axis=0).max() if m.kind == "counts" else 20)
            fh.write(f"MOTIF {mid}{name}\n")
            fh.write(f"letter-probability matrix: alength= 4 w= {m.length} nsites= {_fmt(nsites)} E= 0\n")
            for col in probs.T:
                fh.write(" ".join(f"{v:.6f}" for v in col) + "\n")
            fh.write("\n")


# ---------------------------------------------------------------------- HOMER
def parse_homer(text: str) -> list[Motif]:
    """Parse HOMER ``.motif`` files (``>consensus  name  log-odds-threshold ...``)."""
    motifs = []
    for block in re.split(r"^>", text, flags=re.M):
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        header = lines[0].split("\t")
        name = header[1].strip() if len(header) > 1 else header[0].strip()
        rows = [[float(x) for x in _FLOAT.findall(ln)] for ln in lines[1:]]
        meta = {}
        if len(header) > 2:
            try:
                meta["homer_threshold"] = float(header[2])
            except ValueError:
                pass
        motifs.append(Motif.from_probabilities(np.array(rows).T, name=name, id=None, meta=meta))
    return motifs


# ---------------------------------------------------------------------- HOCOMOCO
def parse_hocomoco(text: str) -> list[Motif]:
    """Parse HOCOMOCO ``.pcm``, ``.pfm`` and ``.pwm`` files.

    Each record is a ``>ID`` line followed by one row of A, C, G, T values
    per position. The kind (counts, probability or log-odds) is inferred from
    the values; log-odds are taken as natural-log scores against a uniform
    background, which is how HOCOMOCO computes its PWMs.
    """
    motifs = []
    for block in re.split(r"^>", text, flags=re.M):
        lines = [ln for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        header = lines[0].split(None, 1)
        mid = header[0] if header else None
        name = header[1].strip() if len(header) > 1 else ""
        rows = [[float(x) for x in _FLOAT.findall(ln)] for ln in lines[1:]]
        if any(len(r) != 4 for r in rows):
            raise ValueError(f"HOCOMOCO motif {mid!r}: every row must hold 4 values (A, C, G, T)")
        mat = np.array(rows).T
        motifs.append(Motif(mat, name=name, id=mid, log_base=np.e))
    return motifs


# ---------------------------------------------------------------------- TRANSFAC
def parse_transfac(text: str) -> list[Motif]:
    """Parse TRANSFAC matrices (``ID``/``NA`` + ``P0`` table, ``//`` separated)."""
    motifs = []
    for record in re.split(r"^//\s*$", text, flags=re.M):
        mid = name = None
        rows, cols, in_matrix = [], None, False
        for ln in record.splitlines():
            key = ln[:2]
            if key in ("AC", "ID") and mid is None:
                mid = ln[2:].strip() or None
            elif key == "NA":
                name = ln[2:].strip()
            elif key in ("P0", "PO"):
                cols = ln.split()[1:5]
                in_matrix = True
            elif in_matrix and re.match(r"^\d+\s", ln):
                vals = ln.split()[1:5]
                rows.append([float(v) for v in vals])
            elif in_matrix and ln.strip() and not re.match(r"^\d+\s", ln):
                in_matrix = False
        if rows:
            mat = np.array(rows).T
            if cols and [c.upper() for c in cols] != list("ACGT"):
                mat = mat[[[c.upper() for c in cols].index(b) for b in "ACGT"]]
            motifs.append(Motif(mat, name=name or "", id=mid))
    return motifs


# ---------------------------------------------------------------------- helpers
def _as_counts(m: Motif) -> np.ndarray:
    if m.kind == "counts":
        return m.matrix
    probs = m.matrix if m.kind == "probability" else m.probabilities()
    return probs * (m.nsites or 100.0)


def _fmt(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.4g}"
