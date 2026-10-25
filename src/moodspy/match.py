"""High-level, motifmatchr-style interface."""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._alphabet import base_counts, encode_many
from .genome import _Genome, read_fasta, regions_frame
from .motifs import DEFAULT_PSEUDOCOUNT, check_background
from .scanner import Scanner

Output = Literal["matches", "scores", "positions"]
BackgroundSpec = Literal["subject", "genome", "even"]


@dataclass
class MotifMatches:
    """Per-sequence motif match summary (the analogue of motifmatchr's
    SummarizedExperiment).

    Attributes
    ----------
    matches
        ``(n_seqs, n_motifs)`` boolean CSR matrix: does the sequence have
        at least one hit on either strand?
    scores
        Best hit score per sequence and motif (``out="scores"`` only).
    counts
        Number of hits per sequence and motif (``out="scores"`` only).
    sequences
        One row per sequence; genomic inputs have ``chrom, start, end``.
    motifs
        One row per motif: ``id``, ``name``, ``length``, ``threshold``.
    background
        A/C/G/T frequencies used for scoring and p-values.
    """

    matches: sp.csr_matrix
    sequences: pd.DataFrame
    motifs: pd.DataFrame
    background: np.ndarray
    p_cutoff: float | None
    scores: sp.csr_matrix | None = None
    counts: sp.csr_matrix | None = None
    params: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.matches.shape

    def __repr__(self) -> str:
        assays = ", ".join(a for a in ("matches", "scores", "counts") if getattr(self, a) is not None)
        return (
            f"MotifMatches({self.shape[0]} sequences x {self.shape[1]} motifs; "
            f"assays: {assays}; p_cutoff={self.p_cutoff})"
        )

    def to_dataframe(self, assay: Literal["matches", "scores", "counts"] = "matches") -> pd.DataFrame:
        """Dense ``sequences x motifs`` DataFrame of one assay."""
        mat = getattr(self, assay)
        if mat is None:
            raise ValueError(f"{assay!r} not computed; rerun match_motifs with out='scores'")
        return pd.DataFrame(mat.toarray(), index=self.sequences.index, columns=self.motifs.index)

    def to_anndata(self):
        """AnnData with sequences as ``obs`` and motifs as ``var``.

        ``X`` holds the boolean matches; ``scores`` and ``counts`` (when
        computed) are stored as layers.
        """
        import anndata as ad

        layers = {k: getattr(self, k) for k in ("scores", "counts") if getattr(self, k) is not None}
        obs = self.sequences.copy()
        obs.index = obs.index.astype(str)
        var = self.motifs.copy()
        return ad.AnnData(
            X=self.matches,
            obs=obs,
            var=var,
            layers=layers,
            uns={"motif_matching": {"background": self.background, "p_cutoff": self.p_cutoff, **self.params}},
        )


def match_motifs(
    motifs: Any,
    subject: Any,
    genome: Any = None,
    *,
    bg: BackgroundSpec | Sequence[float] | Mapping[str, float] = "subject",
    out: Output = "matches",
    p_cutoff: float = 5e-5,
    w: int = 7,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    motifmatchr_compat: bool = False,
    n_jobs: int | None = None,
) -> MotifMatches | pd.DataFrame:
    """Find motif matches in sequences or genomic regions.

    Python port of motifmatchr's ``matchMotifs``, built on a MOODS-style
    scanner. Both strands are scanned; a hit is any placement whose log2
    odds score reaches the motif's p-value threshold.

    Parameters
    ----------
    motifs
        A [`Motif`][moodspy.Motif], a list of them, a ``{label: matrix}``
        mapping, Biopython / pyjaspar motifs, or a path to a motif file
        (JASPAR, MEME, HOMER, TRANSFAC). Count matrices are converted to
        log-odds with ``pseudocount`` as in TFBSTools' ``toPWM``.
    subject
        What to scan:

        * a DNA string, list of strings or ``{name: sequence}`` dict;
        * a FASTA file path (``.fa``, ``.fasta``, ``.fna``, optionally ``.gz``);
        * genomic regions (needs ``genome``): a DataFrame with
          ``chrom, start, end`` (0-based, half-open), a BED file path,
          a pyranges object, a list of ``"chr:start-end"`` strings, or an
          AnnData whose ``var_names`` (or ``obs_names``) are such strings.
    genome
        FASTA path, [`FastaFile`][moodspy.FastaFile], ``pysam.FastaFile``,
        ``pyfaidx.Fasta`` or ``{chrom: sequence}`` dict. Needed for genomic
        regions and for ``bg="genome"``.
    bg
        Background nucleotide frequencies: ``"subject"`` (computed from the
        scanned sequences), ``"genome"``, ``"even"`` (uniform), or 4 values
        in A, C, G, T order.
    out
        ``"matches"``: sparse boolean match matrix.
        ``"scores"``: matches plus best score and hit count per sequence.
        ``"positions"``: a DataFrame with one row per hit.
    p_cutoff
        P-value cutoff for calling a hit, in (0, 1]. ``p_cutoff=1`` keeps every
        placement, so ``out="scores"`` gives each sequence's best score over
        all placements and strands, and ``counts`` becomes the number of
        placements scored.
    w
        Lookup window size. Affects speed, not which hits are found.
    pseudocount
        Pseudocount added to count / probability matrices.
    motifmatchr_compat
        Reproduce motifmatchr bit for bit, including its inverted
        background correction of the score matrix (see
        [`Motif.log_odds`][moodspy.Motif.log_odds]). Results only differ from the default when
        the background is not uniform.
    n_jobs
        Number of threads (default: all cores).

    Returns
    -------
    MotifMatches or pandas.DataFrame
        ``out="positions"`` returns a DataFrame with columns ``motif, seq,
        start, end, strand, score`` (plus ``chrom`` for genomic regions, in
        which case ``start`` / ``end`` are genomic coordinates).
    """
    if out not in ("matches", "scores", "positions"):
        raise ValueError("out must be 'matches', 'scores' or 'positions'")
    seqs, frame = _load_subject(subject, genome)
    if len(seqs) == 0:
        raise ValueError("no sequences to scan")
    background = _resolve_bg(bg, seqs, genome)
    scanner = Scanner(
        motifs,
        background,
        p_cutoff,
        w=w,
        pseudocount=pseudocount,
        motifmatchr_compat=motifmatchr_compat,
        n_jobs=n_jobs,
    )
    motif_frame = pd.DataFrame(
        {
            "id": [m.id or "" for m in scanner.motifs],
            "name": [m.name for m in scanner.motifs],
            "length": [m.length for m in scanner.motifs],
            "threshold": scanner.thresholds,
        },
        index=pd.Index(scanner.labels, name="motif"),
    )

    if out == "positions":
        hits = scanner.scan(seqs)
        seq_names = frame.index.astype(str).to_numpy()
        res = pd.DataFrame(
            {
                "motif": hits["motif"],
                "seq": pd.Categorical.from_codes(hits["seq"].to_numpy(), categories=pd.unique(seq_names))
                if len(set(seq_names)) == len(seq_names)
                else seq_names[hits["seq"].to_numpy()],
            }
        )
        if "chrom" in frame:
            offs = frame["start"].to_numpy()[hits["seq"].to_numpy()]
            chroms = frame["chrom"].astype(str).to_numpy()[hits["seq"].to_numpy()]
            res["chrom"] = pd.Categorical(chroms)
            res["start"] = hits["start"].to_numpy() + offs
            res["end"] = hits["end"].to_numpy() + offs
        else:
            res["start"] = hits["start"].to_numpy()
            res["end"] = hits["end"].to_numpy()
        res["strand"] = hits["strand"]
        res["score"] = hits["score"].to_numpy()
        res.attrs.update(background=background, p_cutoff=p_cutoff, thresholds=motif_frame["threshold"])
        return res

    counts, scores = scanner.summarize(seqs)
    matches = counts.astype(bool)
    result = MotifMatches(
        matches=matches,
        sequences=frame,
        motifs=motif_frame,
        background=background,
        p_cutoff=p_cutoff,
        params={"w": w, "pseudocount": pseudocount, "motifmatchr_compat": motifmatchr_compat},
    )
    if out == "scores":
        result.scores = scores
        result.counts = counts
    return result


# ---------------------------------------------------------------------- input handling
_FASTA_EXT = (".fa", ".fasta", ".fna", ".fas", ".fa.gz", ".fasta.gz", ".fna.gz")
_BED_EXT = (".bed", ".narrowpeak", ".broadpeak", ".bed.gz")


def _load_subject(subject: Any, genome: Any) -> tuple[list[str], pd.DataFrame]:
    """Return (sequences, per-sequence DataFrame)."""
    if isinstance(subject, os.PathLike) or (isinstance(subject, str) and os.path.isfile(subject)):
        path = os.fspath(subject)
        low = path.lower()
        if low.endswith(_BED_EXT):
            return _load_regions(regions_frame(path), genome)
        if low.endswith(_FASTA_EXT) or genome is None:
            subject = read_fasta(path)
        else:
            return _load_regions(regions_frame(path), genome)
    if isinstance(subject, str):
        if re.search(r"[/\\.]", subject):
            raise FileNotFoundError(
                f"{subject!r} is not an existing file (strings are only scanned as DNA "
                "when they contain no '/', '\\' or '.')"
            )
        return [subject], pd.DataFrame(index=pd.Index(["seq0"], name="seq"))
    if _is_anndata(subject):
        names = subject.var_names if _looks_like_regions(subject.var_names) else subject.obs_names
        return _load_regions(regions_frame(list(names)), genome)
    if isinstance(subject, Mapping):
        names = [str(k) for k in subject]
        return [str(v) for v in subject.values()], pd.DataFrame(index=pd.Index(names, name="seq"))
    if isinstance(subject, pd.DataFrame) or hasattr(subject, "df"):
        return _load_regions(regions_frame(subject), genome)
    if isinstance(subject, pd.Series):
        return [str(s) for s in subject], pd.DataFrame(index=subject.index.astype(str).rename("seq"))
    seqs = list(subject)
    if seqs and _looks_like_regions(seqs):
        # region strings without a genome raise in _load_regions
        return _load_regions(regions_frame(seqs), genome)
    seqs = [str(s) for s in seqs]
    return seqs, pd.DataFrame(index=pd.Index([f"seq{i}" for i in range(len(seqs))], name="seq"))


def _load_regions(frame: pd.DataFrame, genome: Any) -> tuple[list[str], pd.DataFrame]:
    if genome is None:
        raise ValueError("genomic regions given: pass genome= (FASTA path or genome object)")
    g = _Genome(genome)
    seqs = g.fetch_many(frame["chrom"], frame["start"], frame["end"])
    frame = frame.copy()
    frame.index = frame.index.astype(str).rename("seq")
    return seqs, frame


def _resolve_bg(bg: Any, seqs: list[str], genome: Any) -> np.ndarray:
    if isinstance(bg, str):
        if bg == "even":
            return np.full(4, 0.25)
        if bg == "subject":
            counts = base_counts(encode_many(seqs)[0])
        elif bg == "genome":
            if genome is None:
                raise ValueError("bg='genome' requires a genome")
            counts = _Genome(genome).base_counts()
        else:
            raise ValueError("bg must be 'subject', 'genome', 'even' or 4 frequencies")
        if counts.sum() == 0:
            raise ValueError("no A/C/G/T characters found to estimate the background")
        freqs = counts / counts.sum()
        if (freqs == 0).any():
            warnings.warn(
                f"background frequencies contain zeros ({freqs.round(4)}); adding a small pseudocount",
                stacklevel=3,
            )
            freqs = (counts + 1.0) / (counts.sum() + 4.0)
        return freqs
    return check_background(bg)


def _is_anndata(obj: Any) -> bool:
    return type(obj).__name__ == "AnnData" and hasattr(obj, "var_names")


def _looks_like_regions(names: Sequence[str]) -> bool:
    from .genome import _REGION_RE

    sample = list(names[:20]) if hasattr(names, "__getitem__") else list(names)[:20]
    return bool(sample) and all(_REGION_RE.match(str(s)) for s in sample)
