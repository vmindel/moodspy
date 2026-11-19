"""Per-position motif score tracks over whole sequences (no threshold)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from .match import BackgroundSpec, _load_subject, _resolve_bg
from .motifs import DEFAULT_PSEUDOCOUNT, Motif, as_motif_list, motif_labels
from .scanner import Scanner

Strand = Literal["best", "both", "+", "-"]
TrackValue = Literal["score", "pvalue", "neglog10p"]
Anchor = Literal["center", "start"]


def select_motifs(motifs: Any, select: str | Sequence[str]) -> list[Motif]:
    """The motifs whose label, ID or name equals one of ``select``.

    Matching is exact, then case-insensitive. Motifs come back in the order
    of ``select``; a name shared by several motifs returns all of them.
    Raises ``KeyError`` for an entry that matches nothing.
    """
    mots = as_motif_list(motifs)
    labels = motif_labels(mots)
    if isinstance(select, str):
        select = [select]
    keys = [{lab, m.id or "", m.name} - {""} for lab, m in zip(labels, mots)]
    out, seen, missing = [], set(), []
    for want in select:
        hit = [i for i, k in enumerate(keys) if want in k]
        if not hit:
            low = want.lower()
            hit = [i for i, k in enumerate(keys) if low in {x.lower() for x in k}]
        if not hit:
            missing.append(want)
        for i in hit:
            if i not in seen:
                seen.add(i)
                out.append(mots[i])
    if missing:
        raise KeyError(f"no motif with label, ID or name {missing}")
    return out


def motif_tracks(
    motifs: Any,
    subject: Any,
    genome: Any = None,
    *,
    select: str | Sequence[str] | None = None,
    bg: BackgroundSpec | Sequence[float] | Mapping[str, float] = "subject",
    strand: Strand = "best",
    pvalues: bool = True,
    p_cutoff: float = 5e-5,
    pseudocount: float = DEFAULT_PSEUDOCOUNT,
    motifmatchr_compat: bool = False,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """Score every motif at every position of the sequences, for plotting
    motif affinity along peaks as a track instead of as discrete hits.

    Every placement is scored with a step of 1 bp, with no threshold. Scores
    and p-values are those [`match_motifs`][moodspy.match_motifs] uses with
    the same motifs, background and options: a row has ``pvalue <= p_cutoff``
    exactly when ``match_motifs(..., p_cutoff=p_cutoff)`` reports it as a hit
    (except for motifs too short to reach ``p_cutoff`` at all, whose best
    word MOODS still calls a hit).

    Parameters
    ----------
    motifs, subject, genome
        As in [`match_motifs`][moodspy.match_motifs].
    select
        Only these motifs, by label, ID or name (see
        [`select_motifs`][moodspy.select_motifs]). Output grows with the
        number of motifs times the number of bases, so select a few.
    bg
        Background, as in ``match_motifs``. ``"subject"`` is computed from
        the sequences given here, so a track of one peak uses that peak's
        composition. To match a scan of a whole peak set, pass that scan's
        background (``res.background``) explicitly.
    strand
        ``"best"`` (default): one row per placement with the higher strand
        score and its strand. ``"both"``: one row per strand. ``"+"`` /
        ``"-"``: one strand.
    pvalues
        Add a ``pvalue`` column (upper tail under the background). Both
        strands use the forward matrix's null distribution, as the
        thresholds do. With ``strand="best"`` it is the better strand's
        p-value, not corrected for choosing the best of two.
    p_cutoff
        Only used to fill ``attrs["thresholds"]`` (the score cutoff per
        motif, for drawing a hit line); nothing is filtered.
    pseudocount, motifmatchr_compat, n_jobs
        As in ``match_motifs``.

    Returns
    -------
    pandas.DataFrame
        One row per placement whose window lies inside the sequence and
        holds only ``ACGT``. Columns ``motif, seq, [chrom], start, end,
        strand, score[, pvalue]``. ``start``/``end`` are the placement's
        window (0-based, half-open; genomic for regions), so each sequence
        has ``length - motif length + 1`` rows per motif and strand.
        ``attrs`` holds ``background``, ``p_cutoff`` and ``thresholds``.
        Use [`write_bedgraph`][moodspy.write_bedgraph] for a genome browser
        or plotgardener track.
    """
    if select is not None:
        motifs = select_motifs(motifs, select)
    seqs, frame = _load_subject(subject, genome)
    if len(seqs) == 0:
        raise ValueError("no sequences to score")
    background = _resolve_bg(bg, seqs, genome)
    scanner = Scanner(
        motifs, background, p_cutoff, pseudocount=pseudocount,
        motifmatchr_compat=motifmatchr_compat, n_jobs=n_jobs,
    )
    tr = scanner.tracks(seqs, strand=strand, pvalues=pvalues)
    s_idx = tr["seq"].to_numpy()
    names = frame.index.astype(str).to_numpy()
    res = pd.DataFrame({"motif": tr["motif"]})
    res["seq"] = (
        pd.Categorical.from_codes(s_idx, categories=pd.unique(names))
        if len(set(names)) == len(names)
        else names[s_idx]
    )
    start, end = tr["start"].to_numpy(), tr["end"].to_numpy()
    if "chrom" in frame:
        chroms = pd.Categorical(frame["chrom"].astype(str))
        res["chrom"] = pd.Categorical.from_codes(chroms.codes[s_idx], categories=chroms.categories)
        offs = frame["start"].to_numpy()[s_idx]
        start, end = start + offs, end + offs
    res["start"] = start
    res["end"] = end
    res["strand"] = tr["strand"]
    res["score"] = tr["score"].to_numpy()
    if pvalues:
        res["pvalue"] = tr["pvalue"].to_numpy()
    res.attrs.update(
        background=background,
        p_cutoff=p_cutoff,
        thresholds=pd.Series(scanner.thresholds, index=pd.Index(scanner.labels, name="motif")),
    )
    return res


def write_bedgraph(
    tracks: pd.DataFrame,
    path: str | os.PathLike,
    *,
    motif: str | None = None,
    value: TrackValue = "score",
    anchor: Anchor = "center",
    precision: int = 6,
) -> list[str]:
    """Write a [`motif_tracks`][moodspy.motif_tracks] result as bedGraph.

    Each placement becomes one 1-bp interval, at the centre base of its
    window (``start + length // 2``) or at its first base. A bedGraph file
    holds one value per base, so:

    * with ``strand="both"`` the two strands are collapsed to the better
      one (higher score, lower p-value);
    * several motifs need ``motif=`` or a ``{motif}`` placeholder in
      ``path`` (one file per motif);
    * overlapping input regions would give one base two values and raise
      ``ValueError``; merge peaks first (e.g. ``bedtools merge``).

    Plain (non-genomic) sequences use the sequence name as chromosome and
    positions within the sequence.

    Parameters
    ----------
    tracks
        Output of ``motif_tracks``.
    path
        Output file; may contain ``{motif}``.
    motif
        Write only this motif (its label in ``tracks["motif"]``).
    value
        ``"score"``, ``"pvalue"`` or ``"neglog10p"`` (``-log10 pvalue``).
    anchor
        ``"center"`` or ``"start"`` of the motif window.
    precision
        Significant digits written.

    Returns
    -------
    list of str
        The files written.
    """
    if value not in ("score", "pvalue", "neglog10p"):
        raise ValueError("value must be 'score', 'pvalue' or 'neglog10p'")
    if anchor not in ("center", "start"):
        raise ValueError("anchor must be 'center' or 'start'")
    if value != "score" and "pvalue" not in tracks:
        raise ValueError("tracks have no pvalue column; rerun motif_tracks with pvalues=True")
    path = os.fspath(path)
    labels = [str(m) for m in pd.unique(tracks["motif"].astype(str))]
    if motif is not None:
        if motif not in labels:
            raise KeyError(f"motif {motif!r} not in tracks (have {labels[:5]}...)")
        labels = [motif]
    if len(labels) > 1 and "{motif}" not in path:
        raise ValueError(
            f"tracks hold {len(labels)} motifs; pass motif= or put '{{motif}}' in the path"
        )
    chrom_col = "chrom" if "chrom" in tracks else "seq"
    motif_cat = pd.Categorical(tracks["motif"].astype(str)) if not isinstance(
        tracks["motif"].dtype, pd.CategoricalDtype) else tracks["motif"].array
    chrom_cat = pd.Categorical(tracks[chrom_col])
    # bedGraph tools want chromosomes in byte order
    chrom_rank = np.argsort(np.argsort(np.asarray(chrom_cat.categories.astype(str)), kind="stable"))
    written = []
    for lab in labels:
        rows = np.flatnonzero(motif_cat.codes == list(motif_cat.categories).index(lab))
        start = tracks["start"].to_numpy()[rows]
        pos = start + (tracks["end"].to_numpy()[rows] - start) // 2 if anchor == "center" else start
        # lower p is better: collapse strands on -p, then transform back
        v = tracks["score"].to_numpy()[rows] if value == "score" else -tracks["pvalue"].to_numpy()[rows]
        chrom = chrom_rank[chrom_cat.codes[rows]].astype(np.int64)
        rev = (np.asarray(tracks["strand"].array.codes)[rows] if "strand" in tracks
               else np.zeros(rows.size, dtype=np.int8)).astype(np.int64)
        key = chrom * (int(pos.max(initial=0)) + 1) + pos
        order = np.argsort(key * 2 + rev, kind="stable")
        key, v, kr = key[order], v[order], (key * 2 + rev)[order]
        # the same base and strand twice can only come from overlapping regions
        dup = np.flatnonzero(kr[1:] == kr[:-1])
        if dup.size:
            i = order[dup[0]]
            raise ValueError(
                f"overlapping regions give {chrom_cat[rows[i]]}:{pos[i]} more than one value; "
                "merge overlapping peaks first (e.g. bedtools merge)"
            )
        first = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1] if key.size else np.zeros(0, np.int64)
        best = np.maximum.reduceat(v, first) if key.size else v
        if value == "pvalue":
            best = -best
        elif value == "neglog10p":
            best = -np.log10(-best)
        base = pos[order][first]
        names = np.asarray(chrom_cat.categories.astype(str))[chrom_cat.codes[rows][order][first]]
        out = pd.DataFrame({"chrom": names, "start": base, "end": base + 1, "value": best})
        dest = path.replace("{motif}", _safe(lab))
        out.to_csv(dest, sep="\t", header=False, index=False, float_format=f"%.{precision}g")
        written.append(dest)
    return written


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
