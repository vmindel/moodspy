"""Sequence logos with matplotlib: single motifs, aligned stacks and trees.

Needs matplotlib (``pip install "moodspy[plot]"``). The layout follows
universalmotif's ``view_motifs``: information content logos by default,
letters sorted so the tallest is on top, multiple motifs aligned to the
highest-information motif and stacked, reverse-complemented motifs marked
``[RC]``.
"""

from __future__ import annotations

from functools import cache
from typing import Any, Literal

import numpy as np

from .manipulate import UM_PSEUDOCOUNT, icm, ppm
from .motifs import as_motif, as_motif_list, motif_labels

#: universalmotif / ggseqlogo nucleotide colours.
DNA_COLOURS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}

LogoType = Literal["ICM", "PPM", "PWM"]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError('plotting needs matplotlib: pip install "moodspy[plot]"') from e
    return plt


@cache
def _glyph(letter: str, font: str):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    path = TextPath((0, 0), letter, size=1, prop=FontProperties(family=font, weight="bold"))
    ext = path.get_extents()
    return path, ext


def _draw_letter(ax, letter, x, y, width, height, colour, font):
    from matplotlib.patches import PathPatch
    from matplotlib.transforms import Affine2D

    if height <= 0:
        return
    path, ext = _glyph(letter, font)
    tr = (Affine2D().translate(-ext.x0, -ext.y0)
          .scale(width / ext.width, height / ext.height).translate(x, y))
    ax.add_patch(PathPatch(tr.transform_path(path), facecolor=colour, edgecolor="none", lw=0))


def _heights(motif, kind: LogoType, pseudocount: float, background=None, relative_entropy=False):
    m = as_motif(motif)
    if kind == "ICM":
        return icm(m, pseudocount, background, relative_entropy)
    if kind == "PPM":
        return ppm(m, pseudocount)
    if kind == "PWM":
        return m.log_odds(background, pseudocount=pseudocount)
    raise ValueError("kind must be 'ICM', 'PPM' or 'PWM'")


def _draw_matrix(ax, mat, kind, colours, font, min_height, x0=0, sort=None):
    """Draw a (4, L) height matrix with column i centred at x0 + i + 1."""
    sort = kind != "PPM" if sort is None else sort
    letters = "ACGT"
    for i in range(mat.shape[1]):
        col = mat[:, i]
        order = np.argsort(col, kind="stable") if sort else np.arange(4)
        ypos, yneg = 0.0, 0.0
        for j in order:
            h = col[j]
            if abs(h) < min_height:
                continue
            if h > 0:
                _draw_letter(ax, letters[j], x0 + i + 0.55, ypos, 0.9, h, colours[letters[j]], font)
                ypos += h
        for j in order[::-1]:
            h = col[j]
            if h < 0 and abs(h) >= min_height:
                _draw_letter(ax, letters[j], x0 + i + 0.55, yneg + h, 0.9, -h, colours[letters[j]], font)
                yneg += h


def _style(ax, kind, width, ymin, ymax, show_positions):
    ax.set_xlim(0.5, width + 0.5)
    ax.set_ylim(ymin, ymax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if show_positions:
        ax.set_xticks(range(1, width + 1))
    else:
        ax.set_xticks([])
    ax.set_ylabel({"ICM": "bits", "PPM": "probability", "PWM": "log2 odds"}[kind])
    if kind == "PWM":
        ax.axhline(0, color="0.75", lw=0.5)


def plot_logo(
    motif: Any,
    ax=None,
    *,
    kind: LogoType = "ICM",
    pseudocount: float = UM_PSEUDOCOUNT,
    background: Any = None,
    relative_entropy: bool = False,
    colours: dict | None = None,
    font: str = "DejaVu Sans",
    min_height: float = 0.01,
    title: str | None = None,
    show_positions: bool = True,
):
    """Draw a sequence logo of one motif.

    Parameters
    ----------
    kind
        ``"ICM"``: letter heights in bits (information content). ``"PPM"``:
        probabilities. ``"PWM"``: log2-odds scores, negative ones drawn
        below zero.
    pseudocount
        Pseudocount for count matrices (universalmotif's default 1). For
        ``"PWM"`` it is passed to [`Motif.log_odds`][moodspy.Motif.log_odds].
    background
        Background for ``"PWM"`` and relative-entropy ICMs.
    min_height
        Letters shorter than this are not drawn.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _plt()
    if ax is None:
        m = as_motif(motif)
        _, ax = plt.subplots(figsize=(max(2.5, 0.45 * m.length + 0.8), 1.8))
    mat = _heights(motif, kind, pseudocount, background, relative_entropy)
    _draw_matrix(ax, mat, kind, colours or DNA_COLOURS, font, min_height)
    if kind == "ICM":
        ymin, ymax = 0.0, 2.0
    elif kind == "PPM":
        ymin, ymax = 0.0, 1.0
    else:
        ymin = min(0.0, np.minimum(mat, 0).sum(axis=0).min()) * 1.05
        ymax = max(0.0, np.maximum(mat, 0).sum(axis=0).max()) * 1.05
    _style(ax, kind, mat.shape[1], ymin, ymax, show_positions)
    if title is None:
        title = as_motif(motif).label
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    return ax


def view_motifs(
    motifs: Any,
    *,
    kind: Literal["ICM", "PPM"] = "ICM",
    align: bool = True,
    method: str = "ALLR",
    try_rc: bool = True,
    min_overlap: float = 6,
    min_mean_ic: float = 0.25,
    score_strat: str = "sum",
    sort_by: Literal["none", "ic", "similarity"] = "none",
    pseudocount: float = UM_PSEUDOCOUNT,
    colours: dict | None = None,
    font: str = "DejaVu Sans",
    min_height: float = 0.01,
    rc_text: str = " [RC]",
    show_positions_once: bool = True,
    panel_height: float = 1.3,
):
    """Plot several motifs as stacked logos, aligned on a common frame.

    A port of universalmotif's ``view_motifs``. The motifs are aligned with
    [`align_motifs`][moodspy.align_motifs] (anchor: highest total
    information content; ``method``/``score_strat``/``min_overlap`` as in
    [`compare_motifs`][moodspy.compare_motifs]) and drawn one per row.
    Motifs drawn reverse-complemented get ``rc_text`` after their name.

    Parameters
    ----------
    kind
        ``"ICM"`` (bits) or ``"PPM"`` (probabilities).
    align
        ``False`` draws each motif left-aligned as given.
    sort_by
        Row order: ``"none"`` (input order), ``"ic"`` (decreasing
        information content) or ``"similarity"`` (clustered by the
        correlation of the aligned matrices).

    Returns
    -------
    matplotlib.figure.Figure
    """
    from .compare import align_motifs

    plt = _plt()
    mots = as_motif_list(motifs)
    labels = motif_labels(mots)
    if align and len(mots) > 1:
        al = align_motifs(mots, method=method, try_rc=try_rc, min_overlap=min_overlap,
                          min_mean_ic=min_mean_ic, score_strat=score_strat, pseudocount=pseudocount)
        mats = al.padded(kind, pseudocount)
        names = [lab + (rc_text if s == "-" else "") for lab, s in zip(al.table.index, al.table["strand"])]
        input_order = [labels.index(lab) for lab in al.table.index]
        width = al.width
    else:
        width = max(m.length for m in mots)
        mats = []
        for m in mots:
            full = np.zeros((4, width))
            full[:, : m.length] = _heights(m, kind, pseudocount)
            mats.append(full)
        names = labels
        input_order = list(range(len(mots)))
    if sort_by == "none":
        perm = np.argsort(input_order, kind="stable")
    elif sort_by == "ic":
        ics = [icm(mots[i], pseudocount).sum() for i in input_order]
        perm = np.argsort(-np.array(ics), kind="stable")
    elif sort_by == "similarity":
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        flat = np.array([x.ravel() for x in mats])
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.nan_to_num(np.corrcoef(flat), nan=0.0)
        np.fill_diagonal(r, 1.0)
        perm = leaves_list(linkage(squareform(1 - r, checks=False), "complete")) if len(mats) > 2 else np.arange(len(mats))
    else:
        raise ValueError("sort_by must be 'none', 'ic' or 'similarity'")
    n = len(mats)
    fig, axes = plt.subplots(n, 1, figsize=(max(3.0, 0.45 * width + 1.0), panel_height * n),
                             squeeze=False, sharex=True)
    for row, k in enumerate(perm):
        ax = axes[row, 0]
        _draw_matrix(ax, mats[k], kind, colours or DNA_COLOURS, font, min_height)
        _style(ax, kind, width, 0.0, 2.0 if kind == "ICM" else 1.0,
               not show_positions_once or row == n - 1)
        ax.set_title(names[k], fontsize=10, loc="left")
    fig.tight_layout()
    return fig


def plot_motif_tree(
    tree: Any,
    *,
    logos: bool = True,
    kind: Literal["ICM", "PPM"] = "ICM",
    align: bool = True,
    pseudocount: float = UM_PSEUDOCOUNT,
    colours: dict | None = None,
    font: str = "DejaVu Sans",
    row_height: float = 0.55,
    **align_kw,
):
    """Dendrogram of a [`MotifTree`][moodspy.MotifTree] with a logo next to each leaf.

    Logos are aligned to each other (see [`align_motifs`][moodspy.align_motifs])
    unless ``align=False``. Pass a list of motifs to build the tree with
    default [`motif_tree`][moodspy.motif_tree] settings first.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from scipy.cluster.hierarchy import dendrogram

    from .compare import MotifTree, align_motifs, motif_tree

    plt = _plt()
    if not isinstance(tree, MotifTree):
        tree = motif_tree(tree)
    n = len(tree.motifs)
    labels = tree.labels
    if logos:
        if align and n > 1:
            al = align_motifs(tree.motifs, pseudocount=pseudocount, **align_kw)
            mats = dict(zip(al.table.index, al.padded(kind, pseudocount)))
            rc = dict(zip(al.table.index, al.table["strand"] == "-"))
            width = al.width
        else:
            width = max(m.length for m in tree.motifs)
            mats, rc = {}, {}
            for lab, m in zip(labels, tree.motifs):
                full = np.zeros((4, width))
                full[:, : m.length] = _heights(m, kind, pseudocount)
                mats[lab], rc[lab] = full, False
        fig, (ax_t, ax_l) = plt.subplots(
            1, 2, figsize=(3 + 0.3 * width, max(2.0, row_height * n)),
            gridspec_kw={"width_ratios": [1.0, max(1.0, 0.12 * width)], "wspace": 0.02},
        )
    else:
        fig, ax_t = plt.subplots(figsize=(4, max(2.0, row_height * n)))
    d = dendrogram(tree.linkage, orientation="left", labels=labels, ax=ax_t, no_labels=logos,
                   color_threshold=0, above_threshold_color="0.3")
    for side in ("top", "right", "left"):
        ax_t.spines[side].set_visible(False)
    ax_t.set_xlabel("distance")
    if not logos:
        fig.tight_layout()
        return fig
    # dendrogram leaves sit at y = 5, 15, 25, ... from the bottom
    ax_l.set_ylim(0, 10 * n)
    ax_l.set_xlim(-0.5, width + 0.5 + 0.45 * width)
    ax_l.axis("off")
    scale = 8.0 / (2.0 if kind == "ICM" else 1.0)
    for k, lab in enumerate(d["ivl"]):
        before = len(ax_l.patches)
        _draw_matrix(ax_l, mats[lab] * scale, kind, colours or DNA_COLOURS, font, 0.01 * scale)
        _shift(ax_l.patches[before:], 10 * k + 1)  # up to this leaf's row
        ax_l.text(width + 0.8, 10 * k + 5, lab + (" [RC]" if rc[lab] else ""), va="center", fontsize=9)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=min(0.3, 0.7 / fig.get_figheight()), top=0.98)
    return fig


def _shift(patches, dy: float) -> None:
    from matplotlib.transforms import Affine2D

    for p in patches:
        p.set_path(Affine2D().translate(0, dy).transform_path(p.get_path()))
