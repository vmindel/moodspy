"""Small datasets bundled with moodspy."""

from __future__ import annotations

from importlib.resources import files

from .io import read_motifs
from .motifs import Motif


def example_motifs() -> list[Motif]:
    """Three JASPAR motifs (KLF5, RELA, STAT1), as in motifmatchr's ``example_motifs``."""
    return read_motifs(files("moodspy") / "data" / "example_motifs.jaspar")
