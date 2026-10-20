"""moodspy: fast PWM motif matching (MOODS algorithms, motifmatchr-style API)."""

from importlib.metadata import PackageNotFoundError, version

from .genome import FastaFile, read_bed, read_fasta, regions_frame
from .io import (
    parse_homer,
    parse_jaspar,
    parse_meme,
    parse_pfm,
    parse_transfac,
    read_motifs,
    write_jaspar,
    write_meme,
)
from .match import MotifMatches, match_motifs
from .motifs import Motif, check_background
from .scanner import Scanner
from .thresholds import threshold_from_p, thresholds_from_p

try:
    __version__ = version("moodspy")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "FastaFile",
    "Motif",
    "MotifMatches",
    "Scanner",
    "check_background",
    "match_motifs",
    "parse_homer",
    "parse_jaspar",
    "parse_meme",
    "parse_pfm",
    "parse_transfac",
    "read_bed",
    "read_fasta",
    "read_motifs",
    "regions_frame",
    "threshold_from_p",
    "thresholds_from_p",
    "write_jaspar",
    "write_meme",
]
