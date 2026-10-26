"""moodspy: fast PWM motif matching (MOODS algorithms, motifmatchr-style API)."""

from importlib.metadata import PackageNotFoundError, version

from .datasets import example_motifs
from .genome import FastaFile, read_bed, read_fasta, regions_frame
from .hits import remove_overlaps
from .io import (
    parse_hocomoco,
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
from .compare import (
    MotifAlignment,
    MotifTree,
    align_motifs,
    compare_motifs,
    compare_motifs_lite,
    merge_motifs,
    merge_similar,
    motif_tree,
)
from .manipulate import (
    average_ic,
    create_motif,
    icm,
    iupac_consensus,
    motif_ic,
    motif_table,
    round_motif,
    sample_sites,
    score_match,
    shuffle_motifs,
    trim_motifs,
)
from .sequences import (
    create_sequences,
    get_bkg,
    implant_motifs,
    mask_ranges,
    mask_seqs,
    revcomp,
    shuffle_sequences,
    write_fasta,
)
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
    "example_motifs",
    "match_motifs",
    "parse_hocomoco",
    "parse_homer",
    "parse_jaspar",
    "parse_meme",
    "parse_pfm",
    "parse_transfac",
    "read_bed",
    "read_fasta",
    "read_motifs",
    "regions_frame",
    "remove_overlaps",
    "threshold_from_p",
    "thresholds_from_p",
    "write_jaspar",
    "write_meme",
]
