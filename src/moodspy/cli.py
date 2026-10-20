"""Command-line interface: ``moodspy scan ...``."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .match import match_motifs


def _bg(value: str):
    if value in ("subject", "genome", "even"):
        return value
    parts = [float(x) for x in value.replace(",", " ").split()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--bg takes subject/genome/even or 4 numbers 'A,C,G,T'")
    return parts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="moodspy", description="Fast PWM motif matching.")
    p.add_argument("--version", action="version", version=f"moodspy {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scan", help="scan sequences or genomic regions for motifs")
    s.add_argument("-m", "--motifs", required=True, help="motif file (JASPAR, MEME, HOMER, TRANSFAC, PFM)")
    src = s.add_mutually_exclusive_group(required=True)
    src.add_argument("-s", "--sequences", help="FASTA file of sequences to scan")
    src.add_argument("-r", "--regions", help="BED file of regions (requires --genome)")
    s.add_argument("-g", "--genome", help="indexed or plain FASTA genome")
    s.add_argument("-p", "--p-cutoff", type=float, default=5e-5, help="p-value cutoff (default 5e-5)")
    s.add_argument("--bg", type=_bg, default="subject", help="subject, genome, even or 'A,C,G,T'")
    s.add_argument(
        "--out",
        choices=["positions", "matches", "counts", "scores"],
        default="positions",
        help="positions: one row per hit; others: sequences x motifs table",
    )
    s.add_argument("-w", "--window", type=int, default=7, help="lookup window size (speed only)")
    s.add_argument("--pseudocount", type=float, default=0.8)
    s.add_argument("--motifmatchr-compat", action="store_true",
                   help="reproduce motifmatchr scores exactly (inverted background correction)")
    s.add_argument("-j", "--jobs", type=int, default=None, help="threads (default: all cores)")
    s.add_argument("-o", "--output", default="-", help="output TSV (default stdout)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subject = args.sequences or args.regions
    if args.regions and not args.genome:
        print("moodspy: --regions requires --genome", file=sys.stderr)
        return 2
    out = "positions" if args.out == "positions" else ("matches" if args.out == "matches" else "scores")
    res = match_motifs(
        args.motifs,
        subject,
        genome=args.genome,
        bg=args.bg,
        out=out,
        p_cutoff=args.p_cutoff,
        w=args.window,
        pseudocount=args.pseudocount,
        motifmatchr_compat=args.motifmatchr_compat,
        n_jobs=args.jobs,
    )
    if args.out == "positions":
        table, index = res, False
    else:
        table = res.to_dataframe(args.out)
        if args.out == "matches":
            table = table.astype(int)
        index = True
    dest = sys.stdout if args.output == "-" else args.output
    table.to_csv(dest, sep="\t", index=index, float_format="%.6g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
