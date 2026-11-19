"""Command-line interface: ``moodspy scan ...``."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .match import match_motifs
from .tracks import motif_tracks, write_bedgraph


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
    s.add_argument("-w", "--window", type=int, default=None,
                   help="lookup window size, 1-12 (affects speed, not the hits; default: automatic)")
    s.add_argument("--pseudocount", type=float, default=0.8)
    s.add_argument("--motifmatchr-compat", action="store_true",
                   help="reproduce motifmatchr scores exactly (inverted background correction)")
    s.add_argument("-j", "--jobs", type=int, default=None, help="threads (default: all cores)")
    s.add_argument("-o", "--output", default="-", help="output TSV (default stdout)")

    t = sub.add_parser(
        "track",
        help="score selected motifs at every position (1-bp step, no threshold)",
        description="Score motifs at every position of sequences or regions, for plotting "
        "motif affinity along peaks as a track. Writes a long TSV, or one bedGraph per motif.",
    )
    t.add_argument("-m", "--motifs", required=True, help="motif file (JASPAR, MEME, HOMER, TRANSFAC, PFM)")
    t.add_argument("--select", help="comma-separated motif labels, IDs or names to keep (default: all)")
    src = t.add_mutually_exclusive_group(required=True)
    src.add_argument("-s", "--sequences", help="FASTA file of sequences to score")
    src.add_argument("-r", "--regions", help="BED file of regions (requires --genome)")
    t.add_argument("-g", "--genome", help="indexed or plain FASTA genome")
    t.add_argument("--bg", type=_bg, default="subject", help="subject, genome, even or 'A,C,G,T'")
    t.add_argument("--strand", choices=["best", "both", "+", "-"], default="best",
                   help="best: higher-scoring strand per position (default); both: one row per strand")
    t.add_argument("-p", "--p-cutoff", type=float, default=5e-5,
                   help="only sets the 'hit' column (score >= threshold at this p); nothing is filtered")
    t.add_argument("--pseudocount", type=float, default=0.8)
    t.add_argument("--motifmatchr-compat", action="store_true",
                   help="reproduce motifmatchr scores exactly (inverted background correction)")
    t.add_argument("-j", "--jobs", type=int, default=None, help="threads (default: all cores)")
    t.add_argument("-o", "--output", default="-",
                   help="output TSV (default stdout), or with --bedgraph a path containing {motif}")
    t.add_argument("--bedgraph", action="store_true",
                   help="write bedGraph (one file per motif) instead of a TSV")
    t.add_argument("--value", choices=["score", "pvalue", "neglog10p"], default="score",
                   help="bedGraph value (default score)")
    t.add_argument("--anchor", choices=["center", "start"], default="center",
                   help="bedGraph base a placement is drawn at (default center of the motif)")
    return p


def _track(args) -> int:
    if args.bedgraph and (args.output == "-" or "{motif}" not in args.output):
        print("moodspy: --bedgraph needs -o with a {motif} placeholder, e.g. tracks.{motif}.bedgraph",
              file=sys.stderr)
        return 2
    select = [x.strip() for x in args.select.split(",") if x.strip()] if args.select else None
    try:
        res = motif_tracks(
            args.motifs,
            args.sequences or args.regions,
            genome=args.genome,
            select=select,
            bg=args.bg,
            strand=args.strand,
            p_cutoff=args.p_cutoff,
            pseudocount=args.pseudocount,
            motifmatchr_compat=args.motifmatchr_compat,
            n_jobs=args.jobs,
        )
        if args.bedgraph:
            for path in write_bedgraph(res, args.output, value=args.value, anchor=args.anchor):
                print(path, file=sys.stderr)
            return 0
    except (KeyError, ValueError) as err:
        print(f"moodspy: {err.args[0] if err.args else err}", file=sys.stderr)
        return 2
    # the scanner's own rule, so short motifs whose best word misses -p match `moodspy scan`
    thr = res.attrs["thresholds"]
    res["hit"] = res["score"].to_numpy() >= thr.to_numpy()[res["motif"].cat.codes.to_numpy()]
    dest = sys.stdout if args.output == "-" else args.output
    res.to_csv(dest, sep="\t", index=False, float_format="%.6g")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subject = args.sequences or args.regions
    if args.regions and not args.genome:
        print("moodspy: --regions requires --genome", file=sys.stderr)
        return 2
    if args.command == "track":
        return _track(args)
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
