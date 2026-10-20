"""Brute-force check of moodspy hits on a random sample of sequences.

Every placement of every motif on both strands is scored directly with numpy,
with no lookup tables and no early abandoning, and compared with
``moodspy.Scanner.scan``, using the same score matrices and thresholds.
Prints the number of agreeing, missing and extra hits and exits with 1 on
any difference.

    python check_hits.py motifs.jaspar seqs.fa [--n 50] [--p 1e-4] [--bg even|subject|A,C,G,T]
    python check_hits.py motifs.meme peaks.bed --genome hg38.fa --n 20

Brute force is slow (pure numpy), so keep --n small for long sequences.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

import moodspy as mp
from moodspy._alphabet import base_counts, encode, encode_many
from moodspy.genome import _Genome, read_fasta, regions_frame


def brute_force(seqs, pwms, thresholds, tol=1e-9):
    hits = set()
    for si, s in enumerate(seqs):
        codes = encode(s)
        valid = codes < 4
        for j, pwm in enumerate(pwms):
            length = pwm.shape[1]
            if len(s) < length:
                continue
            n = len(s) - length + 1
            idx = np.arange(n)[:, None] + np.arange(length)
            ok = valid[idx].all(axis=1)
            win = np.where(valid, codes, 0)[idx]
            for strand, mat in (("+", pwm), ("-", pwm[::-1, ::-1])):
                score = mat[win, np.arange(length)].sum(axis=1)
                for p in np.flatnonzero(ok & (score >= thresholds[j] - tol)):
                    hits.add((si, j, int(p), strand, round(float(score[p]), 6)))
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("motifs", help="motif file")
    ap.add_argument("subject", help="FASTA of sequences, or BED of regions with --genome")
    ap.add_argument("--genome", help="genome FASTA for BED input")
    ap.add_argument("--n", type=int, default=50, help="number of sequences to sample (default 50)")
    ap.add_argument("--p", type=float, default=1e-4, help="p-value cutoff (default 1e-4)")
    ap.add_argument("--bg", default="even", help="even, subject or A,C,G,T (default even)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    motifs = mp.read_motifs(args.motifs)
    if args.genome:
        regions = regions_frame(args.subject)
        seqs = _Genome(args.genome).fetch_many(regions.chrom, regions.start, regions.end)
    else:
        seqs = list(read_fasta(args.subject).values())
    rng = np.random.default_rng(args.seed)
    sample = [seqs[i] for i in rng.choice(len(seqs), size=min(args.n, len(seqs)), replace=False)]

    if args.bg == "even":
        bg = None
    elif args.bg == "subject":
        counts = base_counts(encode_many(sample)[0])
        bg = counts / counts.sum()
    else:
        bg = [float(x) for x in args.bg.split(",")]

    sc = mp.Scanner(motifs, bg, args.p, n_jobs=1)
    df = sc.scan(sample)
    ours = {(r.seq, sc.labels.index(r.motif), r.start, r.strand, round(r.score, 6)) for r in df.itertuples()}
    ref = brute_force(sample, sc.pwms, sc.thresholds)
    missing, extra = ref - ours, ours - ref
    print(f"{len(sample)} sequences, {len(motifs)} motifs, p={args.p}: "
          f"{len(ref & ours)} hits agree, {len(missing)} missing, {len(extra)} extra")
    for label, rows in (("missing", missing), ("extra", extra)):
        for si, j, p, strand, score in sorted(rows)[:10]:
            print(f"  {label}: seq {si} motif {sc.labels[j]} start {p} {strand} score {score}")
    return 1 if missing or extra else 0


if __name__ == "__main__":
    sys.exit(main())
