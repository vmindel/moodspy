---
name: moodspy
description: Use for any task involving the moodspy library — PWM / transcription-factor motif matching in Python (a rewrite of MOODS with a motifmatchr-style API, plus a port of universalmotif). Triggers on scanning sequences, peaks, BED files or a genome FASTA for motif hits; motif x peak matrices for chromVAR-style or scATAC analysis; hit positions; p-value thresholds; reading or writing JASPAR, MEME, HOMER or TRANSFAC motifs; backgrounds; comparing, merging, clustering, trimming or plotting motifs (logos, Tomtom-style matching against a database); k-mer backgrounds, shuffled or simulated sequences, implanting motifs; porting R motifmatchr, universalmotif or MOODS-python code. This is a router skill — read the relevant file under references/ before writing moodspy code.
---

# moodspy

Finds every placement of a position weight matrix on both strands of DNA
whose log2-odds score reaches a p-value threshold. It does this for many
motifs and many sequences at once, multithreaded. This file only routes:
the reference files hold the signatures, defaults and silent failures. Read
the file for the task before writing code. Several defaults change results
without warning, for example the background, compat mode, and categorical
columns in pandas groupby.

## Task to reference file

| task | read |
|---|---|
| **`match_motifs`**: inputs it accepts, `out=` modes, the `MotifMatches` result, positions DataFrame | `references/match-motifs.md` |
| **peaks / regions + genome FASTA**: coordinates, BED, chrom names, `FastaFile`, pysam/pyfaidx | `references/genomes-regions.md` |
| **scATAC / AnnData**: peaks x motifs matrix into `adata.varm`, per-cell motif counts | `references/single-cell-atac.md` |
| **motifs**: building `Motif`, counts vs probabilities vs log-odds, pseudocounts, labels, file formats | `references/motifs-io.md` |
| **thresholds and background**: `p_cutoff`, `bg=`, `threshold_from_p`, negative thresholds, `motifmatchr_compat` | `references/thresholds-background.md` |
| **reusing a `Scanner`**, threads, `w`, memory, numba compile time, batching a genome | `references/scanner-performance.md` |
| **motif and sequence tools** (universalmotif port): IC, consensus, trimming, `compare_motifs` / `compare_motifs_lite`, merging, clustering, logos, `get_bkg`, shuffling, simulated sequences, implanting, masking | `references/motif-tools.md` |
| **porting R motifmatchr / MOODS-python** code or comparing against their results | `references/from-motifmatchr.md` |
| "**are these hits correct?**": independent brute-force check of a scan | `scripts/check_hits.py` |

`scripts/check_hits.py` (path relative to this skill's root) rescans a
random sample of sequences by brute force and compares the hits with
`Scanner.scan`. Run `python scripts/check_hits.py --help`.

## Conventions

- `import moodspy as mp`. The one entry point is
  `mp.match_motifs(motifs, subject, genome=None, *, bg="subject", out="matches", p_cutoff=5e-5, w=7, pseudocount=0.8, motifmatchr_compat=False, n_jobs=None)`.
  Everything after `genome` is keyword-only.
- Coordinates are **0-based, half-open** (BED / Python slicing) everywhere,
  in input regions and output positions. motifmatchr is 1-based and closed.
- Matrices are `(4, length)` with rows `A, C, G, T`. A `(length, 4)` array
  is transposed automatically. A 4x4 matrix is assumed to be `(4, length)`.
- Scores are log2 odds `log2(p / bg)`. Counts get the TFBSTools
  pseudocount: `(n + 0.25 * 0.8) / (N + 0.8)`.
- Both strands are always scanned. A `-` hit reports the forward-strand
  `start`/`end` of the site and the score of the reverse-complement matrix.
  Palindromic motifs therefore hit one site twice; `mp.remove_overlaps(hits)`
  collapses those pairs.
- Characters other than `ACGTacgt` (N, IUPAC codes) never match.
  Soft-masked lowercase does match.
- Results are SciPy CSR matrices (`matches` bool, `counts` int32,
  `scores` float64) with rows = sequences and columns = motifs. Positions
  come back as one long pandas DataFrame.
- `n_jobs=None` uses every core. Pass `n_jobs=1` inside multiprocessing,
  dask or joblib workers. The first call in a fresh environment compiles
  numba kernels, which takes a few seconds and is cached on disk afterwards.
