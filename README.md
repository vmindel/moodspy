# moodspy

Fast position weight matrix (PWM) motif matching for Python. It is a modern rewrite of
[MOODS](https://github.com/jhkorhonen/MOODS) with an API modelled on the R/Bioconductor
package [motifmatchr](https://github.com/GreenleafLab/motifmatchr).

* **No compiler, no SWIG.** Pure Python with [numba](https://numba.pydata.org) kernels, so
  `pip install` just works on Python 3.10+ on Linux, macOS and Windows.
* **The MOODS algorithms.** It uses MOODS' exact dynamic-programming p-value thresholds and its
  lookahead-filtration multi-matrix scanner, ported line by line where it matters.
* **Multithreaded.** It scales across all cores, and long chromosomes are split into
  seamlessly overlapping pieces.
* **motifmatchr semantics.** It returns the same matches, best scores, hit counts and positions
  (verified against motifmatchr 1.30; see [Differences](#differences-from-motifmatchr--moods)).
* **Python-native I/O.** It reads JASPAR, MEME, HOMER and TRANSFAC motifs, takes sequences
  as strings, dicts or FASTA files, takes regions as pandas DataFrames, BED files,
  pyranges objects or AnnData peak names, and writes results as SciPy sparse matrices,
  pandas or AnnData.

## Installation

```bash
pip install -e .            # from this directory
pip install -e ".[test]"    # + pytest and anndata
```

Dependencies: numpy, scipy, pandas, numba. `anndata` is optional (for `.to_anndata()` and
AnnData input).

## Quick start

```python
import moodspy as mp

motifs = mp.read_motifs("JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt")

# 1. Peaks (0-based, half-open BED coordinates) + genome FASTA
peaks = mp.read_bed("peaks.bed")                 # or any DataFrame with chrom/start/end
res = mp.match_motifs(motifs, peaks, genome="hg38.fa")
res                    # e.g. MotifMatches(50000 sequences x 879 motifs; assays: matches; p_cutoff=5e-05)
res.matches            # scipy.sparse CSR, bool, peaks x motifs
res.motifs             # id, name, length, score threshold per motif
res.sequences          # chrom, start, end per peak

# 2. Best score and number of hits as well
res = mp.match_motifs(motifs, peaks, genome="hg38.fa", out="scores")
res.scores, res.counts
adata = res.to_anndata()              # obs = peaks, var = motifs, layers = scores/counts

# 3. Every hit, one row each (genomic coordinates)
hits = mp.match_motifs(motifs, peaks, genome="hg38.fa", out="positions")
hits.columns           # motif, seq, chrom, start, end, strand, score
```

Plain sequences work the same way:

```python
mp.match_motifs(motifs, "ACGTGGGCGGGGC")                 # one string
mp.match_motifs(motifs, ["ACGT...", "GGCC..."])          # list
mp.match_motifs(motifs, {"enhancer1": "ACGT..."})        # named
mp.match_motifs(motifs, "sequences.fa")                  # FASTA (gzip ok)
```

scATAC peak matrices can be passed directly. Their `var_names` must look like `chr1:100-600`
or `chr1-100-600`:

```python
res = mp.match_motifs(motifs, adata, genome="hg38.fa")
```

### Motifs

Anything reasonable is accepted: a motif file path, a list of `mp.Motif`, a
`{label: matrix}` dict, raw `(4, L)` or `(L, 4)` arrays, pandas DataFrames with A/C/G/T
labels, or Biopython / [pyjaspar](https://github.com/asntech/pyjaspar) motifs:

```python
from pyjaspar import jaspardb
motifs = jaspardb().fetch_motifs(collection="CORE", tax_group=["vertebrates"])
res = mp.match_motifs(motifs, peaks, genome="hg38.fa")
```

`mp.Motif` holds counts, probabilities or log-odds scores. The kind is inferred like
motifmatchr's `pwmType`:

```python
m = mp.Motif(counts, name="KLF5", id="MA0599.1")
m.log_odds(background=[0.3, 0.2, 0.2, 0.3])   # log2(p / bg), TFBSTools pseudocount 0.8
m.reverse_complement(), m.consensus
mp.threshold_from_p(m.log_odds(), 5e-5)        # MOODS DP threshold
mp.write_jaspar(motifs, "out.jaspar"); mp.write_meme(motifs, "out.meme")
```

### Background

`bg="subject"` (default; computed from the scanned sequences), `"genome"` (whole FASTA),
`"even"`, or four numbers / a `{"A": ..}` dict. The background is used both for the
log-odds scores and for the p-value thresholds.

### Reusing a scanner

`match_motifs` builds thresholds and lookup tables every call. To scan many sequence sets
with the same motifs, build a `Scanner` once:

```python
sc = mp.Scanner(motifs, background=[0.29, 0.21, 0.21, 0.29], p_cutoff=1e-4)
hits = sc.scan(list_of_seqs)              # DataFrame of all hits
counts, best = sc.summarize(list_of_seqs) # sparse (n_seqs, n_motifs)
```

### Command line

```bash
moodspy scan -m motifs.jaspar -r peaks.bed -g hg38.fa -p 5e-5 -o hits.tsv
moodspy scan -m motifs.meme -s seqs.fa --out counts --bg 0.25,0.25,0.25,0.25 -o counts.tsv
```

## From motifmatchr

| motifmatchr (R)                                   | moodspy (Python)                                         |
|---------------------------------------------------|----------------------------------------------------------|
| `matchMotifs(pwms, peaks, genome = "hg38")`       | `match_motifs(motifs, peaks, genome="hg38.fa")`          |
| `out = "matches" / "scores" / "positions"`        | `out="matches" / "scores" / "positions"`                 |
| `p.cutoff = 5e-5`, `w = 7`, `bg = "subject"`      | `p_cutoff=5e-5`, `w=7`, `bg="subject"`                   |
| `motifMatches(res)`                               | `res.matches` (scipy CSR, bool)                          |
| `motifScores(res)`, `motifCounts(res)`            | `res.scores`, `res.counts`                               |
| `rowRanges(res)`, `colData(res)`                  | `res.sequences`, `res.motifs`                            |
| `SummarizedExperiment`                            | `res.to_anndata()`                                       |
| `GRangesList` of positions                        | one long DataFrame (`motif, seq, chrom, start, end, strand, score`) |
| 1-based, closed ranges                            | **0-based, half-open** (BED / Python convention)         |

## Differences from motifmatchr / MOODS

moodspy reproduces motifmatchr exactly with `motifmatchr_compat=True`, with one exception
(item 2). The test suite checks this against fixtures generated by motifmatchr 1.30
(`tests/data/make_motifmatchr_fixtures.R`). By default moodspy fixes two problems:

1. **Background correction sign (motifmatchr bug).** `motifmatchr:::convert_pwm` subtracts
   `log2(bg_pwm) - log2(bg)` from the log2 PWM instead of adding it. A count matrix is
   therefore scored as `log2(16 · p · bg)` instead of `log2(p / bg)`, which rewards bases
   that are common in the background instead of penalizing them. Thresholds are still
   computed under the true background, so false-positive rates hold, but scores and
   site rankings shift with sequence composition. The default `bg="subject"` is
   affected, uniform backgrounds are not. moodspy scores `log2(p / bg)`;
   `motifmatchr_compat=True` restores the old behaviour.
2. **Hits at sequence ends (MOODS bug).** For motifs shorter than the window `w`, MOODS
   skips hits that end exactly at the end of a sequence or right before an `N`
   (`size < end - i` should be `<=`). moodspy reports them in both modes.

Other conventions: both strands are always scanned. Reverse-strand hits report the
forward-strand start of the site and the reverse-complement matrix score. Characters other
than `ACGTacgt` (N, IUPAC codes) never match. Scores are log2 odds.

## Performance

These are timings on an Apple M4 Pro (12 cores) for 20,000 sequences × 500 bp and 200 motifs
at p = 5e-5, using `benchmarks/`. All of them include threshold computation and table
building. Both tools find the same 284,536 hits (moodspy finds 172 more, the short-motif
end-of-sequence hits described above).

| task                               | motifmatchr 1.30 | moodspy, 1 thread | moodspy, 12 threads |
|------------------------------------|-----------------:|------------------:|--------------------:|
| `out="matches"`                    | 3.3 s            | 2.8 s             | 0.45 s              |
| `out="scores"`                     | 3.3 s            | 2.8 s             | 0.45 s              |
| `out="positions"`, genomic ranges  | 4.0 s            | 2.9 s             | 0.5 s               |

Single-threaded, the two are close, since both run the same MOODS algorithm. Most of
moodspy's gain comes from multithreading.

Avoid motifmatchr's `positions` output on plain sequences (a character vector or
`DNAStringSet`). It builds a separate `IRanges` object for every (motif, sequence) pair,
about 1 ms each, so it took 94 s for 400 sequences and 200 motifs. The scan itself, which
is MOODS' C++ code, takes 3.2 s for all 20,000 sequences. moodspy returns one DataFrame
for either kind of input, so this cost doesn't apply.

The first call in a fresh environment also compiles the numba kernels, which takes a few
seconds. The compiled code is cached on disk after that. For very large inputs, `w=8` or
`w=9` trades a slightly slower setup for faster scanning. The results never depend on `w`.

## How it works

1. **Thresholds.** Each log-odds matrix is rounded to a grid of 1/2000 score units. The exact
   null score distribution under the background is computed by dynamic programming, and
   the threshold is the lowest grid score whose upper tail is ≤ `p_cutoff`. This is MOODS'
   `threshold_from_p`.
2. **Lookup table.** Every matrix and its reverse complement gets a window of `w` adjacent
   columns, chosen to maximize the expected score difference. For each of the `4^w`
   possible window contents, the table lists the matrices that could still reach their
   threshold.
3. **Scan.** A rolling `w`-mer code selects the candidate matrices at each position. Each
   candidate's remaining columns are added most informative first, and the candidate is
   dropped as soon as the threshold is out of reach.

## Development

```bash
pip install -e ".[test]"
pytest
# regenerate the motifmatchr fixtures (needs R + motifmatchr):
cd tests/data && python make_inputs.py && Rscript make_motifmatchr_fixtures.R
```

## License and credits

GPL-3.0-or-later. moodspy ports algorithms from MOODS by Pasi Rastas, Janne H. Korhonen and
Petri Martinmäki (GPLv3 / Biopython license), and follows the interface of motifmatchr by
Alicia Schep (GPL-3). If you use it, please cite MOODS:

> Korhonen J, Martinmäki P, Pizzi C, Rastas P, Ukkonen E. MOODS: fast search for position weight
> matrix matches in DNA sequences. *Bioinformatics* 25(23):3181–3182, 2009.
>
> Korhonen JH, Palin K, Taipale J, Ukkonen E. Fast motif matching revisited: high-order PWMs,
> SNPs and indels. *Bioinformatics* 33(4):514–521, 2017.
