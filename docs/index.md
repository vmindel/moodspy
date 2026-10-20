# moodspy

**Fast PWM motif matching for Python**, a modern rewrite of
[MOODS](https://github.com/jhkorhonen/MOODS) with an API modelled on the R/Bioconductor
package [motifmatchr](https://github.com/GreenleafLab/motifmatchr).

```py
import moodspy as mp

motifs = mp.read_motifs("JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt")
res = mp.match_motifs(motifs, "peaks.bed", genome="hg38.fa", out="scores")

res.matches    # peaks x motifs, sparse bool
res.scores     # best log2-odds score per peak and motif
res.counts     # number of hits
res.to_anndata()
```

- **No compiler, no SWIG.** It is pure Python with numba kernels, so `pip install` works
  everywhere.
- **The MOODS algorithms.** It uses exact dynamic-programming p-value thresholds and the
  lookahead-filtration multi-matrix scanner.
- **Multithreaded.** It is about 7× faster than motifmatchr on 12 cores, and on par with
  it on one core.
- **Same answers as motifmatchr.** With `motifmatchr_compat=True` it agrees exactly with
  motifmatchr 1.30, and by default it fixes two upstream bugs
  ([details](tutorials/from-motifmatchr.md)).
- **Python-native I/O.** It reads JASPAR, MEME, HOMER and TRANSFAC motifs, takes regions
  as DataFrames, BED, pyranges or AnnData, and returns SciPy sparse matrices, pandas or
  AnnData.
- **A [Claude Code skill](claude-skill.md)** ships with the package.

## Where next

| | |
|---|---|
| [Quick start](tutorials/quickstart.md) | scan sequences; the three output modes |
| [Peaks and genomes](tutorials/peaks-and-genomes.md) | BED / DataFrame regions with a FASTA genome |
| [Single-cell ATAC](tutorials/single-cell-atac.md) | AnnData peaks, `varm` annotation, per-cell counts |
| [Working with motifs](tutorials/motifs.md) | motif kinds, pseudocounts, thresholds, file formats |
| [Scanners and performance](tutorials/scanner.md) | reusing a scanner, threads, whole genomes |
| [Coming from motifmatchr](tutorials/from-motifmatchr.md) | translation table and differences |
| [API reference](api/match.md) | every public function and class |

Every code block with output in the tutorials is executed and checked against its
printed output by the test suite.
