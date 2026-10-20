# Coming from motifmatchr

## Side by side

=== "R (motifmatchr)"

    ```r
    library(motifmatchr)
    data(example_motifs)
    peaks <- GRanges(c("chr1", "chr2"), IRanges(c(76585873, 42772928), width = 500))
    res <- matchMotifs(example_motifs, peaks, genome = "BSgenome.Hsapiens.UCSC.hg19",
                       out = "scores", p.cutoff = 5e-5)
    motifMatches(res); motifScores(res); motifCounts(res)
    ```

=== "Python (moodspy)"

    ```py
    import pandas as pd, moodspy as mp
    peaks = pd.DataFrame({"chrom": ["chr1", "chr2"],
                          "start": [76585872, 42772927]})   # 1-based -> 0-based
    peaks["end"] = peaks.start + 500
    res = mp.match_motifs(mp.example_motifs(), peaks, genome="hg19.fa",
                          out="scores", p_cutoff=5e-5)
    res.matches; res.scores; res.counts
    ```

| motifmatchr | moodspy |
|---|---|
| `matchMotifs(pwms, subject, genome, bg, out, p.cutoff, w)` | `match_motifs(motifs, subject, genome, bg=, out=, p_cutoff=, w=)` |
| `motifMatches()` / `motifScores()` / `motifCounts()` | `.matches` / `.scores` / `.counts` (SciPy CSR) |
| `rowRanges()` / `colData()` | `.sequences` / `.motifs` |
| `SummarizedExperiment` | `.to_anndata()` |
| `GRangesList` of positions | one DataFrame with a `motif` column |
| 1-based closed ranges | 0-based half-open (subtract 1 from `start`) |
| BSgenome object or name | FASTA path or genome object |

## Reproducing motifmatchr numbers

motifmatchr has a sign error in its background correction (`convert_pwm`). It scores a
count matrix as `log2(16 · p · bg)` instead of `log2(p / bg)`. moodspy uses the correct
formula by default, and `motifmatchr_compat=True` reproduces motifmatchr. The two agree
whenever the background is uniform:

```python
import numpy as np
import moodspy as mp

motifs = mp.example_motifs()
rng = np.random.default_rng(4)
seqs = ["".join(rng.choice(list("ACGT"), p=[0.35, 0.15, 0.15, 0.35], size=400)) for _ in range(500)]

for bg in ("even", "subject"):
    new = mp.match_motifs(motifs, seqs, bg=bg, out="scores")
    old = mp.match_motifs(motifs, seqs, bg=bg, out="scores", motifmatchr_compat=True)
    print(bg, "differing scores:", (new.scores != old.scores).nnz,
          "| hits:", new.counts.sum(), "vs", old.counts.sum())
```

```text
even differing scores: 0 | hits: 40 vs 40
subject differing scores: 92 | hits: 76 vs 59
```

With a 70% AT background, the old formula favours A/T in every column, so AT-rich
sites score higher. Both modes compute thresholds under the true background, so both
call false positives at rate `p_cutoff`. Over 40 million random positions, the observed
rate was 5.0e-5 at p = 5e-5 in both modes. The hit counts above differ only by chance.
What changes is which sites pass and how they rank.

The moodspy test suite compares `motifmatchr_compat=True` results against reference
output from motifmatchr 1.30, and they agree exactly with one exception.

## The one remaining difference

MOODS, and therefore motifmatchr, misses hits of motifs shorter than the window `w`
(7 bp by default) that end exactly at the end of a sequence or just before an `N`.
moodspy reports them in both modes:

```python
short = mp.Motif(np.eye(4) * 20, name="ACGT")   # a 4 bp motif
hits = mp.match_motifs([short], ["TTTTACGT", "ACGTNNNN"], bg="even", out="positions", p_cutoff=1e-2)
print(hits)
```

```text
  motif   seq  start  end strand     score
0  ACGT  seq0      4    8      +  7.831087
1  ACGT  seq0      4    8      -  7.831087
2  ACGT  seq1      0    4      +  7.831087
3  ACGT  seq1      0    4      -  7.831087
```

`ACGT` is its own reverse complement, so each site is hit on both strands. motifmatchr
reports none of these four hits. Motifs of 7 bp or longer, which covers
nearly all of JASPAR, are unaffected.
