# Scanners and performance

## Reusing a scanner

Each call to `match_motifs` computes thresholds and builds a lookup table. That takes
about a second for a full JASPAR collection. When you scan many batches of sequences
with the same motifs and background, build a [`Scanner`][moodspy.Scanner] once:

```python
import numpy as np
import moodspy as mp

rng = np.random.default_rng(3)
batches = [["".join(rng.choice(list("ACGT"), size=500)) for _ in range(200)] for _ in range(3)]

sc = mp.Scanner(mp.example_motifs(), background=[0.25] * 4, p_cutoff=1e-4)
print(sc, sc.thresholds.round(3))
for batch in batches:
    counts, best = sc.summarize(batch)   # sparse (n_seqs, n_motifs) counts and best scores
    print(counts.sum(axis=0))
```

```text
Scanner(3 motifs, p_cutoff=0.0001, w=7) [10.838  9.902  9.188]
[[14 21 19]]
[[15 23 22]]
[[25 11 11]]
```

!!! warning "The Scanner background defaults to uniform"
    `match_motifs` defaults to `bg="subject"`, but `Scanner` has no subject at
    construction time, so its default background is uniform. Pass `background=`
    explicitly.

`Scanner.scan` returns every hit. Its `seq` column is the integer position in the input
list:

```python
hits = sc.scan(batches[0])
print(hits.head())
```

```text
   seq           motif  start  end strand      score
0    0   MA0107.1_RELA     25   35      +  14.183245
1    1  MA0137.3_STAT1    477  488      +  11.516745
2    2   MA0107.1_RELA     43   53      -  11.071035
3    3  MA0137.3_STAT1    478  489      -  11.677980
4    5   MA0107.1_RELA    128  138      -  10.910509
```

Fixed score cutoffs instead of p-values:

```python
sc_fixed = mp.Scanner(mp.example_motifs(), thresholds=[8.0, 8.0, 8.0])
print(len(sc_fixed.scan(batches[0])), sc_fixed.p_cutoff)
```

```text
163 None
```

## Threads

`n_jobs=None`, the default, uses all cores. Sequences are processed in parallel chunks,
and sequences longer than 1 Mb are split into overlapping pieces, so a single chromosome
also uses every core. Results are the same for any `n_jobs`:

```python
a = mp.Scanner(mp.example_motifs(), p_cutoff=1e-3, n_jobs=1).scan(batches[0])
b = mp.Scanner(mp.example_motifs(), p_cutoff=1e-3, n_jobs=4).scan(batches[0])
print(a.equals(b))
```

```text
True
```

Inside multiprocessing, joblib or dask workers, pass `n_jobs=1` so that the workers
don't oversubscribe the cores.

## The window size `w`

`w` sets the size of the lookup table (`4**w` entries). Larger windows filter more
candidate positions per lookup but take longer to build. It never changes which hits are
found. Scores can differ in the last floating-point digit, because a different window
sums the matrix columns in a different order:

```python
ref = mp.Scanner(mp.example_motifs(), p_cutoff=1e-3, w=7).scan(batches[1])
cols = ["seq", "motif", "start", "end", "strand"]
for w in (4, 9):
    other = mp.Scanner(mp.example_motifs(), p_cutoff=1e-3, w=w).scan(batches[1])
    print(w, other[cols].equals(ref[cols]), np.abs(other.score - ref.score).max() < 1e-12)
```

```text
4 True True
9 True True
```

The default `w=7` (as in motifmatchr) suits typical inputs. `w=8` or `w=9` is faster for
100k+ peaks or whole genomes.

## Timings

For 20,000 sequences × 500 bp and 200 motifs on an Apple M4 Pro (12 cores), including setup
(see `benchmarks/` in the repository):

| | motifmatchr 1.30 | moodspy, 1 thread | moodspy, 12 threads |
|---|---:|---:|---:|
| matches / scores | 3.3 s | 2.8 s | 0.45 s |
| positions, genomic ranges | 4.0 s | 2.9 s | 0.5 s |

The first call in a fresh environment compiles the numba kernels, which takes a few
seconds. The compiled code is cached on disk after that.

## Whole genomes

Scan one chromosome at a time and write the hits out as you go. Background hits alone
number about `2 × length × p_cutoff` per motif, roughly 60k per motif on hg38 at 1e-5:

```py
g = mp.FastaFile("hg38.fa")
counts = g.base_counts()
sc = mp.Scanner(motifs, counts / counts.sum(), p_cutoff=1e-5, w=8)
for chrom in g.keys():
    hits = sc.scan([g.fetch(chrom)])
    hits["chrom"] = chrom
    hits.to_csv(f"hits_{chrom}.tsv.gz", sep="\t", index=False)
```
