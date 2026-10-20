# Quick start

This page scans a few short sequences with the three example motifs from motifmatchr
(KLF5, RELA, STAT1) and walks through the three kinds of output.

```python
import moodspy as mp

motifs = mp.example_motifs()
for m in motifs:
    print(m)
```

```text
Motif('MA0599.1_KLF5', length=10, kind='counts', consensus='GCCCCGCCCC')
Motif('MA0107.1_RELA', length=10, kind='counts', consensus='GGGAATTTCC')
Motif('MA0137.3_STAT1', length=11, kind='counts', consensus='TTTCCAGGAAA')
```

`mp.example_motifs()` returns plain [`Motif`][moodspy.Motif] objects built from JASPAR count
matrices. Any motif file works the same way, for example `mp.read_motifs("JASPAR2024.txt")`.
See [Working with motifs](motifs.md).

## Which sequences contain which motifs?

```python
seqs = {
    "klf_site":  "TTTAGCCCCGCCCCTTTAGAGAT",
    "rela_site": "CAAGGGAATTTCCAAATCTGGT",
    "both":      "GCCCCGCCCCATTTGGAAATTCCC",
    "nothing":   "ATATATATATATATATATATATAT",
}
res = mp.match_motifs(motifs, seqs, bg="even")
print(res)
print(res.to_dataframe())
```

```text
MotifMatches(4 sequences x 3 motifs; assays: matches; p_cutoff=5e-05)
motif      MA0599.1_KLF5  MA0107.1_RELA  MA0137.3_STAT1
seq
klf_site            True          False           False
rela_site          False           True           False
both                True           True           False
nothing            False          False           False
```

The result is a [`MotifMatches`][moodspy.MotifMatches], the counterpart of
motifmatchr's `SummarizedExperiment`. `res.matches` is a sparse boolean matrix with
sequences as rows and motifs as columns. `to_dataframe()` converts it to a dense
`DataFrame`, which is only sensible for small inputs.

!!! note "Both strands"
    `rela_site` contains `GGGAATTTCC` and `both` contains its reverse complement
    `GGAAATTCCC`. Both strands are always scanned, so both count as RELA matches.

## Scores and counts

`out="scores"` adds the best log2-odds score and the number of hits per sequence and motif:

```python
res = mp.match_motifs(motifs, seqs, bg="even", out="scores")
print(res.to_dataframe("scores").round(2))
print(res.to_dataframe("counts"))
```

```text
motif      MA0599.1_KLF5  MA0107.1_RELA  MA0137.3_STAT1
seq
klf_site           15.54           0.00             0.0
rela_site           0.00          16.88             0.0
both               15.54          16.88             0.0
nothing             0.00           0.00             0.0
motif      MA0599.1_KLF5  MA0107.1_RELA  MA0137.3_STAT1
seq
klf_site               1              0               0
rela_site              0              2               0
both                   1              2               0
nothing                0              0               0
```

Empty cells in the sparse matrices mean *no hit*. `to_dataframe` fills them with 0,
so use `res.matches` to tell the two apart.

Each motif's score threshold is in `res.motifs`:

```python
print(res.motifs)
```

```text
                      id   name  length  threshold
motif
MA0599.1_KLF5   MA0599.1   KLF5      10     12.194
MA0107.1_RELA   MA0107.1   RELA      10     11.120
MA0137.3_STAT1  MA0137.3  STAT1      11     11.005
```

## Every hit

`out="positions"` returns one row per hit. Coordinates are 0-based and half-open, as in
Python slicing and BED files:

```python
hits = mp.match_motifs(motifs, seqs, bg="even", out="positions")
print(hits)
```

```text
           motif        seq  start  end strand      score
0  MA0599.1_KLF5   klf_site      4   14      +  15.543401
1  MA0107.1_RELA  rela_site      3   13      +  16.878390
2  MA0107.1_RELA  rela_site      4   14      -  12.190617
3  MA0599.1_KLF5       both      0   10      +  15.543401
4  MA0107.1_RELA       both     13   23      +  12.190617
5  MA0107.1_RELA       both     14   24      -  16.878390
```

```python
row = hits.iloc[0]
print(seqs[row.seq][row.start:row.end])
```

```text
GCCCCGCCCC
```

## The background

By default (`bg="subject"`), the nucleotide background comes from the sequences being
scanned, as in motifmatchr. It is used both for the log-odds scores and for the p-value
thresholds:

```python
res = mp.match_motifs(motifs, seqs)
print(res.background.round(3))
```

```text
[0.29  0.247 0.14  0.323]
```

These four short sequences are AT-rich, so the thresholds and scores differ from the
`bg="even"` run above. For results that don't depend on which sequences you scan
together, pass `bg="even"`, `bg="genome"` or explicit frequencies such as
`bg=[0.29, 0.21, 0.21, 0.29]`. [Thresholds and background](../api/thresholds.md) explains
the details.

## FASTA files

Paths work anywhere sequences are expected:

```python
import numpy as np

rng = np.random.default_rng(0)
with open("seqs.fa", "w") as fh:
    for i in range(1000):
        fh.write(f">s{i}\n" + "".join(rng.choice(list("ACGT"), size=300)) + "\n")

res = mp.match_motifs(motifs, "seqs.fa", out="scores")
print(res)
print(res.counts.sum(axis=0))
```

```text
MotifMatches(1000 sequences x 3 motifs; assays: matches, scores, counts; p_cutoff=5e-05)
[[23 32 27]]
```

These sequences are random, so every hit is a false positive. With 1000 × 300 bp × 2
strands at p = 5e-5, you expect about 30 hits per motif.
