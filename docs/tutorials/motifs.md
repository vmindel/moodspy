# Working with motifs

## Building a motif

A [`Motif`][moodspy.Motif] wraps a `(4, length)` matrix with rows A, C, G, T. It can hold
counts, probabilities or log-odds scores:

```python
import numpy as np
import pandas as pd
import moodspy as mp

counts = np.array([
    [ 0, 18,  0,  1],   # A
    [ 9,  0,  0, 17],   # C
    [ 9,  0, 19,  0],   # G
    [ 1,  1,  0,  1],   # T
])
m = mp.Motif(counts, name="TOY", id="X0001.1")
print(m, m.label)
```

```text
Motif('X0001.1_TOY', length=4, kind='counts', consensus='CAGC') X0001.1_TOY
```

The kind is inferred the way motifmatchr's `pwmType` does it:

- negative values mean log-odds;
- columns summing to 1 mean probabilities;
- anything else means counts.

`(length, 4)` arrays and pandas DataFrames with A/C/G/T labels also work:

```python
df = pd.DataFrame(counts.T, columns=list("ACGT"))
print(mp.Motif(df).kind, mp.Motif(counts / counts.sum(axis=0)).kind)
```

```text
counts probability
```

Pass `kind=` explicitly when the guess could be wrong. For example, a log-odds matrix
without negative entries would be read as counts.

## Log-odds scores

Scanning uses `log2(p / background)`. Counts are smoothed with the TFBSTools pseudocount,
`(n + 0.25 × 0.8) / (N + 0.8)`, so that zero counts don't give infinite penalties:

```python
print(m.log_odds().round(2))
print(m.log_odds(background=[0.3, 0.2, 0.2, 0.3]).round(2))
```

```text
[[-4.63  1.88 -4.63 -2.04]
 [ 0.89 -4.63 -4.63  1.8 ]
 [ 0.89 -4.63  1.96 -4.63]
 [-2.04 -2.04 -4.63 -2.04]]
[[-4.89  1.62 -4.89 -2.31]
 [ 1.22 -4.31 -4.31  2.12]
 [ 1.22 -4.31  2.28 -4.31]
 [-2.31 -2.31 -4.89 -2.31]]
```

## P-value thresholds

Hits are placements whose score reaches the threshold for `p_cutoff`. The threshold is
computed exactly by MOODS' dynamic programme:

```python
pwm = m.log_odds()
for p in (1e-2, 1e-3, 1e-4):
    print(p, round(mp.threshold_from_p(pwm, p), 3))
print("max score", pwm.max(axis=0).sum().round(3))
```

```text
0.01 3.587
0.001 5.056
0.0001 5.056
max score 6.525
```

This toy motif is only 4 bp long. Its best word, `CAGC`, has probability 1/256 ≈ 0.004
under a uniform background. Any p below that admits only the best word, so the threshold
for 1e-3 and 1e-4 sits between the best and second-best scores.

## Motif files

`read_motifs` detects JASPAR, raw `.pfm`, MEME, HOMER and TRANSFAC from the file contents:

```python
mp.write_meme(mp.example_motifs(), "example.meme")
print("".join(open("example.meme").readlines()[:13]))
again = mp.read_motifs("example.meme")
print(again[0], again[0].nsites)
```

```text
MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.2500 C 0.2500 G 0.2500 T 0.2500

MOTIF MA0599.1 KLF5
letter-probability matrix: alength= 4 w= 10 nsites= 13611 E= 0
0.104989 0.148630 0.556315 0.190067
0.000000 0.874293 0.000000 0.125707

Motif('MA0599.1_KLF5', length=10, kind='probability', consensus='GCCCCGCCCC') 13611.0
```

MEME and HOMER files hold probabilities. The pseudocount is scaled by `nsites`, which
defaults to 20 when the file doesn't give it.

Downloads from [JASPAR](https://jaspar.elixir.no/downloads/) in "JASPAR" format, e.g.
`JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt`, can be read directly:

```py
motifs = mp.read_motifs("JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt")
```

Motifs from [pyjaspar](https://github.com/asntech/pyjaspar) or Biopython can be passed to
`match_motifs` as they are:

```py
from pyjaspar import jaspardb
motifs = jaspardb().fetch_motifs(collection="CORE", tax_group=["vertebrates"])
res = mp.match_motifs(motifs, peaks, genome="hg38.fa")
```

## Labels

Results are labelled with `Motif.label`, which is `id_name` when both exist. A dict of
motifs uses its keys as labels:

```python
seq = "TTTCAGGTTTT" * 3
res = mp.match_motifs({"my_motif": counts, "klf5": mp.example_motifs()[0]}, seq, bg="even")
print(list(res.motifs.index))
```

```text
['my_motif', 'klf5']
```
