# Peaks and genomes

The most common use is annotating ChIP-seq or ATAC-seq peaks with motif matches. You pass
the regions and a genome FASTA, and moodspy fetches the sequences.

This page uses a small synthetic genome so that it runs anywhere. With real data, replace
`toy_genome.fa` with e.g. `hg38.fa`.

```python
import numpy as np
import pandas as pd
import moodspy as mp

rng = np.random.default_rng(1)
genome = {c: "".join(rng.choice(list("ACGT"), p=[0.3, 0.2, 0.2, 0.3], size=100_000))
          for c in ("chr1", "chr2")}

# plant a KLF5 site at chr1:10,000 and a RELA site at chr2:20,000
klf5, rela = "GCCCCGCCCC", "GGGAATTTCC"
genome["chr1"] = genome["chr1"][:10_000] + klf5 + genome["chr1"][10_010:]
genome["chr2"] = genome["chr2"][:20_000] + rela + genome["chr2"][20_010:]

with open("toy_genome.fa", "w") as fh:
    for chrom, seq in genome.items():
        fh.write(f">{chrom}\n")
        for i in range(0, len(seq), 60):
            fh.write(seq[i:i + 60] + "\n")
```

## Regions

Regions can be a DataFrame with `chrom`, `start` and `end` columns, a BED file, a pyranges
object, or `"chr1:100-200"` strings. They are 0-based and half-open (BED convention):

```python
peaks = pd.DataFrame({
    "chrom": ["chr1", "chr1", "chr2", "chr2"],
    "start": [9_800, 50_000, 19_750, 70_000],
    "end":   [10_300, 50_500, 20_250, 70_500],
})
motifs = mp.example_motifs()
res = mp.match_motifs(motifs, peaks, genome="toy_genome.fa", out="scores")
print(res.sequences)
print(res.to_dataframe("counts"))
```

```text
                 chrom  start    end
seq
chr1:9800-10300   chr1   9800  10300
chr1:50000-50500  chr1  50000  50500
chr2:19750-20250  chr2  19750  20250
chr2:70000-70500  chr2  70000  70500
motif             MA0599.1_KLF5  MA0107.1_RELA  MA0137.3_STAT1
seq
chr1:9800-10300               1              0               0
chr1:50000-50500              1              0               0
chr2:19750-20250              0              3               0
chr2:70000-70500              0              0               0
```

The first call writes `toy_genome.fa.fai` next to the FASTA, the same index `samtools
faidx` builds, and reuses it afterwards. For a read-only reference directory, the index
goes to `~/.cache/moodspy` instead, so it is never rebuilt on each call. moodspy reads the FASTA through a memory map, so
it never loads the whole genome into memory.

The same regions as a BED file give identical results:

```python
peaks.to_csv("peaks.bed", sep="\t", header=False, index=False)
res_bed = mp.match_motifs(motifs, "peaks.bed", genome="toy_genome.fa", out="scores")
print((res_bed.counts != res.counts).nnz)
```

```text
0
```

## Hit positions in genome coordinates

For genomic input, `out="positions"` reports genome coordinates, the chromosome, and the
peak each hit came from:

```python
hits = mp.match_motifs(motifs, peaks, genome="toy_genome.fa", out="positions")
print(hits)
```

```text
           motif               seq chrom  start    end strand      score
0  MA0599.1_KLF5   chr1:9800-10300  chr1  10000  10010      +  18.053219
1  MA0599.1_KLF5  chr1:50000-50500  chr1  50293  50303      +  11.967769
2  MA0107.1_RELA  chr2:19750-20250  chr2  19774  19784      +  12.769978
3  MA0107.1_RELA  chr2:19750-20250  chr2  20000  20010      +  17.117718
4  MA0107.1_RELA  chr2:19750-20250  chr2  20001  20011      -  11.908685
```

```python
fa = mp.FastaFile("toy_genome.fa")
top = hits.sort_values("score").iloc[-1]
print(top.motif, top.chrom, top.start, top.end, fa.fetch(top.chrom, top.start, top.end))
```

```text
MA0599.1_KLF5 chr1 10000 10010 GCCCCGCCCC
```

To write the hits as BED:

```python
hits[["chrom", "start", "end", "motif", "score", "strand"]].to_csv(
    "hits.bed", sep="\t", header=False, index=False)
print(open("hits.bed").readline().strip())
```

```text
chr1	10000	10010	MA0599.1_KLF5	18.05321891056414	+
```

!!! warning "Categorical columns"
    `motif`, `seq`, `chrom` and `strand` are pandas categoricals. Pass `observed=True`
    when grouping by them. Otherwise pandas < 3 returns every combination of categories,
    including empty ones.

```python
best = hits.loc[hits.groupby(["seq", "motif"], observed=True)["score"].idxmax()]
print(best[["seq", "motif", "start", "score"]])
```

```text
                seq          motif  start      score
0   chr1:9800-10300  MA0599.1_KLF5  10000  18.053219
1  chr1:50000-50500  MA0599.1_KLF5  50293  11.967769
3  chr2:19750-20250  MA0107.1_RELA  20000  17.117718
```

## Genome background

`bg="genome"` uses the base composition of the whole FASTA. The counts take 1–2 s to compute
for hg38 and are cached on disk afterwards. You can also compute them yourself and pass them
explicitly:

```python
counts = fa.base_counts()
bg = counts / counts.sum()
print(bg.round(4))
res_bg = mp.match_motifs(motifs, peaks, genome=fa, bg=bg)
print(res_bg)
```

```text
[0.3002 0.2005 0.1992 0.3001]
MotifMatches(4 sequences x 3 motifs; assays: matches; p_cutoff=5e-05)
```

## Other genome objects

You can pass a `pysam.FastaFile` (which also reads bgzip-compressed FASTA), a
`pyfaidx.Fasta`, or a plain `{chrom: sequence}` dict in place of a path:

```python
res_dict = mp.match_motifs(motifs, peaks, genome=genome, out="scores")
print((res_dict.counts != res.counts).nnz)
```

```text
0
```

```py
import pysam
res = mp.match_motifs(motifs, peaks, genome=pysam.FastaFile("hg38.fa.gz"))
```

Regions that run past a chromosome end raise `ValueError` instead of being clipped.
Chromosome names must match the FASTA exactly (`chr1` vs `1`).
