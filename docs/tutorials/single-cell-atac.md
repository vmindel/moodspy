# Single-cell ATAC

In scATAC-seq analyses (scanpy, muon, SnapATAC2), peaks are usually the `var` axis of an
AnnData object, with names like `chr1:1000-1500`. You can pass such an object to
`match_motifs` directly. The result has one row per peak, in `var` order, ready to store
in `adata.varm`.

This page uses a toy genome and random counts so that it runs anywhere.

```python
import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad
import moodspy as mp

rng = np.random.default_rng(2)
chroms = {c: "".join(rng.choice(list("ACGT"), p=[0.3, 0.2, 0.2, 0.3], size=200_000))
          for c in ("chr1", "chr2")}
with open("genome.fa", "w") as fh:
    for c, s in chroms.items():
        fh.write(f">{c}\n" + "\n".join(s[i:i + 80] for i in range(0, len(s), 80)) + "\n")

starts = np.arange(0, 195_000, 1_000)
peaks = [f"{c}:{s}-{s + 500}" for c in chroms for s in starts]
adata = ad.AnnData(
    X=sp.random(300, len(peaks), density=0.05, format="csr", random_state=0, dtype=np.float32),
    var=pd.DataFrame(index=peaks),
)
print(adata)
```

```text
AnnData object with n_obs × n_vars = 300 × 390
    layers: None (.X)
```

## Peak x motif annotation

```python
motifs = mp.example_motifs()
res = mp.match_motifs(motifs, adata, genome="genome.fa", out="scores", bg="genome")
print(res)
assert res.shape[0] == adata.n_vars

adata.varm["motif_match"] = res.matches
adata.uns["motif_names"] = list(res.motifs.index)
print(adata)
```

```text
MotifMatches(390 sequences x 3 motifs; assays: matches, scores, counts; p_cutoff=5e-05)
AnnData object with n_obs × n_vars = 300 × 390
    uns: 'motif_names'
    varm: 'motif_match'
    layers: None (.X)
```

`bg="genome"` keeps the annotation independent of which peaks are in the object. With
`bg="subject"`, a subset of peaks would get slightly different thresholds.

If the peak names can't be parsed (e.g. `peak_123`), pass the coordinate columns instead.
The DataFrame's index is kept, so rows still line up with `var`:

```py
res = mp.match_motifs(motifs, adata.var[["chrom", "start", "end"]], genome="hg38.fa")
```

## Per-cell motif counts

Multiplying the cell × peak matrix by the peak × motif match matrix counts, for each cell,
the fragments in peaks that contain each motif:

```python
motif_counts = adata.X @ res.matches.astype(np.float32)   # cells x motifs, sparse
per_cell = pd.DataFrame(motif_counts.toarray(), index=adata.obs_names, columns=res.motifs.index)
print(per_cell.describe().loc[["mean", "std"]].round(2))
```

```text
motif  MA0599.1_KLF5  MA0107.1_RELA  MA0137.3_STAT1
mean            0.54           0.24            0.57
std             0.59           0.40            0.59
```

These raw counts scale with each cell's depth. A proper chromVAR-style deviation needs
GC- and accessibility-matched background peaks, which is the step after motif matching.

## A motif AnnData

`to_anndata()` turns the result itself into an AnnData object, with peaks as `obs`, motifs
as `var`, and scores and counts as layers:

```python
mad = res.to_anndata()
print(mad)
print(mad.var)
```

```text
AnnData object with n_obs × n_vars = 390 × 3
    obs: 'chrom', 'start', 'end'
    var: 'id', 'name', 'length', 'threshold'
    uns: 'motif_matching'
    layers: 'scores', 'counts', None (.X)
                      id   name  length  threshold
motif
MA0599.1_KLF5   MA0599.1   KLF5      10    11.1730
MA0107.1_RELA   MA0107.1   RELA      10    10.8125
MA0137.3_STAT1  MA0137.3  STAT1      11    11.2055
```
