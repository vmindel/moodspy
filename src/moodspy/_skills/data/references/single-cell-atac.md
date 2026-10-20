# scATAC / AnnData workflows

An AnnData whose `var_names` are peaks (`chr1:100-600`, `chr1-100-600` or
`chr1_100_600`) can be passed directly. The result has one row per var, in
var order.

```python
res = mp.match_motifs(motifs, adata, genome="hg38.fa", out="scores", bg="genome")
adata.varm["motif_match"] = res.matches            # (n_peaks, n_motifs) sparse bool
adata.uns["motif_names"] = list(res.motifs.index)
```

- `var_names` are used only when the first 20 all parse as regions.
  Otherwise moodspy falls back to `obs_names`, for peaks x cells
  orientation. Check `res.shape[0] == adata.n_vars` before assigning to
  `varm`.
- If peak names are not in a parseable format (e.g. `peak_123`), build a
  DataFrame from the var columns instead:
  `mp.match_motifs(motifs, adata.var[["chrom", "start", "end"]], genome=...)`.
  Its index is `adata.var_names`, so row order still matches.
- With a non-uniform background, moodspy's scores differ from
  motifmatchr's, because motifmatchr inverts the background correction
  (`references/thresholds-background.md`). Annotations made in R with
  motifmatchr and chromVAR are reproduced only with
  `motifmatchr_compat=True`.
- `bg="subject"` derives the background from the peaks being scanned, so
  results on a subset of peaks differ from results on all peaks. Use
  `bg="genome"` or explicit frequencies when combining batches.

## Per-cell motif accessibility

```python
import numpy as np, scipy.sparse as sp
X = sp.csr_matrix(adata.X)                          # cells x peaks counts
M = res.matches.astype(np.float32)                  # peaks x motifs
motif_counts = X @ M                                # cells x motifs (sparse): reads in peaks with each motif
```

Raw `motif_counts` scale with each cell's depth and with the motif's
frequency. They are **not** chromVAR deviations. chromVAR deviations need GC- and
accessibility-matched background peaks. moodspy only provides the
annotation matrix, which is the motifmatchr part of that pipeline.

## Memory

At the default `p_cutoff=5e-5` with ~800 JASPAR motifs, the
`peaks x motifs` match matrix typically has a few percent of entries set.
That is small as CSR, but **do not `.toarray()`** it for 500k peaks
(400M cells). Avoid `out="positions"` on whole peak sets unless you need
the positions, since it can return hundreds of millions of rows.
