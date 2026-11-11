# Scanner reuse and performance (moodspy.scanner)

```python
sc = mp.Scanner(
    motifs, background=None,       # None -> UNIFORM, unlike match_motifs' bg="subject"
    p_cutoff=5e-5, *, thresholds=None, w=None, pseudocount=0.8, log_base=2.0,
    precision=2000.0, motifmatchr_compat=False, n_jobs=None,
)
sc.thresholds, sc.labels, sc.pwms, sc.background
hits = sc.scan(seqs)               # DataFrame: seq (INT index), motif, start, end, strand, score
counts, best = sc.summarize(seqs)  # two CSR (n_seqs, n_motifs): int32 counts, float64 best score
```

`match_motifs` builds a new Scanner on every call: thresholds plus a
`4**w` lookup table, about 0.5 s for 800 motifs (mostly thresholds). Build one yourself when
scanning many batches with the same motifs and background.

- `Scanner.scan` returns `seq` as the **integer position** in `seqs`, not a
  name. `match_motifs` maps it to names.
- `thresholds=[...]` (one per motif) bypasses p-values. It is useful for
  fixed score cutoffs, and then `sc.p_cutoff` is `None`.
- `seqs` is a list of str; a single str is wrapped. Regions are not handled
  here, so fetch them first, e.g. `mp.FastaFile(...).fetch(...)`.

## Speed knobs

- **Threads**: `n_jobs=None` uses all cores, through Python threads running
  nogil numba kernels. Sequences over 1 Mb are split into overlapping
  pieces, so a single chromosome also parallelises. Inside
  multiprocessing, joblib or dask workers, set `n_jobs=1` to avoid
  oversubscription.
- **`w`** (1–12) changes speed and memory, never **which hits** are found.
  Scores may differ by ~1e-15 because the summation order changes, so
  compare positions exactly and scores with a tolerance. The
  table holds `4**w` entries times candidate motifs. The default
  `w=None` picks it per call from the total sequence length: 7 below
  1 Mb, 9 below 100 Mb, 10 above, reduced if the table would exceed
  256 MB. Tables are built on first use and cached on the Scanner. Set
  `w` only to pin memory or to compare runs. `match_motifs` uses 7 when
  `motifmatchr_compat=True`.
- **First call compiles** the numba kernels, which takes a few seconds. They
  are cached in `__pycache__` next to the installed package, or in a
  user-wide cache directory when that is read-only. `NUMBA_CACHE_DIR`
  overrides the location, which is useful on clusters with read-only
  environments.
- For reference: 20k × 500 bp × 200 motifs takes ≈ 2.8 s on 1 thread and
  ≈ 0.45 s on 12 (Apple M4 Pro), including setup.

## Whole genomes

```python
g = mp.FastaFile("hg38.fa")
counts = g.base_counts(); bg = counts / counts.sum()
sc = mp.Scanner(motifs, bg, p_cutoff=1e-5)
for chrom in ["chr1", "chr2"]:                    # one chromosome at a time bounds memory
    hits = sc.scan([g.fetch(chrom)])
    hits["chrom"] = chrom
    hits.to_csv(f"hits_{chrom}.tsv.gz", sep="\t", index=False)
```

Expect about `2 × genome_length × p_cutoff` hits per motif from background
alone, roughly 60k per motif for hg38 at 1e-5. Positions for hundreds of
motifs genome-wide do not fit in memory at once, so write them out per
chromosome or per motif batch.
