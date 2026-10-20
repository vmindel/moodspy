# match_motifs (moodspy.match)

```python
res = mp.match_motifs(
    motifs,            # file path, Motif, list of Motif, {label: matrix}, (4, L) array, Bio.motifs / pyjaspar motifs
    subject,           # see "Subject types" below
    genome=None,       # FASTA path, mp.FastaFile, pysam.FastaFile, pyfaidx.Fasta or {chrom: str}; needed for regions
    bg="subject",      # "subject" | "genome" | "even" | 4 floats (A, C, G, T) | {"A": .., ...}
    out="matches",     # "matches" | "scores" | "positions"
    p_cutoff=5e-5, w=7, pseudocount=0.8, motifmatchr_compat=False, n_jobs=None,
)
```

## What comes back depends on `out`

- `out="matches"` returns `MotifMatches` with only `.matches` filled in.
  `.scores` and `.counts` are **`None`**, not empty. Ask for `out="scores"` if
  you need them, because `to_dataframe("counts")` raises otherwise.
- `out="scores"` returns `MotifMatches` with all three:
  - `matches`: bool CSR
  - `counts`: int32 CSR, the number of hits on both strands
  - `scores`: float64 CSR, the best hit score on either strand
- `out="positions"` returns a **pandas DataFrame**, not `MotifMatches`, with
  one row per hit. Its columns are `motif, seq, start, end, strand, score`,
  with `chrom` added after `seq` for genomic input. For genomic input,
  `start`/`end` are genome coordinates, and `seq` names the region the hit
  came from.

```python
res.matches            # (n_seqs, n_motifs) CSR
res.sequences          # DataFrame, index = sequence/region names; chrom/start/end for regions
res.motifs             # DataFrame, index = motif labels; id, name, length, threshold
res.background         # np.ndarray (4,), what was actually used
res.to_dataframe("counts")   # dense DataFrame; only for small inputs
res.to_anndata()             # obs = sequences, var = motifs, X = matches, layers scores/counts
```

- **Absent sparse entries mean no hit, not a score of 0.** At loose
  `p_cutoff` a real hit can score below 0 (see
  `references/thresholds-background.md`), so `scores.toarray()` ranks
  no-hit above that hit. Mask with `matches` before comparing scores.
- Positions columns `motif`, `seq`, `chrom` and `strand` are **pandas
  Categoricals**. Under pandas < 3, `groupby` on them defaults to
  `observed=False` and returns every category combination, including
  zero-count ones. Always pass `observed=True`, or `.astype(str)` first.
- Positions at loose `p_cutoff` over many sequences can run to tens of
  millions of rows. If you only need per-sequence summaries, use
  `out="scores"`, which aggregates inside the kernel.
- Rows keep input order. For `list` input the names are `seq0`, `seq1`, and
  so on. For a dict they are its keys. For regions they are `chrom:start-end`,
  or BED column 4 when every name there is present and unique.

## Subject types

| subject | treated as |
|---|---|
| `"ACGT..."` (a str with no `/`, `\` or `.`) | one sequence named `seq0` |
| str/Path to an existing `.fa/.fasta/.fna(.gz)` | sequences from the FASTA |
| str/Path to an existing `.bed/.narrowPeak/.broadPeak` | regions (needs `genome`) |
| list / tuple / `pd.Series` of str | sequences |
| list of `"chr1:100-200"` or `"chr1-100-200"` strings | regions (needs `genome`) |
| `{name: seq}` dict | named sequences |
| DataFrame with chrom/start/end (aliases: chr, seqnames, Chromosome/Start/End, ...) | regions |
| pyranges object | regions (via `.df`) |
| AnnData | regions from `var_names` if they parse as regions, else `obs_names` |

- A string that contains `/`, `\` or `.` but is not an existing file raises
  `FileNotFoundError`. It is never scanned as DNA.
- Regions without `genome=` raise `ValueError`. Regions running past a
  chromosome end raise `ValueError`. moodspy never clips them silently.
- Region strands are ignored, since both strands are always scanned. This
  matches motifmatchr forcing `strand = "+"`.

## Recipes

```python
# best-scoring hit per (peak, motif)
pos = mp.match_motifs(motifs, peaks, genome="hg38.fa", out="positions")
best = pos.loc[pos.groupby(["seq", "motif"], observed=True)["score"].idxmax()]

# hits as a BED file
pos[["chrom", "start", "end", "motif", "score", "strand"]].to_csv("hits.bed", sep="\t", header=False, index=False)

# fraction of peaks with each motif
frac = pd.Series(res.matches.mean(axis=0).A1, index=res.motifs.index)
```
