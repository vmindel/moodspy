# Score tracks (moodspy.tracks)

Scores every motif at every placement, with a 1 bp step and no threshold. Use it to plot
motif affinity along peaks as a genomic track instead of plotting discrete hits.

```python
tr = mp.motif_tracks(
    motifs, subject, genome=None,   # same inputs as mp.match_motifs
    select=None,       # labels, IDs or names, e.g. ["CTCF", "MA0599.1"]; exact, then case-insensitive
    bg="subject", strand="best",    # "best" | "both" | "+" | "-"
    pvalues=True, p_cutoff=5e-5,    # p_cutoff only fills attrs["thresholds"]; nothing is filtered
    pseudocount=0.8, motifmatchr_compat=False, n_jobs=None,
)
# columns: motif, seq, [chrom], start, end, strand, score, pvalue
# attrs: background, p_cutoff, thresholds (pd.Series indexed by motif label)
mp.write_bedgraph(tr, "out.{motif}.bedgraph", value="score" | "pvalue" | "neglog10p",
                  anchor="center" | "start", motif=None)
mp.select_motifs(motifs, ["CTCF", "KLF5"])          # list of Motif; KeyError on a miss
mp.pvalues_from_scores(pwm_4xL, scores, background)  # p-values for any scores
sc.tracks(seqs, strand="best", pvalues=True)         # Scanner method; seq is an integer index
```

## Contracts

- **One row per placement**, not per base. `start`/`end` is the motif window (0-based,
  half-open, genome coordinates for regions). A sequence of length `n` gives
  `n - L + 1` rows per motif (times 2 with `strand="both"`). Windows that touch a non-ACGT
  character have **no row**; they are not NaN or zero.
- **`pvalue <= p_cutoff` reproduces the hits** of `match_motifs` / `Scanner.scan` with the
  same motifs, background, pseudocount and compat flag, exactly. The p-value is
  `P(S >= floor(score · 2000) / 2000)` under the MOODS rounded-score null, which is the
  distribution the thresholds come from. The one exception: if a motif is so short that
  its single best word has p above the cutoff, MOODS still calls that word a hit, and its
  p-value stays above the cutoff.
- **Background.** `bg="subject"` uses the sequences passed to this call. A track of one
  peak therefore uses that peak's composition and does not line up with a scan of the
  whole peak set. Pass `bg=res.background` from that scan, or explicit frequencies.
- **Strand.** Both strands use the forward matrix's null distribution, as the thresholds
  do. Under a background that is not complement-symmetric (A != T or C != G), the
  reverse-strand p-value is therefore the motifmatchr/MOODS convention, not that strand's
  exact tail. `strand="best"` reports the better strand's p-value **uncorrected** for
  taking the best of two, so it is up to 2x anti-conservative as a per-placement p-value.
  Use `strand="both"` when the per-strand value matters.
- **Size.** Rows = bases × motifs (× 2 for `both`). With 50,000 peaks × 500 bp this is
  25 M rows per motif. Use `select=` and subset peaks; for whole-database summaries use
  `match_motifs(out="scores")`.

## bedGraph

- One value per base. A placement is drawn at `start + L // 2` (`anchor="center"`) or at
  `start`. With `strand="both"` the strands collapse to the max score / min p-value.
- Several motifs need `motif=` or `{motif}` in the path (one file per motif); otherwise
  `ValueError`.
- **Overlapping or duplicate regions raise `ValueError`**, because a base would get two
  values. Merge peaks first (`bedtools merge`, or `pyranges.merge`).
- For non-genomic input the sequence name is written as the chromosome.
- Output is sorted by chromosome name in byte (C locale) order, then start, which is the
  order `bedGraphToBigWig` expects, so a bigWig is one command away. No bigWig writer is
  bundled.
- `value="neglog10p"` puts motifs of different lengths and information content on one
  scale. Raw log2-odds scores are not comparable across motifs.

## CLI

```bash
moodspy track -m motifs.jaspar -r peaks.bed -g hg38.fa --select CTCF,KLF5 -o tracks.tsv
moodspy track -m motifs.jaspar -r peaks.bed -g hg38.fa --select CTCF --bedgraph \
    --value neglog10p -o 'tracks.{motif}.bedgraph'
```

The TSV adds a boolean `hit` column (`pvalue <= -p`). `--bedgraph` without `{motif}` in
`-o` exits with status 2.
