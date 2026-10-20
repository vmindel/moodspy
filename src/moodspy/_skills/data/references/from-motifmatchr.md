# Porting from R motifmatchr / MOODS-python

## motifmatchr

| R | Python |
|---|---|
| `matchMotifs(pwms, peaks, genome = BSgenome.Hsapiens.UCSC.hg38)` | `mp.match_motifs(motifs, peaks_df, genome="hg38.fa")` |
| `p.cutoff = 5e-5, w = 7, bg = "subject"` | `p_cutoff=5e-5, w=7, bg="subject"` |
| `out = "matches" / "scores" / "positions"` | `out="matches" / "scores" / "positions"` |
| `motifMatches(res)` / `motifScores(res)` / `motifCounts(res)` | `res.matches` / `res.scores` / `res.counts` |
| `rowRanges(res)`, `colData(res)` | `res.sequences`, `res.motifs` |
| `GRangesList` (positions) | one long DataFrame, `motif` column |
| `example_motifs` | `mp.example_motifs()` |
| `getJasparMotifs()` / `JASPAR2020` | `mp.read_motifs("JASPAR..._jaspar.txt")` or pyjaspar motifs |
| `toPWM(pfm)` | `Motif.log_odds()` (same pseudocount 0.8) |

- **Coordinates**: R ranges are 1-based and closed. Subtract 1 from the R
  `start` to get moodspy's `start`. `end` is the same number in both.
- **`pwms` of `PWMatrix` type (log2 odds) from R**: use
  `Motif.from_log_odds(m, log_base=2, background=bg_used_in_R)`.
- **Column names**: motifmatchr uses `names(pwms)`. That is `MA0599.1`
  for TFBSTools `getMatrixSet`, `MA0599.1_KLF5` for `example_motifs`, and
  `ENSG..._LINE..._NAME` for chromVAR's `getJasparMotifs`. moodspy uses
  `Motif.label` (`id_name`). Pass a `{r_name: motif}` dict to get identical
  column names.

## Reproducing motifmatchr numbers

Pass `motifmatchr_compat=True`. This reproduces motifmatchr's inverted
background correction (see `references/thresholds-background.md`) and its
operation order, so thresholds and scores match to floating-point
precision.

With `bg="even"` the flag makes no difference. One real difference remains
in both modes:

- **Edge hits.** For motifs **shorter than `w`** (default 7), MOODS and
  therefore motifmatchr miss hits that end exactly at the end of a sequence
  or right before an `N`. moodspy reports them, so moodspy counts can be
  higher by those hits. Motifs ≥ 7 bp (nearly all of JASPAR) are
  unaffected.

The moodspy test suite checks this against motifmatchr 1.30 output in
`tests/test_motifmatchr_parity.py`.

## MOODS-python

| MOODS | moodspy |
|---|---|
| `MOODS.parsers.pfm_to_log_odds(file, bg, ps)` | `mp.read_motifs(file)[0].log_odds(bg, pseudocount=ps, log_base=np.e)` |
| `MOODS.tools.threshold_from_p(m, bg, p)` | `mp.threshold_from_p(m, p, bg)`: **argument order differs**; any `(4, L)` array-like |
| `MOODS.tools.bg_from_sequence_dna(seq, ps)` | `mp.match_motifs(..., bg="subject")` or count bases yourself |
| `MOODS.scan.Scanner(7); set_motifs(mats, bg, thr); scan(seq)` | `mp.Scanner(motifs, bg, thresholds=thr[:n], w=7).scan([seq])` (forward thresholds only) |
| `MOODS.scan.scan_dna(seq, mats, bg, thr, 7)` | `mp.Scanner(...).scan([seq])` |
| results `[[match(pos, score)], ...]` per matrix, rc matrices appended by the caller | one DataFrame, both strands, `strand` column |

- MOODS-python's `pfm_to_log_odds` uses natural log and MOODS' own
  pseudocount rule: `ps * bg[j]` per cell. moodspy uses TFBSTools'
  `0.25 * ps`. To compare scores exactly, compute the matrix yourself and
  build `Motif.from_log_odds(..., log_base=np.e)`.
- MOODS needs you to append reverse-complement matrices yourself. moodspy
  always scans both strands.
- Variant-aware scanning (`MOODS.scan.variants`) and high-order PWMs are
  **not** implemented in moodspy.
