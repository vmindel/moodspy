# Thresholds and background (moodspy.thresholds)

```python
mp.threshold_from_p(pwm, p, background=None, precision=2000.0)        # pwm (4, L) log-odds -> float
mp.thresholds_from_p(list_of_pwms, p, background=None, n_jobs=1)      # np.ndarray
res.motifs["threshold"]                                               # what match_motifs used
```

The algorithm is MOODS' `threshold_from_p`. Scores are rounded to a grid of
1/`precision` units. The exact null distribution of the rounded score
under `background` is computed by dynamic programming, and the threshold is
the lowest grid score whose upper tail is ≤ p. A hit is `score >= threshold`.

- **Thresholds can be negative.** For very specific motifs at loose p,
  passing p requires admitting sites with a near-forbidden base. KLF5 at
  p=1e-3 gives -1.9 against a uniform background and -5.0 against a 58% AT
  background. That is mathematically correct. Use a stricter p,
  or filter on `score`, if negative-scoring hits are unwanted.
- If even the single best word has probability > p, the threshold is
  `max_score - min_delta / 2`, so only the best word passes. If p is near 1,
  almost everything passes.
- **`p_cutoff=1` disables the threshold.** It is set 1 below the lowest
  possible score, so every placement is a hit. Use it with `out="scores"` to
  get each sequence's best score over all placements and strands, e.g. for
  short windows around known sites. `counts` then equals
  `2 × (len - motif_len + 1)` for N-free sequences. Avoid
  `out="positions"` with it on long inputs, since it returns every
  placement.
- The pwm passed to `threshold_from_p` must already be log-odds against the
  same `background`. Pass `m.log_odds(bg)` with `background=bg`, not raw
  counts.
- `p_cutoff` is per position and per strand. A 500 bp peak at 5e-5 expects
  about 2 × 500 × 5e-5 = 0.05 false hits per motif, and an 800-motif scan
  expects about 40 false hits per peak.

## Background (`bg=`)

`bg` is used twice: for the log-odds scores and for the p-value thresholds.

| `bg` | meaning |
|---|---|
| `"subject"` (default, as in motifmatchr) | A/C/G/T frequencies of all sequences in **this call** |
| `"genome"` | frequencies of the whole genome FASTA (needs `genome=`; reads it all once) |
| `"even"` | 0.25 each |
| 4 floats / `{"A":..}` | explicit; must be > 0 and sum to 1 within 1% (renormalised) |

- **`"subject"` couples sequences within a call.** Scanning peaks in
  batches, or a subset of peaks, changes the background and therefore
  every threshold and score. For reproducible results across calls, pass
  explicit frequencies (`res.background` from a reference run) or
  `bg="genome"`.
- If a base is completely absent from the subject (e.g. scanning only
  `"AAAATTTT"`), all counts get +1 and a `UserWarning` is raised.
- `mp.Scanner(...)` defaults to a **uniform** background, not "subject"
  (see `references/scanner-performance.md`).

## motifmatchr_compat

motifmatchr's `convert_pwm` applies the background correction with the
wrong sign. Count matrices end up scored as `log2(16 · p · bg)` instead of
`log2(p / bg)`, which rewards bases that are common in the background.

- moodspy's default is the correct `log2(p / bg)`.
  `motifmatchr_compat=True` reproduces motifmatchr exactly: thresholds,
  scores and matches, except for the edge hits described in
  `references/from-motifmatchr.md`.
- The two modes are identical when the background is uniform (`bg="even"`).
- Thresholds are computed against the true background in both modes, so
  false-positive rates hold either way. What changes is which sites score
  highest.
