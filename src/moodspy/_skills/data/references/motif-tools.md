# Motif and sequence tools (moodspy.manipulate, .compare, .sequences, .plotting)

A port of R/Bioconductor universalmotif. Scores, p-values, merges, clusters,
IC, consensus, trimming and `get_bkg` match universalmotif 1.31 to 1e-6
(`tests/test_universalmotif_parity.py`).

## Signatures

```python
mp.motif_ic(m, pseudocount=1.0, background=None, relative_entropy=False)  # bits per position
mp.icm(m, ...)                        # (4, L) letter heights; mp.average_ic(motifs, "a.mean")
mp.iupac_consensus(m)                 # "GGGRATTTCC" (m.consensus is argmax-only "GGGAATTTCC")
mp.motif_table(motifs)                # DataFrame: name, id, kind, length, nsites, ic, mean_ic, consensus
mp.trim_motifs(motifs, min_ic=0.25, trim_from="both")   # list; fully trimmed ones dropped + warning
mp.create_motif("TGASTCA" | ["ACGT", ...] | 12, nsites=100, name=, rng=)
mp.shuffle_motifs(motifs, k=2, rng=)  # pooled column shuffle (null motifs)
mp.sample_sites(m, n=100, rng=); mp.round_motif(m, 0.05); mp.score_match(m, sites)

mp.compare_motifs(motifs, targets=None, *, method="PCC", use_type="PPM", try_rc=True,
                  min_overlap=6, min_mean_ic=0.25, min_position_ic=0, relative_entropy=False,
                  normalise_scores=False, score_strat="a.mean", pseudocount=1.0, n_jobs=None)
mp.compare_motifs_lite(motifs, targets=None, *, qvalue=0.1, min_overlap=5, rc=True,
                       null="empirical", background=None, matrix_out="score")
mp.merge_motifs(motifs, *, method="ALLR", score_strat="sum", min_overlap=6, ...)  # -> Motif
mp.merge_similar(motifs, threshold=0.95, *, method="PCC", return_clusters=False, ...)
mp.align_motifs(motifs, *, method="ALLR", ...)  # MotifAlignment: .motifs, .table, .width, .padded()
mp.motif_tree(motifs, *, method="EUCL", linkage="complete", ...)  # MotifTree
tree.linkage; tree.distances; tree.clusters(threshold= | n_clusters=); tree.leaves(); tree.plot()

mp.get_bkg(seqs, k=(1, 2, 3), *, rc=False, pseudocount=0, merge=True, meme_file=None)
mp.shuffle_sequences(seqs, k=1, method="euler", rng=None)   # "euler" | "markov" | "linear"
mp.create_sequences(n=100, length=100, freqs=None, rng=None)  # freqs: 4 probs, {klet: p}, get_bkg table
seqs2, truth = mp.implant_motifs(motifs, seqs, *, n_per_seq=1, rate=None, positions=None,
                                 strand="both", centre_bias=1, min_spacing=0, rng=None)
mp.mask_seqs(seqs, "TGACTCA", rc=False, letter="N"); mp.mask_ranges(seqs, df, letter="N")
mp.write_fasta(seqs, path); mp.revcomp(s)

from moodspy.plotting import plot_logo, view_motifs, plot_motif_tree   # needs matplotlib
plot_logo(m, ax=None, kind="ICM")     # "ICM" | "PPM" | "PWM"; returns Axes
view_motifs(motifs, kind="ICM", align=True, sort_by="none")            # returns Figure
```

Sequence functions accept str / list / dict / Series / FASTA path and
return the same container (FASTA path -> dict). Motif arguments accept
anything `match_motifs` does.

## Footguns

- **Pseudocount is 1 here, 0.8 in scanning.** These functions follow
  universalmotif: counts become `(n + 1/4) / (N + 1)`; probability
  matrices (MEME, HOMER) are used as written, without the nsites-scaled
  pseudocount that `Motif.probabilities()` applies. Pass `pseudocount=`
  for one convention throughout.
- **NaN in `compare_motifs`** means every alignment failed `min_mean_ic`.
  The mean IC is taken over the window *including overhang gap columns,
  which count as 0 bits*, so short or flat motifs with overhangs fail
  easily. Drop flat motifs with `mp.average_ic(motifs) > 0.5`, or set
  `min_mean_ic=0`. `merge_similar` and `motif_tree` raise on NaN.
- **Direction depends on the metric.** PCC, WPCC, SW, ALLR, ALLR_LL, BHAT
  are similarities; EUCL, WEUCL, SEUCL, MAN, HELL, KL are distances.
  `g.mean`/`wg.mean` are refused for PCC and ALLR (negative scores).
  `motif_tree` turns similarities into `self - score`; ALLR cannot be a
  distance.
- **`min_overlap < 1` is a fraction** of each motif's length, and a value
  above both lengths forbids overhangs.
- **`compare_motifs_lite` with `targets`**: the empirical null is built
  from all target columns, so a database of a few motifs gives a coarse
  null; use `null="parametric"` then. It returns only rows with
  `qvalue <= qvalue` (default 0.1); pass `qvalue=1` for all. `offset` is
  the query column (in the query's aligned orientation) under target
  column 0. R's `compare_motifs_lite(c(q, db), compare.to = 1)` equals
  `compare_motifs_lite([q], targets=[q] + db)`.
- **`compare_motifs(..., targets=)` gives scores only.** universalmotif's
  `db.scores` p-values are not ported; use `compare_motifs_lite`.
- **`merge_motifs` is order dependent**: motifs are folded in one by one.
  `merge_similar` sorts each cluster by decreasing IC first.
- **Shuffling works on the characters present**: `N` and lowercase are
  separate letters (soft-masked `a` is not `A`). Uppercase first if you
  want base composition. Euler keeps the first k-1 letters; `markov` only
  matches k-let counts on average; `linear` only keeps the cut k-mers.
- **`get_bkg` is case-insensitive** and skips k-lets with non-ACGT.
- **`implant_motifs` always returns `(sequences, table)`**, with 0-based
  half-open `start`/`end` and `site` as read on the forward strand. Sites
  are sampled, so many score below a strict p threshold (especially for
  motifs from few sites).
- **`mask_ranges` needs within-sequence coordinates** (`seq`, `start`,
  `end`). Positions from `match_motifs` on genomic regions are genomic;
  subtract the region start first.
- universalmotif tops up count columns that sum to less than `nsites` when
  reading files; moodspy does not, so counts from ragged JASPAR files give
  slightly different probabilities than in R.
- Random functions use NumPy generators: same `rng` seed, different
  sequences than R's `rng.seed`.
