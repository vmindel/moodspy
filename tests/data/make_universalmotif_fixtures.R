# Reference results from universalmotif for tests/test_universalmotif_parity.py.
# Requires universalmotif (>= 1.31 for the *_lite functions) and Biostrings.
# Run from this directory: Rscript make_universalmotif_fixtures.R
suppressMessages({
  library(universalmotif)
  library(Biostrings)
})

motifs <- c(read_jaspar("um_motifs.jaspar"), read_meme("um_motifs.meme"))
n <- length(motifs)

# ---------------------------------------------------------------- per-motif summaries
icm <- lapply(motifs, function(m) colSums(convert_type(m, "ICM")@motif))
trimmed <- trim_motifs(motifs, min.ic = 0.5)
trim_lens <- vapply(trimmed, function(m) ncol(m@motif), integer(1))
trim_names <- vapply(trimmed, function(m) m@name, character(1))
write.csv(data.frame(
  motif = seq_len(n) - 1,
  ic = vapply(icm, function(x) paste(sprintf("%.10f", x), collapse = ";"), ""),
  consensus = vapply(motifs, function(m) m@consensus, ""),
  average_ic = average_ic(motifs),
  trimmed_length = trim_lens[match(vapply(motifs, function(m) m@name, ""), trim_names)]
), "um_motif_summary.csv", row.names = FALSE)

# ---------------------------------------------------------------- compare_motifs
configs <- list()
add <- function(id, ...) configs[[length(configs) + 1]] <<- list(id = id, args = list(...))
for (m in c("PCC", "EUCL", "SW", "KL", "WEUCL", "ALLR", "BHAT", "HELL", "WPCC",
            "SEUCL", "MAN", "ALLR_LL")) {
  strats <- c("a.mean", "sum", "median", "fzt", "wa.mean")
  if (!m %in% c("ALLR", "ALLR_LL", "PCC")) strats <- c(strats, "g.mean", "wg.mean")
  for (s in strats) add(paste(m, s, sep = "|"), method = m, score.strat = s)
}
add("PCC|norm", method = "PCC", normalise.scores = TRUE)
add("EUCL|norm", method = "EUCL", normalise.scores = TRUE)
add("PCC|noRC", method = "PCC", tryRC = FALSE)
add("PCC|overlap0.5", method = "PCC", min.overlap = 0.5)
add("PCC|overlap99", method = "PCC", min.overlap = 99)
add("PCC|overlap2", method = "PCC", min.overlap = 2)
add("PCC|posic", method = "PCC", min.position.ic = 0.5, normalise.scores = TRUE)
add("SW|posic_wa", method = "SW", min.position.ic = 0.3, score.strat = "wa.mean")
add("PCC|meanic1", method = "PCC", min.mean.ic = 1)
add("PCC|meanic0", method = "PCC", min.mean.ic = 0)
add("EUCL|ICM", method = "EUCL", use.type = "ICM")
add("PCC|ICM_rel", method = "PCC", use.type = "ICM", relative_entropy = TRUE)
add("PCC|rel", method = "PCC", relative_entropy = TRUE, min.mean.ic = 0.5)

rows <- list()
for (cfg in configs) {
  res <- suppressWarnings(do.call(compare_motifs, c(list(motifs), cfg$args)))
  idx <- which(upper.tri(res, diag = TRUE), arr.ind = TRUE)
  rows[[length(rows) + 1]] <- data.frame(config = cfg$id, i = idx[, 1] - 1,
                                         j = idx[, 2] - 1, value = res[idx])
}
write.csv(do.call(rbind, rows), "um_compare.csv", row.names = FALSE)

# ---------------------------------------------------------------- merge_motifs
sets <- list(c(1, 3), c(3, 1), c(2, 3), c(1, 2, 3), c(5, 6, 7), c(1, 11), c(12, 13),
             c(3, 9, 10))
mrows <- list()
for (k in seq_along(sets)) {
  for (m in c("ALLR", "PCC", "SW")) {
    mm <- merge_motifs(motifs[sets[[k]]], method = m)
    mat <- mm@motif
    mrows[[length(mrows) + 1]] <- data.frame(
      set = paste(sets[[k]] - 1, collapse = ";"), method = m,
      col = seq_len(ncol(mat)) - 1, A = mat[1, ], C = mat[2, ], G = mat[3, ], T = mat[4, ])
  }
}
write.csv(do.call(rbind, mrows), "um_merge.csv", row.names = FALSE)

# ---------------------------------------------------------------- merge_similar clusters
crow <- list()
for (thr in c(0.95, 0.8, 0.6)) {
  cl <- merge_similar(motifs, threshold = thr, return.clusters = TRUE)
  nm <- vapply(motifs, function(m) m@name, "")
  for (g in seq_along(cl)) {
    members <- vapply(cl[[g]], function(m) m@name, "")
    crow[[length(crow) + 1]] <- data.frame(threshold = thr, cluster = g,
                                           motif = match(members, nm) - 1)
  }
}
write.csv(do.call(rbind, crow), "um_clusters.csv", row.names = FALSE)

# ---------------------------------------------------------------- compare_motifs_lite
lrows <- list()
for (null in c("empirical", "parametric")) {
  for (out in c("score", "pvalue", "qvalue")) {
    res <- compare_motifs_lite(motifs, null = null, matrix.out = out)
    idx <- which(!is.na(res) | is.na(res), arr.ind = TRUE)
    lrows[[length(lrows) + 1]] <- data.frame(null = null, out = out, i = idx[, 1] - 1,
                                             j = idx[, 2] - 1, value = res[idx])
  }
}
res <- compare_motifs_lite(motifs, null = "parametric", matrix.out = "pvalue",
                           bkg = c(0.3, 0.2, 0.2, 0.3), min.overlap = 3)
idx <- which(!is.na(res) | is.na(res), arr.ind = TRUE)
lrows[[length(lrows) + 1]] <- data.frame(null = "parametric_bkg_mo3", out = "pvalue",
                                         i = idx[, 1] - 1, j = idx[, 2] - 1, value = res[idx])
write.csv(do.call(rbind, lrows), "um_lite.csv", row.names = FALSE)

long <- compare_motifs_lite(motifs, compare.to = 3, qvalue = 1)
long$subject.i <- long$subject.i - 1
long$target.i <- long$target.i - 1
write.csv(long, "um_lite_long.csv", row.names = FALSE)

# ---------------------------------------------------------------- get_bkg
seqs <- readDNAStringSet("seqs.fa")
b1 <- get_bkg(seqs, k = 1:3)
b2 <- get_bkg(seqs, k = 1:2, RC = TRUE, pseudocount = 1)
write.csv(rbind(data.frame(config = "plain", as.data.frame(b1)),
                data.frame(config = "rc_pc1", as.data.frame(b2))),
          "um_bkg.csv", row.names = FALSE)
