# Reference results from motifmatchr for tests/test_motifmatchr_parity.py.
# Requires motifmatchr, TFBSTools, Biostrings. Run from this directory.
suppressMessages({
  library(motifmatchr)
  library(TFBSTools)
  library(Biostrings)
})

pfms <- readJASPARMatrix("motifs.jaspar", matrixClass = "PFM")
seqs <- readDNAStringSet("seqs.fa")
chars <- as.character(seqs)

write_assays <- function(res, prefix) {
  for (a in c("motifScores", "motifCounts")) {
    m <- as(SummarizedExperiment::assays(res)[[a]], "TsparseMatrix")
    write.csv(data.frame(seq = m@i, motif = m@j, value = m@x),
              paste0(prefix, "_", a, ".csv"), row.names = FALSE)
  }
}

for (bg in c("subject", "even")) {
  res <- matchMotifs(pfms, chars, bg = bg, out = "scores", p.cutoff = 5e-5)
  write_assays(res, paste0("mm_", bg))
}

# positions with p = 1e-3 to get many hits
pos <- matchMotifs(pfms, chars, bg = "subject", out = "positions", p.cutoff = 1e-3)
rows <- list()
for (k in seq_along(pos)) {
  irl <- pos[[k]]
  for (s in seq_along(irl)) {
    ir <- irl[[s]]
    if (length(ir) == 0) next
    rows[[length(rows) + 1]] <- data.frame(
      motif = k - 1, seq = s - 1, start = start(ir) - 1, end = end(ir),
      strand = as.character(S4Vectors::mcols(ir)$strand),
      score = S4Vectors::mcols(ir)$score)
  }
}
write.csv(do.call(rbind, rows), "mm_positions.csv", row.names = FALSE)

# thresholds against the subject background used above
bgf <- letterFrequency(seqs, c("A", "C", "G", "T"))
bgf <- colSums(bgf) / sum(bgf)
mats <- motifmatchr:::convert_pwms(do.call(PWMatrixList, lapply(pfms, toPWM)), bgf)
thr <- motifmatchr:::get_thresholds(mats, bgf, 5e-5)[seq_along(mats)]
write.csv(data.frame(motif = seq_along(mats) - 1, threshold = thr,
                     bgA = bgf[1], bgC = bgf[2], bgG = bgf[3], bgT = bgf[4]),
          "mm_thresholds.csv", row.names = FALSE)
