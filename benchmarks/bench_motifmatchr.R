suppressMessages({library(motifmatchr); library(TFBSTools); library(Biostrings); library(GenomicRanges)})
pfms <- readJASPARMatrix("bench_motifs.jaspar", matrixClass = "PFM")
seqs <- readDNAStringSet("bench_seqs.fa")
for (out in c("matches", "scores")) {
  t <- system.time(res <- matchMotifs(pfms, seqs, out = out))[["elapsed"]]
  cat(sprintf("motifmatchr %-9s %6.2f s\n", out, t))
}
cat("  total hits:", sum(SummarizedExperiment::assays(res)$motifCounts), "\n")

# Positions for genomic ranges (the usual peaks + genome workflow): put the
# sequences on one synthetic chromosome and scan them as 500 bp GRanges.
ch <- as.character(seqs)
genome <- DNAStringSet(c(chrB = paste(ch, collapse = "")))
gr <- GRanges("chrB", IRanges(start = (seq_along(ch) - 1) * 500 + 1, width = 500))
t <- system.time(res <- matchMotifs(pfms, gr, genome = genome, out = "positions"))[["elapsed"]]
cat(sprintf("motifmatchr positions (GRanges) %6.2f s, %d hits\n", t, sum(lengths(res))))

# Positions for plain sequences return one IRanges per (motif, sequence), which
# costs ~1 ms each in R, so this path is only timed on a subset.
t <- system.time(matchMotifs(pfms, seqs[1:400], out = "positions"))[["elapsed"]]
cat(sprintf("motifmatchr positions (character, first 400 seqs) %6.2f s\n", t))
