suppressMessages({library(motifmatchr); library(TFBSTools); library(Biostrings)})
pfms <- readJASPARMatrix("bench_motifs.jaspar", matrixClass = "PFM")
seqs <- readDNAStringSet("bench_seqs.fa")
for (out in c("matches", "scores")) {
  t <- system.time(res <- matchMotifs(pfms, seqs, out = out))[["elapsed"]]
  cat(sprintf("motifmatchr %-9s %6.2f s\n", out, t))
}
cat("  total hits:", sum(SummarizedExperiment::assays(res)$motifCounts), "\n")
# positions output for plain sequences builds one IRanges per (motif, sequence),
# which is very slow, so time it on a subset
t <- system.time(res <- matchMotifs(pfms, seqs[1:2000], out = "positions"))[["elapsed"]]
cat(sprintf("motifmatchr positions (first 2000 seqs) %6.2f s\n", t))
