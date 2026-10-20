"""Benchmark inputs: random informative motifs and AT-rich random sequences."""
import sys

import numpy as np

from moodspy import Motif, read_motifs, write_jaspar

n_seqs, n_motifs = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(7)
motifs = read_motifs("../tests/data/example_motifs.jaspar")
while len(motifs) < n_motifs:
    length = int(rng.integers(6, 21))
    counts = np.round(rng.dirichlet(np.full(4, 0.4), size=length).T * 100)
    motifs.append(Motif.from_counts(counts, id=f"RND{len(motifs):04d}"))
write_jaspar(motifs, "bench_motifs.jaspar")
with open("bench_seqs.fa", "w") as fh:
    for i in range(n_seqs):
        s = "".join(rng.choice(list("ACGT"), p=[0.29, 0.21, 0.21, 0.29], size=500))
        fh.write(f">s{i}\n{s}\n")
