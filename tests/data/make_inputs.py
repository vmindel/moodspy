"""Generate the motif and sequence inputs used by the motifmatchr parity tests.

Run from this directory, then run make_motifmatchr_fixtures.R:

    python make_inputs.py && Rscript make_motifmatchr_fixtures.R
"""

import numpy as np

from moodspy import Motif, read_motifs, write_jaspar

rng = np.random.default_rng(2024)

motifs = read_motifs("example_motifs.jaspar")
for i, length in enumerate([4, 6, 7, 9, 15, 21, 30]):
    probs = rng.dirichlet(np.full(4, 0.4), size=length).T
    counts = np.round(probs * rng.integers(10, 500)).astype(int)
    motifs.append(Motif.from_counts(counts, name=f"RAND{i}", id=f"RND{i:03d}.1"))
write_jaspar(motifs, "motifs.jaspar")

# AT-rich sequences with planted consensus sites, N runs and soft-masking.
bg = [0.29, 0.21, 0.21, 0.29]
seqs = []
for i in range(300):
    n = int(rng.integers(50, 800))
    s = list(rng.choice(list("ACGT"), p=bg, size=n))
    for _ in range(int(rng.integers(0, 4))):
        m = motifs[int(rng.integers(len(motifs)))]
        site = m.consensus if rng.random() < 0.5 else m.reverse_complement().consensus
        if len(site) < n:
            p = int(rng.integers(0, n - len(site)))
            s[p : p + len(site)] = list(site)
    if rng.random() < 0.2:
        p = int(rng.integers(0, n))
        s[p : p + int(rng.integers(1, 30))] = "N"
    seq = "".join(s)[:n]
    if rng.random() < 0.2:
        a = int(rng.integers(0, n))
        seq = seq[:a] + seq[a : a + 100].lower() + seq[a + 100 :]
    seqs.append(seq)

with open("seqs.fa", "w") as fh:
    for i, s in enumerate(seqs):
        fh.write(f">seq{i}\n")
        for j in range(0, len(s), 60):
            fh.write(s[j : j + 60] + "\n")
