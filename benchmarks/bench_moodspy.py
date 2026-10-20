import time

import moodspy as mp

motifs = mp.read_motifs("bench_motifs.jaspar")
seqs = mp.read_fasta("bench_seqs.fa")
mp.match_motifs(motifs[:2], list(seqs.values())[:5])  # warm up numba cache
for compat in (True, False):
    for n_jobs in (1, None):
        for out in ("matches", "scores", "positions"):
            t = time.perf_counter()
            res = mp.match_motifs(motifs, seqs, out=out, n_jobs=n_jobs, motifmatchr_compat=compat)
            label = f"moodspy {out:<9} jobs={n_jobs or 'all'} compat={compat}"
            print(f"{label:<45} {time.perf_counter() - t:6.2f} s")
            if out == "scores":
                print("  total hits:", res.counts.sum())
