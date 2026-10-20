import numpy as np
import pytest

import moodspy as mp
from conftest import brute_force_hits, random_seqs


def _hit_set(scanner, df):
    return {
        (r.seq, scanner.labels.index(r.motif), r.start, r.strand, round(r.score, 6))
        for r in df.itertuples()
    }


@pytest.mark.parametrize("w", [1, 4, 7, 10])
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_scan_equals_brute_force(rng, w, n_jobs):
    motifs = [rng.integers(0, 30, size=(4, n)).astype(float) for n in (1, 3, 5, 7, 8, 12, 25)]
    seqs = random_seqs(rng, 120) + ["", "N" * 30, "acgtACGT" * 10, "A"]
    scanner = mp.Scanner(motifs, p_cutoff=0.02, w=w, n_jobs=n_jobs)
    df = scanner.scan(seqs, split_length=53)
    expected = brute_force_hits(seqs, scanner.pwms, scanner.thresholds)
    assert len(expected) > 100
    assert _hit_set(scanner, df) == expected
    assert (df["end"] - df["start"]).tolist() == [motifs[scanner.labels.index(m)].shape[1] for m in df["motif"]]


def test_summary_consistent_with_scan(rng):
    motifs = [rng.integers(0, 30, size=(4, n)).astype(float) for n in (6, 10, 14)]
    seqs = random_seqs(rng, 200, max_len=2000)
    scanner = mp.Scanner(motifs, p_cutoff=1e-3, n_jobs=4)
    df = scanner.scan(seqs)
    for split in (10**6, 97):
        counts, scores = scanner.summarize(seqs, split_length=split)
        grouped = df.groupby(["seq", "motif"], observed=True)["score"]
        n = grouped.size()
        best = grouped.max()
        assert counts.nnz == len(n)
        for (s, m), c in n.items():
            j = scanner.labels.index(m)
            assert counts[s, j] == c
            assert scores[s, j] == pytest.approx(best[(s, m)])


def test_split_long_sequence_is_seamless(rng):
    motifs = [rng.integers(0, 30, size=(4, n)).astype(float) for n in (5, 9, 20)]
    seq = random_seqs(rng, 1, min_len=20000, max_len=20000)[0]
    scanner = mp.Scanner(motifs, p_cutoff=1e-3, n_jobs=4)
    a = scanner.scan([seq], split_length=10**6)
    b = scanner.scan([seq], split_length=101)
    assert a.equals(b)


def test_explicit_thresholds(rng):
    motif = rng.integers(0, 30, size=(4, 8)).astype(float)
    sc = mp.Scanner([motif], thresholds=[1e9])
    assert sc.scan(["ACGT" * 100]).empty
    sc = mp.Scanner([motif], thresholds=[-1e9])
    assert len(sc.scan(["ACGT" * 10])) == 2 * (40 - 8 + 1)


def test_palindrome_hits_both_strands():
    # ACGT is its own reverse complement
    m = mp.Motif.from_counts(np.eye(4) * 10)
    sc = mp.Scanner([m], thresholds=[m.log_odds().max(axis=0).sum() - 1e-6])
    df = sc.scan(["TTACGTTT"])
    assert sorted(df["strand"].astype(str)) == ["+", "-"]
    assert set(df["start"]) == {2}


def test_invalid_window():
    with pytest.raises(ValueError):
        mp.Scanner([np.ones((4, 3))], w=13)
