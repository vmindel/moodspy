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


def test_auto_window_matches_fixed(rng, monkeypatch):
    from moodspy import scanner as scanner_mod

    motifs = [rng.integers(0, 30, size=(4, n)).astype(float) for n in (3, 6, 9, 12, 16)]
    seqs = random_seqs(rng, 60, max_len=400)
    ref = mp.Scanner(motifs, p_cutoff=1e-3, w=7).scan(seqs)
    cols = ["seq", "motif", "start", "end", "strand"]
    monkeypatch.setattr(scanner_mod, "AUTO_W_TIERS", ((1000, 5), (None, 9)))
    sc = mp.Scanner(motifs, p_cutoff=1e-3)
    assert "w=auto" in repr(sc) and not sc._table_cache
    small, large = sc.scan(seqs[:1]), sc.scan(seqs)
    assert sorted(sc._table_cache) == [5, 9]
    assert large[cols].equals(ref[cols])
    assert np.allclose(large["score"], ref["score"], rtol=0, atol=1e-10)
    assert small[cols].equals(ref[ref["seq"] == 0][cols].reset_index(drop=True))
    counts, _ = sc.summarize(seqs)
    assert counts.sum() == len(ref)


def test_auto_window_respects_table_budget(rng, monkeypatch):
    from moodspy import scanner as scanner_mod

    motifs = [rng.integers(0, 30, size=(4, 12)).astype(float) for _ in range(3)]
    monkeypatch.setattr(scanner_mod, "AUTO_W_TIERS", ((None, 10),))
    sc = mp.Scanner(motifs, p_cutoff=1e-4)
    assert sc._pick_w(10**6) == 10
    monkeypatch.setattr(scanner_mod, "AUTO_TABLE_BYTES", 8 * 4**8)
    assert mp.Scanner(motifs, p_cutoff=1e-4)._pick_w(10**6) == 7


def test_table_entries_match_enumeration(rng):
    # the pruned depth-first build must list exactly the codes a full enumeration finds
    motifs = [rng.normal(size=(4, n)) * 3 for n in (2, 5, 6, 11)]
    w = 5
    sc = mp.Scanner([mp.Motif.from_log_odds(m) for m in motifs], thresholds=[0.5, 2.0, -1e9, 4.0], w=w)
    _, ptr, tab_mat, tab_score, flat, off, length, wp, _full, thr, la_off, _order, la_bound = sc._table(w)
    got = {(c, int(tab_mat[e])): tab_score[e] for c in range(4**w) for e in range(ptr[c], ptr[c + 1])}
    expected = {}
    for k in range(len(length)):
        mat = flat[off[k] : off[k + 1]].reshape(-1, 4)
        ahead = la_bound[la_off[k]] if la_off[k + 1] > la_off[k] else 0.0
        for c in range(4**w):
            score = 0.0
            for i in range(min(length[k], w)):
                score = score + mat[wp[k] + i, (c >> (2 * (w - 1 - i))) & 3]
            if score + ahead >= thr[k]:
                expected[(c, k)] = score
    assert got == expected and len(got) > 4**w


def test_motif_longer_than_255(rng):
    motif = rng.integers(1, 30, size=(4, 300)).astype(float)
    sc = mp.Scanner([motif], thresholds=[-1e9], n_jobs=1)
    seq = "".join(rng.choice(list("ACGT"), size=700))
    assert len(sc.scan([seq])) == 2 * (700 - 300 + 1)
    assert len(sc.scan([seq[:350] + "N" + seq[351:]])) == 2 * (51 + 50)
