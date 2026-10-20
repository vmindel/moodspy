import numpy as np
import pandas as pd
import pytest

import moodspy as mp


def reference(hits, strands="opposite", min_overlap=1):
    """Plain-Python greedy suppression with the same tie-breaking."""
    order = sorted(range(len(hits)), key=lambda i: (-hits.score.iat[i], hits.start.iat[i], hits.strand.iat[i] == "-"))
    kept = []
    for i in order:
        row = hits.iloc[i]
        clash = False
        for j in kept:
            other = hits.iloc[j]
            if (other.seq, other.motif) != (row.seq, row.motif):
                continue
            if strands == "opposite" and other.strand == row.strand:
                continue
            if min(row.end, other.end) - max(row.start, other.start) >= min_overlap:
                clash = True
                break
        if not clash:
            kept.append(i)
    return sorted(kept)


@pytest.mark.parametrize("strands", ["opposite", "any"])
@pytest.mark.parametrize("min_overlap", [1, 4])
@pytest.mark.parametrize("seed", range(8))
def test_matches_reference(seed, strands, min_overlap):
    rng = np.random.default_rng(seed)
    n = 120
    motif = rng.choice(["m1", "m2"], n)
    length = np.where(motif == "m1", 8, 13)
    start = rng.integers(0, 400, n)
    hits = pd.DataFrame({
        "motif": pd.Categorical(motif), "seq": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        "start": start, "end": start + length,
        "strand": pd.Categorical(rng.choice(["+", "-"], n), categories=["+", "-"]),
        "score": rng.normal(10, 2, n).round(1),   # rounding creates ties
    }, index=rng.permutation(n) + 1000)
    out = mp.remove_overlaps(hits, strands=strands, min_overlap=min_overlap)
    expected = hits.iloc[reference(hits, strands, min_overlap)]
    assert out.index.tolist() == expected.index.tolist()


def test_palindrome_pairs_collapse():
    ap1 = mp.Motif.from_counts(np.array([  # ATGASTCATN: AP-1 / BATF-like, palindromic core
        [20, 0, 0, 10, 0, 0, 0, 20, 0, 5],
        [0, 0, 0, 0, 10, 0, 20, 0, 0, 5],
        [0, 0, 20, 10, 10, 0, 0, 0, 0, 5],
        [0, 20, 0, 0, 0, 20, 0, 0, 20, 5]]), name="AP1")
    seq = "CCCCCATGACTCATCCCCCCCCCCCCCATGACTCATACCCCC"
    hits = mp.match_motifs([ap1], [seq], bg="even", out="positions", p_cutoff=1e-3)
    assert sorted(hits.strand.astype(str)) == ["+", "+", "-", "-"]
    one = mp.remove_overlaps(hits)
    assert len(one) == 2 and set(one.start) < set(hits.start)
    assert all(one.score.to_numpy() == [hits[(hits.end > s) & (hits.start < e)].score.max()
                                          for s, e in zip(one.start, one.end)])


def test_same_strand_overlaps_only_with_any():
    hits = pd.DataFrame({"motif": ["g"] * 3, "seq": ["s"] * 3, "start": [440, 449, 500],
                         "end": [452, 461, 512], "strand": ["+"] * 3, "score": [9.05, 9.54, 8.0]})
    assert len(mp.remove_overlaps(hits)) == 3
    assert mp.remove_overlaps(hits, strands="any").start.tolist() == [449, 500]


def test_works_on_scanner_output_and_validates(rng):
    seqs = ["".join(rng.choice(list("ACGT"), 400)) for _ in range(50)]
    sc = mp.Scanner(mp.example_motifs(), p_cutoff=1e-2)
    hits = sc.scan(seqs)
    out = mp.remove_overlaps(hits, strands="any")
    for (_, _), g in out.groupby(["seq", "motif"], observed=True):
        g = g.sort_values("start")
        assert (g.start.to_numpy()[1:] >= g.end.to_numpy()[:-1]).all()
    assert mp.remove_overlaps(hits.iloc[:0]).empty
    with pytest.raises(ValueError):
        mp.remove_overlaps(hits, strands="both")
    with pytest.raises(ValueError):
        mp.remove_overlaps(hits.drop(columns="score"))
