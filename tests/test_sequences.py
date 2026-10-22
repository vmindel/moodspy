from collections import Counter

import numpy as np
import pandas as pd
import pytest

import moodspy as mp
from moodspy.sequences import _codes, _count_klets


def _klet_counts(s, k):
    return Counter(s[i : i + k] for i in range(len(s) - k + 1))


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_euler_shuffle_preserves_klets_exactly(rng, k):
    seqs = ["".join(rng.choice(list("ACGTN"), p=[0.3, 0.2, 0.2, 0.28, 0.02], size=n)) for n in (k, 10, 57, 400)]
    out = mp.shuffle_sequences(seqs, k=k, method="euler", rng=1)
    for s, t in zip(seqs, out):
        assert _klet_counts(s, k) == _klet_counts(t, k)
        assert t[: k - 1] == s[: k - 1]
    # it does shuffle
    assert out[-1] != seqs[-1]


def test_shuffle_methods_and_containers(rng):
    s = "".join(rng.choice(list("ACGT"), size=300))
    for method in ("markov", "linear"):
        t = mp.shuffle_sequences(s, k=2, method=method, rng=0)
        assert isinstance(t, str) and len(t) == len(s)
    lin = mp.shuffle_sequences(s, k=3, method="linear", rng=0)
    assert sorted(lin[i : i + 3] for i in range(0, 300, 3)) == sorted(s[i : i + 3] for i in range(0, 300, 3))
    d = mp.shuffle_sequences({"a": s, "b": s[:50]}, k=2, rng=0)
    assert list(d) == ["a", "b"]
    ser = mp.shuffle_sequences(pd.Series([s], index=["x"]), rng=0)
    assert isinstance(ser, pd.Series) and ser.index[0] == "x"
    assert mp.shuffle_sequences(s, k=2, rng=5) == mp.shuffle_sequences(s, k=2, rng=5)
    with pytest.raises(ValueError):
        mp.shuffle_sequences("ACG", k=4)


def test_klet_counter_skips_invalid():
    assert _count_klets(_codes("ACNGT"), 2).sum() == 2
    assert _count_klets(_codes("acgt"), 1).tolist() == [1, 1, 1, 1]


def test_get_bkg_options(tmp_path):
    seqs = {"a": "AACGTT", "b": "GGGG"}
    per = mp.get_bkg(seqs, k=1, merge=False)
    assert list(per.columns) == ["seq", "klet", "count", "probability"]
    assert per.groupby("seq")["probability"].sum().tolist() == [1.0, 1.0]
    rc = mp.get_bkg("AAAC", k=1, rc=True)
    np.testing.assert_allclose(rc["probability"], [3 / 8, 1 / 8, 1 / 8, 3 / 8])  # AAAC + GTTT
    bfile = tmp_path / "bg.txt"
    mp.get_bkg(seqs, k=[1, 2], pseudocount=1, meme_file=bfile)
    text = bfile.read_text()
    assert text.startswith("#   order 0\nA") and "#   order 1" in text
    with pytest.raises(ValueError):
        mp.get_bkg(seqs, k=[2], meme_file=bfile)


def test_create_sequences_match_background():
    seqs = mp.create_sequences(50, 400, [0.1, 0.4, 0.4, 0.1], rng=0)
    assert len(seqs) == 50 and {len(s) for s in seqs} == {400}
    p = mp.get_bkg(seqs, k=1)["probability"].to_numpy()
    np.testing.assert_allclose(p, [0.1, 0.4, 0.4, 0.1], atol=0.01)
    # dinucleotide model with no CG: none generated
    di = {a + b: (0.0 if a + b == "CG" else 1.0) for a in "ACGT" for b in "ACGT"}
    out = mp.create_sequences(20, 200, di, rng=1)
    assert not any("CG" in s for s in out)
    bk = mp.get_bkg(out, k=[1, 2])
    more = mp.create_sequences(5, 100, bk, rng=2)
    assert not any("CG" in s for s in more)
    with pytest.raises(ValueError):
        mp.create_sequences(1, 10, {"AA": 1.0})


def test_mask_seqs_and_ranges():
    assert mp.mask_seqs("TTGACTCATT", "TGACTCA") == "TNNNNNNNTT"
    assert mp.mask_seqs("AAAA", "AA") == "NNNN"  # overlapping hits
    assert mp.mask_seqs("ACCGT", "ACG", rc=True, letter="-") == "AC---"
    seqs = ["ACGTACGT", "TTTT"]
    ranges = pd.DataFrame({"seq": [0, 0, 1], "start": [1, 2, 3], "end": [3, 5, 10]})
    assert mp.mask_ranges(seqs, ranges) == ["ANNNNCGT", "TTTN"]


def test_implant_motifs_ground_truth():
    motifs = mp.example_motifs()
    seqs = mp.create_sequences(30, 120, rng=4)
    out, truth = mp.implant_motifs(motifs, seqs, n_per_seq=2, min_spacing=5, rng=5)
    assert len(truth) == 60
    for row in truth.itertuples():
        s = out[int(row.seq[3:])]
        assert s[row.start : row.end] == row.site
    hits = mp.match_motifs(motifs, out, bg="even", out="positions", p_cutoff=1e-3)
    found = truth.merge(hits, left_on=["motif", "start", "end", "strand"],
                        right_on=["motif", "start", "end", "strand"], how="inner")
    assert len(found) > 40  # sampled sites mostly score above a loose threshold
    # explicit positions and the + strand only
    with pytest.warns(UserWarning, match="out-of-bounds"):
        _, truth2 = mp.implant_motifs(motifs[0], ["A" * 30], positions=[[3, 25]], strand="+", rng=0)
    assert truth2["start"].tolist() == [3] and set(truth2["strand"]) == {"+"}
    centred = mp.implant_motifs(motifs[0], ["A" * 1000] * 200, centre_bias=12, rng=1)[1]
    assert abs(centred["start"].mean() - 495) < 30 and centred["start"].std() < 120


def test_write_fasta_round_trip(tmp_path):
    p = tmp_path / "x.fa"
    mp.write_fasta({"a": "ACGT" * 30, "b": "GG"}, p)
    assert mp.read_fasta(p) == {"a": "ACGT" * 30, "b": "GG"}
    mp.write_fasta(["AC", "GT"], p)
    assert mp.read_fasta(p) == {"seq0": "AC", "seq1": "GT"}
    assert mp.revcomp("AACGTn") == "nACGTT"
