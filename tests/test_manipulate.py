import warnings

import numpy as np
import pytest

import moodspy as mp
from moodspy.manipulate import ppm


def test_ppm_conventions():
    counts = mp.Motif.from_counts([[3, 0], [1, 4], [0, 0], [0, 0]])
    np.testing.assert_allclose(ppm(counts)[:, 0], [3.25 / 5, 1.25 / 5, 0.25 / 5, 0.25 / 5])
    np.testing.assert_allclose(ppm(counts, 0.8), counts.probabilities(0.8))
    probs = mp.Motif.from_probabilities([[0.5, 0], [0.5, 1], [0, 0], [0, 0]], nsites=4)
    np.testing.assert_array_equal(ppm(probs), probs.matrix)  # no pseudocount for probabilities


def test_ic_and_icm():
    m = mp.create_motif("ANR")
    ic = mp.motif_ic(m)
    assert ic[0] > 1.9 and abs(ic[1]) < 1e-12 and 0.9 < ic[2] < 1.0
    np.testing.assert_allclose(mp.icm(m).sum(axis=0), ic)
    # relative entropy against the true background of a column is zero
    col = mp.Motif.from_probabilities(np.array([[0.4], [0.1], [0.1], [0.4]]))
    assert mp.motif_ic(col, background=[0.4, 0.1, 0.1, 0.4], relative_entropy=True)[0] == pytest.approx(0)


def test_iupac_consensus_and_create_motif_round_trip():
    for word in ("TGASTCA", "GGGRNTTYCC", "ACGTMKWSHBVD"):
        assert mp.iupac_consensus(mp.create_motif(word)) == word
    sites = mp.create_motif(["ACGT", "ACGA", "TCGA"])
    assert sites.kind == "counts" and sites.matrix.sum(axis=0).tolist() == [3, 3, 3, 3]
    with pytest.raises(ValueError):
        mp.create_motif(["ACG", "AC"])
    with pytest.raises(ValueError):
        mp.create_motif("ACGX")
    rnd = mp.create_motif(12, rng=1)
    assert rnd.length == 12 and rnd.kind == "probability"
    np.testing.assert_allclose(rnd.matrix.sum(axis=0), 1)


def test_trim_motifs():
    m = mp.create_motif("NNACGTNN", name="x")
    assert mp.trim_motifs(m)[0].length == 4
    assert mp.trim_motifs(m, trim_from="left")[0].length == 6
    assert mp.trim_motifs(m, trim_from="right")[0].length == 6
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = mp.trim_motifs([m, mp.create_motif("NNN", name="empty")])
    assert len(out) == 1 and "empty" in str(w[0].message)
    with pytest.raises(ValueError):
        mp.trim_motifs(mp.create_motif("NN"))


def test_shuffle_motifs_keeps_columns():
    ms = mp.example_motifs()
    for k in (1, 2, 3):
        sh = mp.shuffle_motifs(ms, k=k, rng=0)
        assert [m.length for m in sh] == [m.length for m in ms]
        before = np.sort(np.concatenate([ppm(m) for m in ms], axis=1), axis=1)
        after = np.sort(np.concatenate([ppm(m) for m in sh], axis=1), axis=1)
        np.testing.assert_allclose(before, after)


def test_sample_sites_follow_the_motif():
    m = mp.create_motif("ACGTN")
    sites = mp.sample_sites(m, 4000, rng=3)
    assert all(len(s) == 5 for s in sites)
    first = np.mean([s[0] == "A" for s in sites])
    assert first > 0.98
    last = np.array([[s[4] == b for b in "ACGT"] for s in sites]).mean(axis=0)
    np.testing.assert_allclose(last, 0.25, atol=0.03)
    back = mp.create_motif(sites)
    np.testing.assert_allclose(ppm(back, 0), ppm(m), atol=0.03)


def test_round_motif_and_score_match():
    m = mp.Motif.from_probabilities([[0.96, 0.5], [0.02, 0.5], [0.01, 0], [0.01, 0]])
    np.testing.assert_allclose(mp.round_motif(m).matrix, [[1, 0.5], [0, 0.5], [0, 0], [0, 0]])
    klf = mp.example_motifs()[0]
    site = "GCCCCGCCCC"
    hits = mp.match_motifs(klf, site, bg="even", out="positions", p_cutoff=1e-2)
    best = hits[hits["strand"] == "+"]["score"].max()
    assert mp.score_match(klf, site)[0] == pytest.approx(best)
    assert mp.score_match(klf, "GCCCCNCCCC")[0] == -np.inf


def test_motif_table():
    t = mp.motif_table(mp.example_motifs())
    assert list(t.index) == ["MA0599.1_KLF5", "MA0107.1_RELA", "MA0137.3_STAT1"]
    assert {"ic", "mean_ic", "consensus", "nsites"} <= set(t.columns)
    assert t.loc["MA0107.1_RELA", "consensus"] == "GGGRATTTCC"
