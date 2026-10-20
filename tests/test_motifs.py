import numpy as np
import pandas as pd
import pytest

import moodspy as mp
from moodspy.motifs import as_motif_list, infer_kind


def test_kind_inference():
    counts = np.array([[3, 0], [1, 4], [0, 0], [0, 0]])
    assert infer_kind(counts.astype(float)) == "counts"
    assert infer_kind(counts / 4.0) == "probability"
    assert infer_kind(np.log2((counts + 0.2) / 4.8 / 0.25)) == "log-odds"


def test_orientation_and_dataframe():
    m = np.arange(12, dtype=float).reshape(3, 4)  # (length, 4)
    assert mp.Motif(m).matrix.shape == (4, 3)
    df = pd.DataFrame(m, columns=list("ACGT"))
    np.testing.assert_array_equal(mp.Motif(df).matrix, m.T)
    df2 = pd.DataFrame(m.T, index=list("TGCA"))
    np.testing.assert_array_equal(mp.Motif(df2).matrix, m.T[::-1])
    with pytest.raises(ValueError):
        mp.Motif(np.ones((3, 5)))


def test_log_odds_matches_tfbstools_formula():
    counts = np.array([[10, 0, 5], [0, 10, 5], [0, 0, 0], [0, 0, 0]], dtype=float)
    bg = np.array([0.3, 0.2, 0.2, 0.3])
    pwm = mp.Motif.from_counts(counts).log_odds(bg)
    probs = (counts + 0.2) / (counts.sum(axis=0) + 0.8)
    np.testing.assert_allclose(pwm, np.log2(probs / bg[:, None]))
    compat = mp.Motif.from_counts(counts).log_odds(bg, motifmatchr_compat=True)
    np.testing.assert_allclose(compat, np.log2(16 * probs * bg[:, None]))
    np.testing.assert_allclose(mp.Motif.from_counts(counts).log_odds(motifmatchr_compat=True),
                               mp.Motif.from_counts(counts).log_odds())


def test_log_odds_rebackground_round_trip():
    counts = np.random.default_rng(0).integers(0, 20, size=(4, 7)).astype(float)
    bg1 = np.array([0.3, 0.2, 0.2, 0.3])
    bg2 = np.array([0.2, 0.3, 0.3, 0.2])
    m = mp.Motif.from_counts(counts)
    from_lo = mp.Motif.from_log_odds(m.log_odds(bg1), background=bg1)
    np.testing.assert_allclose(from_lo.log_odds(bg2), m.log_odds(bg2))
    np.testing.assert_allclose(from_lo.log_odds(bg2, log_base=np.e), m.log_odds(bg2) * np.log(2))
    np.testing.assert_allclose(from_lo.probabilities(), m.probabilities())


def test_probability_pseudocount_uses_nsites():
    probs = np.array([[1.0, 0], [0, 1], [0, 0], [0, 0]])
    a = mp.Motif.from_probabilities(probs, nsites=10).probabilities()
    b = mp.Motif.from_counts(probs * 10).probabilities()
    np.testing.assert_allclose(a, b)


def test_reverse_complement_and_consensus():
    m = mp.Motif.from_counts(np.array([[9, 0, 0], [0, 9, 0], [0, 0, 9], [0, 0, 0]]), name="X")
    assert m.consensus == "ACG"
    assert m.reverse_complement().consensus == "CGT"
    assert "X" in repr(m)


def test_labels():
    assert mp.Motif(np.ones((4, 2)), name="KLF5", id="MA0599.1").label == "MA0599.1_KLF5"
    assert mp.Motif(np.ones((4, 2)), id="MA1").label == "MA1"
    motifs = as_motif_list({"a": np.ones((4, 3)), "b": mp.Motif(np.ones((4, 2)), name="zzz")})
    assert [m.label for m in motifs] == ["a", "b"]


def test_background_validation():
    np.testing.assert_allclose(mp.check_background({"t": 0.3, "a": 0.3, "c": 0.2, "g": 0.2}), [0.3, 0.2, 0.2, 0.3])
    for bad in ([0.5, 0.5, 0, 0], [0.25] * 3, [0.4] * 4, {"A": 1}):
        with pytest.raises(ValueError):
            mp.check_background(bad)


def test_biopython_like_object():
    class Fake:
        alphabet = "ACGT"
        counts = {"A": [5, 0], "C": [0, 5], "G": [0, 0], "T": [0, 0]}
        matrix_id = "MA9"
        name = "FAKE"

    (m,) = as_motif_list([Fake()])
    assert m.label == "MA9_FAKE" and m.kind == "counts"
