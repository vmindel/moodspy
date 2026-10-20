import itertools

import numpy as np
import pandas as pd
import pytest

import moodspy as mp
from conftest import DATA


def _rounded(pwm, precision):
    x = precision * pwm
    return np.where(pwm > 0, np.trunc(x + 0.5), np.trunc(x - 0.5)).astype(np.int64)


@pytest.mark.parametrize("p", [1e-3, 1e-2, 0.05, 0.3])
def test_threshold_matches_exhaustive_distribution(rng, p):
    counts = rng.integers(0, 40, size=(4, 6)).astype(float)
    bg = np.array([0.3, 0.2, 0.2, 0.3])
    pwm = mp.Motif.from_counts(counts).log_odds(bg)
    thr = mp.threshold_from_p(pwm, p, bg)

    rounded = _rounded(pwm, 2000.0)
    scores, probs = [], []
    for word in itertools.product(range(4), repeat=6):
        scores.append(rounded[list(word), range(6)].sum())
        probs.append(np.prod(bg[list(word)]))
    scores, probs = np.array(scores), np.array(probs)
    grid = int(round(thr * 2000))
    # threshold is the smallest grid score whose tail mass is <= p
    assert probs[scores >= grid].sum() <= p + 1e-12
    assert probs[scores >= grid - 1].sum() > p


def test_threshold_extremes():
    pwm = mp.Motif.from_counts(np.eye(4) * 10).log_odds()
    best = pwm.max(axis=0).sum()
    second = best - (pwm.max(axis=0) - np.sort(pwm, axis=0)[-2]).min()
    # p smaller than the best word's probability -> only the best word passes
    assert second < mp.threshold_from_p(pwm, 1e-9) < best
    # p close to 1 -> everything except the single worst word
    worst = pwm.min(axis=0).sum()
    assert worst < mp.threshold_from_p(pwm, 0.999999) < worst + (second - best) * -1


def test_vectorised_equals_single(rng):
    pwms = [mp.Motif.from_counts(rng.integers(0, 30, size=(4, n))).log_odds() for n in (3, 8, 15)]
    many = mp.thresholds_from_p(pwms, 1e-4, n_jobs=3)
    single = [mp.threshold_from_p(p, 1e-4) for p in pwms]
    np.testing.assert_array_equal(many, single)


def test_thresholds_match_motifmatchr():
    ref = pd.read_csv(DATA / "mm_thresholds.csv")
    bg = ref.loc[0, ["bgA", "bgC", "bgG", "bgT"]].to_numpy(float)
    motifs = mp.read_motifs(DATA / "motifs.jaspar")
    ours = mp.thresholds_from_p([m.log_odds(bg, motifmatchr_compat=True) for m in motifs], 5e-5, bg)
    np.testing.assert_allclose(ours, ref["threshold"], rtol=0, atol=1e-9)


def test_bad_p():
    with pytest.raises(ValueError):
        mp.threshold_from_p(np.zeros((4, 3)), 0)
