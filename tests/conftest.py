from pathlib import Path

import numpy as np
import pytest

from moodspy._alphabet import encode

DATA = Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path_factory, monkeypatch):
    """Keep moodspy's index / base-count cache out of the user's home."""
    monkeypatch.setenv("MOODSPY_CACHE_DIR", str(tmp_path_factory.mktemp("moodspy_cache")))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def random_seqs(rng, n, min_len=1, max_len=400, n_frac=0.03):
    p = [(1 - n_frac) / 4] * 4 + [n_frac]
    return [
        "".join(rng.choice(list("ACGTN"), p=p, size=int(rng.integers(min_len, max_len + 1))))
        for _ in range(n)
    ]


def brute_force_hits(seqs, pwms, thresholds):
    """Reference scan: every placement on both strands, no tricks."""
    hits = set()
    for si, s in enumerate(seqs):
        codes = encode(s)
        for j, pwm in enumerate(pwms):
            length = pwm.shape[1]
            cols = np.arange(length)
            for strand, mat in (("+", pwm), ("-", pwm[::-1, ::-1])):
                for p in range(len(s) - length + 1):
                    window = codes[p : p + length]
                    if (window > 3).any():
                        continue
                    score = mat[window, cols].sum()
                    if score >= thresholds[j] - 1e-9:
                        hits.add((si, j, p, strand, round(float(score), 6)))
    return hits
