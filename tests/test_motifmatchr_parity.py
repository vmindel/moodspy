"""Compare against results produced by motifmatchr 1.30 (see data/make_motifmatchr_fixtures.R).

With ``motifmatchr_compat=True`` moodspy is expected to agree exactly,
except that MOODS (and so motifmatchr) misses hits of motifs shorter than the
window ``w`` that end exactly at the end of a sequence or right before an N.
moodspy reports those.
"""

import numpy as np
import pandas as pd
import pytest

import moodspy as mp
from conftest import DATA


@pytest.fixture(scope="module")
def inputs():
    return mp.read_motifs(DATA / "motifs.jaspar"), list(mp.read_fasta(DATA / "seqs.fa").values())


def _is_moods_edge_miss(motif, seq, start, w=7):
    end = start + motif.length
    return motif.length < w and (end == len(seq) or seq[end] not in "ACGTacgt")


@pytest.mark.parametrize("bg", ["subject", "even"])
def test_scores_and_counts(inputs, bg):
    motifs, seqs = inputs
    res = mp.match_motifs(motifs, seqs, bg=bg, out="scores", n_jobs=2, motifmatchr_compat=True)
    pos = mp.match_motifs(motifs, seqs, bg=bg, out="positions", motifmatchr_compat=True)
    extra = pos[[_is_moods_edge_miss(motifs[m], seqs[s], st)
                 for m, s, st in zip(pos.motif.cat.codes, pos.seq.cat.codes, pos.start)]]
    for assay, ours in (("motifScores", res.scores), ("motifCounts", res.counts)):
        ref = pd.read_csv(DATA / f"mm_{bg}_{assay}.csv")
        theirs = np.zeros(res.shape)
        theirs[ref.seq, ref.motif] = ref.value
        ours = ours.toarray().astype(float)
        diff = np.argwhere(~np.isclose(ours, theirs, rtol=0, atol=1e-9))
        for i, j in diff:
            n_edge = ((extra.seq.cat.codes == i) & (extra.motif.cat.codes == j)).sum()
            assert n_edge > 0, f"unexplained difference at seq {i}, motif {j}"
            if assay == "motifCounts":
                assert ours[i, j] - theirs[i, j] == n_edge


def test_positions(inputs):
    motifs, seqs = inputs
    pos = mp.match_motifs(motifs, seqs, out="positions", p_cutoff=1e-3, motifmatchr_compat=True)
    ref = pd.read_csv(DATA / "mm_positions.csv")
    key = lambda m, s, st, strand, sc: (int(m), int(s), int(st), str(strand), round(float(sc), 6))  # noqa: E731
    ours = {key(*r) for r in zip(pos.motif.cat.codes, pos.seq.cat.codes, pos.start, pos.strand, pos.score)}
    theirs = {key(*r) for r in zip(ref.motif, ref.seq, ref.start, ref.strand, ref.score)}
    assert theirs <= ours
    extra = ours - theirs
    assert all(_is_moods_edge_miss(motifs[m], seqs[s], st) for m, s, st, *_ in extra)
    assert (ref.end - ref.start).equals(pd.Series([motifs[m].length for m in ref.motif]))


def test_default_differs_only_for_nonuniform_background(inputs):
    motifs, seqs = inputs
    even = [mp.match_motifs(motifs, seqs, bg="even", out="scores", motifmatchr_compat=c) for c in (False, True)]
    assert (even[0].scores != even[1].scores).nnz == 0
    subj = [mp.match_motifs(motifs, seqs, out="scores", motifmatchr_compat=c) for c in (False, True)]
    assert (subj[0].scores != subj[1].scores).nnz > 0
