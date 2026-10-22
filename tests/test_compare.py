
import numpy as np
import pytest

import moodspy as mp


@pytest.fixture
def motifs():
    return mp.example_motifs() + [mp.create_motif("GGGAATTTCC", name="kB"),
                                  mp.create_motif("TTCCCGGAA", name="STATlike")]


def test_compare_is_symmetric_and_rc_invariant(motifs):
    res = mp.compare_motifs(motifs)
    np.testing.assert_allclose(res, res.T)
    rc = mp.compare_motifs([m.reverse_complement() for m in motifs])
    np.testing.assert_allclose(res, rc, atol=1e-12)
    norc = mp.compare_motifs(motifs, try_rc=False)
    assert (norc.to_numpy() <= res.to_numpy() + 1e-12).all()


def test_compare_input_validation(motifs):
    with pytest.raises(ValueError):
        mp.compare_motifs(motifs, method="XYZ")
    with pytest.raises(ValueError):
        mp.compare_motifs(motifs, method="PCC", score_strat="g.mean")
    with pytest.raises(ValueError):
        mp.compare_motifs(motifs, method="ALLR", use_type="ICM")
    with pytest.warns(UserWarning, match="low motif IC"):
        res = mp.compare_motifs(motifs + [mp.create_motif("NNNNNNNN", name="flat")])
    assert res.loc["flat"].isna().all()


def test_compare_lite_finds_planted_relatives(motifs):
    hits = mp.compare_motifs_lite(motifs[1:2], targets=motifs, qvalue=0.5)
    assert hits["target"].iloc[0] == "MA0107.1_RELA"
    assert "kB" in set(hits["target"])
    assert (hits["qvalue"].diff().dropna() >= 0).all()
    para = mp.compare_motifs_lite(motifs, null="parametric", matrix_out="pvalue")
    assert para.shape == (5, 5) and ((para >= 0) & (para <= 1)).all().all()


def test_merge_motifs_of_rc_copies_is_identity():
    m = mp.example_motifs()[2]
    merged = mp.merge_motifs([m, m.reverse_complement()], method="PCC")
    np.testing.assert_allclose(merged.matrix, mp.manipulate.ppm(m), atol=1e-12)
    assert mp.merge_motifs([m]).kind == m.kind


def test_merge_similar_groups_relatives(motifs):
    clusters = mp.merge_similar(motifs, threshold=0.9, return_clusters=True)
    labels = [[m.label for m in c] for c in clusters]
    assert ["MA0107.1_RELA", "kB"] in labels
    merged = mp.merge_similar(motifs, threshold=0.9)
    assert len(merged) == len(clusters)


def test_align_motifs(motifs):
    al = mp.align_motifs(motifs)
    assert al.table.index[0] == max(motifs, key=lambda m: mp.motif_ic(m).sum()).label
    assert (al.table["offset"] >= 0).all() and al.width >= max(m.length for m in motifs)
    t = al.table
    # RELA and its consensus sit on the same columns and strand
    assert t.loc["MA0107.1_RELA", "offset"] == t.loc["kB", "offset"]
    assert t.loc["MA0107.1_RELA", "strand"] == t.loc["kB", "strand"]
    pads = al.padded("PPM")
    assert all(p.shape == (4, al.width) for p in pads)


def test_motif_tree(motifs):
    tree = mp.motif_tree(motifs, method="PCC")
    assert tree.linkage.shape == (4, 4)
    cl = tree.clusters(n_clusters=3)
    assert cl.iloc[0] == 1 and cl["MA0107.1_RELA"] == cl["kB"]
    assert sorted(tree.leaves()) == sorted(m.label for m in motifs)
    d = mp.motif_tree(motifs).distances
    assert (np.diag(d) == 0).all()
    with pytest.raises(ValueError):
        mp.motif_tree(motifs, method="ALLR")
    with pytest.raises(ValueError):
        tree.clusters()


def test_average_ic():
    ms = mp.example_motifs()
    np.testing.assert_allclose(mp.average_ic(ms), [mp.motif_ic(m).mean() for m in ms])
    np.testing.assert_allclose(mp.average_ic(ms, "median"), [np.median(mp.motif_ic(m)) for m in ms])
