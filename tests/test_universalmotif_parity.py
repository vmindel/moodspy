"""moodspy's universalmotif port against reference output from universalmotif 1.31.

Fixtures come from tests/data/make_universalmotif_fixtures.R, run on
tests/data/um_motifs.jaspar (motifs.jaspar with every
count column topped up to nsites, as universalmotif does on reading) +
tests/data/um_motifs.meme and tests/data/seqs.fa.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import moodspy as mp

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def motifs():
    return mp.read_motifs(DATA / "um_motifs.jaspar") + mp.read_motifs(DATA / "um_motifs.meme")


def _args(config):
    method, extra = config.split("|")
    kw = {"method": method}
    strats = {"a.mean", "sum", "median", "fzt", "wa.mean", "g.mean", "wg.mean"}
    if extra in strats:
        kw["score_strat"] = extra
    else:
        kw.update({
            "norm": {"normalise_scores": True},
            "noRC": {"try_rc": False},
            "overlap0.5": {"min_overlap": 0.5},
            "overlap99": {"min_overlap": 99},
            "overlap2": {"min_overlap": 2},
            "posic": {"min_position_ic": 0.5, "normalise_scores": True},
            "posic_wa": {"min_position_ic": 0.3, "score_strat": "wa.mean"},
            "meanic1": {"min_mean_ic": 1},
            "meanic0": {"min_mean_ic": 0},
            "ICM": {"use_type": "ICM"},
            "ICM_rel": {"use_type": "ICM", "relative_entropy": True},
            "rel": {"relative_entropy": True, "min_mean_ic": 0.5},
        }[extra])
    return kw


REF = pd.read_csv(DATA / "um_compare.csv")


@pytest.mark.parametrize("config", sorted(REF["config"].unique()))
def test_compare_motifs(motifs, config):
    ref = REF[REF["config"] == config]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = mp.compare_motifs(motifs, **_args(config)).to_numpy()
    got = res[ref["i"], ref["j"]]
    np.testing.assert_allclose(got, ref["value"].to_numpy(), rtol=1e-6, atol=1e-6, equal_nan=True)


def test_compare_targets_is_a_block_of_all_vs_all(motifs):
    full = mp.compare_motifs(motifs, method="ALLR", score_strat="sum", min_mean_ic=0)
    block = mp.compare_motifs(motifs[:3], motifs[3:8], method="ALLR", score_strat="sum", min_mean_ic=0)
    np.testing.assert_allclose(block.to_numpy(), full.to_numpy()[:3, 3:8])


def test_motif_summaries(motifs):
    ref = pd.read_csv(DATA / "um_motif_summary.csv")
    for row in ref.itertuples():
        m = motifs[row.motif]
        np.testing.assert_allclose(mp.motif_ic(m), [float(x) for x in row.ic.split(";")], atol=1e-6)
        assert mp.iupac_consensus(m) == row.consensus
    np.testing.assert_allclose(mp.average_ic(motifs), ref["average_ic"], atol=1e-6)


def test_trim_motifs(motifs):
    ref = pd.read_csv(DATA / "um_motif_summary.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # trim each motif alone so dropped motifs stay identifiable
        lens = []
        for m in motifs:
            try:
                lens.append(mp.trim_motifs(m, min_ic=0.5)[0].length)
            except ValueError:
                lens.append(np.nan)
    np.testing.assert_array_equal(lens, ref["trimmed_length"].to_numpy())


@pytest.mark.parametrize("method", ["ALLR", "PCC", "SW"])
def test_merge_motifs(motifs, method):
    ref = pd.read_csv(DATA / "um_merge.csv")
    ref = ref[ref["method"] == method]
    for key, block in ref.groupby("set", sort=False):
        idx = [int(x) for x in str(key).split(";")]
        merged = mp.merge_motifs([motifs[i] for i in idx], method=method)
        np.testing.assert_allclose(merged.matrix.T, block[list("ACGT")].to_numpy(), atol=1e-6,
                                   err_msg=f"set {idx}")


def test_merge_similar_clusters(motifs):
    ref = pd.read_csv(DATA / "um_clusters.csv")
    labels = [m.label for m in motifs]
    for thr, block in ref.groupby("threshold"):
        clusters = mp.merge_similar(motifs, threshold=thr, return_clusters=True)
        got = [[labels.index(m.label) for m in cl] for cl in clusters]
        want = [list(b["motif"]) for _, b in block.groupby("cluster", sort=True)]
        assert got == want, f"threshold {thr}"


LITE = pd.read_csv(DATA / "um_lite.csv")


@pytest.mark.parametrize("null,out", sorted({(a, b) for a, b in zip(LITE["null"], LITE["out"])}))
def test_compare_motifs_lite_matrix(motifs, null, out):
    ref = LITE[(LITE["null"] == null) & (LITE["out"] == out)]
    kw = {"null": null.split("_")[0], "matrix_out": out}
    if null == "parametric_bkg_mo3":
        kw.update(background=[0.3, 0.2, 0.2, 0.3], min_overlap=3)
    res = mp.compare_motifs_lite(motifs, **kw).to_numpy()
    np.testing.assert_allclose(res[ref["i"], ref["j"]], ref["value"], rtol=1e-6, atol=1e-12)


def test_compare_motifs_lite_long(motifs):
    ref = pd.read_csv(DATA / "um_lite_long.csv")
    got = mp.compare_motifs_lite([motifs[2]], targets=motifs, qvalue=1)
    assert list(got["target_i"]) == list(ref["target.i"])
    for a, b in [("offset", "offset"), ("strand", "strand"), ("overlap", "overlap"),
                 ("query_consensus", "subject.consensus"), ("target_consensus", "target.consensus")]:
        assert list(got[a]) == list(ref[b]), a
    for a in ("score", "pvalue", "qvalue"):
        np.testing.assert_allclose(got[a], ref[a], rtol=1e-6, atol=1e-14)


def test_get_bkg():
    ref = pd.read_csv(DATA / "um_bkg.csv")
    seqs = mp.read_fasta(DATA / "seqs.fa")
    plain = mp.get_bkg(seqs, k=[1, 2, 3])
    rc = mp.get_bkg(seqs, k=[1, 2], rc=True, pseudocount=1)
    for got, cfg in ((plain, "plain"), (rc, "rc_pc1")):
        want = ref[ref["config"] == cfg]
        assert list(got["klet"]) == list(want["klet"])
        np.testing.assert_allclose(got["count"], want["count"])
        np.testing.assert_allclose(got["probability"], want["probability"])
