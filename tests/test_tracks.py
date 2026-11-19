import itertools

import numpy as np
import pandas as pd
import pytest

import moodspy as mp
from moodspy.cli import main
from moodspy.thresholds import _upper_tail
from conftest import DATA, random_seqs


def _key(df):
    return list(zip(df["seq"], df["motif"].astype(str), df["strand"].astype(str), df["start"]))


@pytest.mark.parametrize("n_jobs", [1, 4])
def test_scores_equal_scan_with_no_threshold(rng, n_jobs):
    motifs = [rng.integers(0, 30, size=(4, n)).astype(float) for n in (1, 5, 8, 15)]
    seqs = random_seqs(rng, 80) + ["", "N" * 30, "acgtACGT" * 10, "A"]
    sc = mp.Scanner(motifs, [0.3, 0.2, 0.2, 0.3], p_cutoff=1.0, n_jobs=n_jobs)
    hits = sc.scan(seqs)
    tr = sc.tracks(seqs, strand="both", pvalues=False)
    assert "pvalue" not in tr
    assert _key(tr) == _key(hits)
    np.testing.assert_allclose(tr["score"], hits["score"], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(tr["end"], hits["end"])


def test_plus_strand_equals_score_match(rng):
    m = mp.Motif(rng.integers(0, 30, size=(4, 9)).astype(float), "m")
    seq = "".join(rng.choice(list("ACGT"), size=200))
    tr = mp.Scanner([m], p_cutoff=1.0).tracks([seq], strand="+")
    sites = [seq[s:s + 9] for s in tr["start"]]
    np.testing.assert_allclose(tr["score"], mp.score_match(m, sites), atol=1e-10)
    assert len(tr) == 200 - 9 + 1


@pytest.mark.parametrize("bg", [None, [0.3, 0.2, 0.2, 0.3], [0.1, 0.4, 0.4, 0.1]])
@pytest.mark.parametrize("p", [0.2, 1e-2, 1e-3, 5e-5])
def test_pvalue_cutoff_reproduces_hits(bg, p):
    motifs = mp.read_motifs(DATA / "motifs.jaspar")
    seqs = list(mp.read_fasta(DATA / "seqs.fa").values())
    sc = mp.Scanner(motifs, bg, p)
    # MOODS calls the best word of a motif a hit even when its p-value is above p
    ok = [lab for k, lab in enumerate(sc.labels) if _upper_tail(sc._mats[k], sc.background, sc.precision)[0][-1] <= p]
    tr = sc.tracks(seqs, strand="both")
    hits = sc.scan(seqs)
    tr, hits = tr[tr["motif"].isin(ok)], hits[hits["motif"].isin(ok)]
    assert len(hits) > 50
    assert set(_key(tr[tr["pvalue"] <= p])) == set(_key(hits))


def test_pvalues_exact_on_grid_matrix(rng):
    # entries on the 1/2000 grid make the rounded null distribution exact
    pwm = rng.integers(-6000, 4000, size=(4, 5)) / 2000.0
    bg = np.array([0.3, 0.2, 0.15, 0.35])
    words = np.array(list(itertools.product(range(4), repeat=5)))
    scores = pwm[words, np.arange(5)].sum(axis=1)
    probs = bg[words].prod(axis=1)
    order = np.argsort(-scores, kind="stable")
    expected = np.empty_like(scores)
    for s in np.unique(scores):
        expected[scores == s] = probs[scores >= s - 1e-12].sum()
    got = mp.pvalues_from_scores(pwm, scores, bg)
    np.testing.assert_allclose(got[order], expected[order], rtol=1e-9)
    assert np.isnan(mp.pvalues_from_scores(pwm, [np.nan]))[0]


@pytest.mark.parametrize("p", [0.3, 1e-2, 1e-4])
def test_pvalue_of_threshold(rng, p):
    pwm = mp.Motif(rng.integers(0, 30, size=(4, 12)).astype(float)).log_odds()
    thr = mp.threshold_from_p(pwm, p)
    assert mp.pvalues_from_scores(pwm, [thr])[0] <= p
    assert mp.pvalues_from_scores(pwm, [thr - 1e-3])[0] > p


def test_best_strand(rng):
    motifs = [rng.integers(0, 30, size=(4, 7)).astype(float)]
    seqs = random_seqs(rng, 20, min_len=5)
    sc = mp.Scanner(motifs)
    both = sc.tracks(seqs, strand="both")
    best = sc.tracks(seqs)
    fwd = both[both["strand"] == "+"].reset_index(drop=True)
    rev = both[both["strand"] == "-"].reset_index(drop=True)
    np.testing.assert_array_equal(best["start"], fwd["start"])
    np.testing.assert_array_equal(best["score"], np.maximum(fwd["score"], rev["score"]))
    np.testing.assert_array_equal(best["pvalue"], np.minimum(fwd["pvalue"], rev["pvalue"]))
    assert (best["strand"].astype(str) == np.where(rev["score"] > fwd["score"], "-", "+")).all()


def test_edges_and_non_acgt():
    m = np.eye(4)[:, [0, 1, 2]] * 2  # length 3
    tr = mp.Scanner([m]).tracks(["ACGTNACGT", "AC", ""])
    # placements touching the N (positions 2..4) and the short sequences have no rows
    assert tr["start"].tolist() == [0, 1, 5, 6]
    assert (tr["seq"] == 0).all()


def test_motif_tracks_regions_and_select():
    motifs = mp.read_motifs(DATA / "motifs.jaspar")
    fa = mp.read_fasta(DATA / "seqs.fa")
    chrom = "".join(list(fa.values())[:10])
    genome = {"chr2": chrom}
    peaks = pd.DataFrame({"chrom": ["chr2", "chr2"], "start": [100, 1500], "end": [400, 1700]})
    tr = mp.motif_tracks(motifs, peaks, genome=genome, select=["klf5", "MA0107.1"], bg="even")
    assert list(tr.columns) == ["motif", "seq", "chrom", "start", "end", "strand", "score", "pvalue"]
    assert list(tr["motif"].cat.categories) == ["MA0599.1_KLF5", "MA0107.1_RELA"]
    assert list(tr.attrs["thresholds"].index) == ["MA0599.1_KLF5", "MA0107.1_RELA"]
    plain = mp.motif_tracks(motifs, chrom[1500:1700], select="RELA", bg="even")
    reg = tr[(tr["seq"] == "chr2:1500-1700") & (tr["motif"] == "MA0107.1_RELA")]
    np.testing.assert_array_equal(reg["start"].to_numpy() - 1500, plain["start"])
    np.testing.assert_array_equal(reg["score"], plain["score"])
    assert len(reg) == 200 - motifs[1].length + 1
    with pytest.raises(KeyError, match="NOPE"):
        mp.select_motifs(motifs, ["KLF5", "NOPE"])


def test_write_bedgraph(tmp_path):
    motifs = mp.read_motifs(DATA / "motifs.jaspar")
    genome = {"chr1": "".join(list(mp.read_fasta(DATA / "seqs.fa").values())[:10])}
    peaks = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [1000, 100], "end": [1200, 300]})
    tr = mp.motif_tracks(motifs, peaks, genome=genome, select=["KLF5", "RELA"], strand="both")
    with pytest.raises(ValueError, match="motif"):
        mp.write_bedgraph(tr, tmp_path / "x.bg")
    files = mp.write_bedgraph(tr, tmp_path / "t.{motif}.bg", value="neglog10p")
    assert len(files) == 2
    bg = pd.read_csv(files[0], sep="\t", header=None, names=["chrom", "start", "end", "v"])
    klf = tr[tr["motif"] == "MA0599.1_KLF5"]
    best_p = klf.groupby("start")["pvalue"].min()
    assert len(bg) == len(best_p) > 300  # windows over N in seqs.fa have no value
    assert bg["start"].is_monotonic_increasing and (bg["end"] - bg["start"] == 1).all()
    np.testing.assert_allclose(bg["v"], -np.log10(best_p.sort_index().to_numpy()), rtol=1e-5)
    assert bg["start"].iloc[0] == 100 + 10 // 2
    mp.write_bedgraph(tr, tmp_path / "s.bg", motif="MA0599.1_KLF5", anchor="start")
    assert pd.read_csv(tmp_path / "s.bg", sep="\t", header=None)[1].iloc[0] == 100

    overlapping = pd.DataFrame({"chrom": ["chr1", "chr1"], "start": [100, 250], "end": [300, 400]})
    tr2 = mp.motif_tracks(motifs, overlapping, genome=genome, select="KLF5")
    with pytest.raises(ValueError, match="overlapping"):
        mp.write_bedgraph(tr2, tmp_path / "o.bg")


def test_cli_track(tmp_path):
    out = tmp_path / "tracks.tsv"
    args = ["track", "-m", str(DATA / "motifs.jaspar"), "-s", str(DATA / "seqs.fa"), "-j", "2"]
    assert main(args + ["--select", "KLF5,RELA", "-o", str(out)]) == 0
    t = pd.read_csv(out, sep="\t")
    assert list(t.columns) == ["motif", "seq", "start", "end", "strand", "score", "pvalue", "hit"]
    assert set(t["motif"]) == {"MA0599.1_KLF5", "MA0107.1_RELA"}
    assert main(["scan", "-m", str(DATA / "motifs.jaspar"), "-s", str(DATA / "seqs.fa"),
                 "-o", str(tmp_path / "hits.tsv")]) == 0
    hits = pd.read_csv(tmp_path / "hits.tsv", sep="\t")
    hits = hits[hits["motif"].isin(["MA0599.1_KLF5", "MA0107.1_RELA"])]
    called = t[t["hit"]]
    # strand="best" keeps one strand per placement, so compare placements
    assert set(zip(called["seq"], called["motif"], called["start"])) == set(zip(hits["seq"], hits["motif"], hits["start"]))
    assert main(args + ["--select", "KLF5", "--bedgraph", "-o", str(tmp_path / "x.bg")]) == 2
    assert main(args + ["--select", "KLF5", "--bedgraph", "-o", str(tmp_path / "{motif}.bg")]) == 0
    assert (tmp_path / "MA0599.1_KLF5.bg").is_file()
    assert main(args + ["--select", "NOPE"]) == 2


def test_threaded_tracks_equal_single_thread(rng):
    motifs = [rng.integers(0, 30, size=(4, n)).astype(float) for n in (4, 9, 21)]
    seqs = random_seqs(rng, 400, min_len=200, max_len=800)  # > 100 kb, so the dense kernel is chunked
    assert sum(map(len, seqs)) > 150_000
    for strand in ("best", "both", "-"):
        one = mp.Scanner(motifs, [0.3, 0.2, 0.2, 0.3], n_jobs=1).tracks(seqs, strand=strand)
        many = mp.Scanner(motifs, [0.3, 0.2, 0.2, 0.3], n_jobs=4).tracks(seqs, strand=strand)
        pd.testing.assert_frame_equal(one, many)
