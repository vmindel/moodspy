import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import moodspy as mp
from conftest import DATA
from test_genome import write_fasta


@pytest.fixture
def motifs():
    return mp.read_motifs(DATA / "example_motifs.jaspar")


@pytest.fixture
def genome(tmp_path, rng):
    seqs = {c: "".join(rng.choice(list("ACGT"), size=5000)) for c in ("chr1", "chr2")}
    # plant a KLF5 site so there is something to find
    seqs["chr1"] = seqs["chr1"][:1000] + "GCCCCGCCCC" + seqs["chr1"][1010:]
    path = tmp_path / "genome.fa"
    write_fasta(path, seqs)
    return path, seqs


def test_matches_and_scores(motifs, rng):
    seqs = {f"s{i}": "".join(rng.choice(list("ACGT"), size=300)) for i in range(50)}
    seqs["s0"] = "GCCCCGCCCC" + seqs["s0"]
    res = mp.match_motifs(motifs, seqs)
    assert isinstance(res, mp.MotifMatches)
    assert res.shape == (50, 3)
    assert sp.issparse(res.matches) and res.matches.dtype == bool
    assert res.scores is None and res.counts is None
    assert res.matches[0, 0]
    assert list(res.motifs.index) == ["MA0599.1_KLF5", "MA0107.1_RELA", "MA0137.3_STAT1"]
    assert list(res.sequences.index[:2]) == ["s0", "s1"]

    full = mp.match_motifs(motifs, seqs, out="scores")
    assert (full.matches != res.matches).nnz == 0
    assert (full.counts.astype(bool) != full.matches).nnz == 0
    pos = mp.match_motifs(motifs, seqs, out="positions")
    assert len(pos) == full.counts.sum()
    best = pos.groupby(["seq", "motif"], observed=True)["score"].max()
    for (s, m), v in best.items():
        assert full.scores[full.sequences.index.get_loc(s), full.motifs.index.get_loc(m)] == pytest.approx(v)

    df = full.to_dataframe("counts")
    assert df.shape == (50, 3) and df.index[0] == "s0"
    ad = full.to_anndata()
    assert ad.shape == (50, 3) and set(ad.layers.keys()) - {None} == {"scores", "counts"}
    assert ad.uns["motif_matching"]["p_cutoff"] == 5e-5


def test_input_types_agree(motifs, rng, tmp_path):
    seqs = ["".join(rng.choice(list("ACGT"), size=200)) for _ in range(20)]
    ref = mp.match_motifs(motifs, seqs, out="scores", bg="even").counts.toarray()
    fa = tmp_path / "s.fa"
    write_fasta(fa, {f"r{i}": s for i, s in enumerate(seqs)})
    for subject in (fa, str(fa), {f"r{i}": s for i, s in enumerate(seqs)}, pd.Series(seqs), tuple(seqs)):
        got = mp.match_motifs(motifs, subject, out="scores", bg="even").counts.toarray()
        np.testing.assert_array_equal(got, ref)
    single = mp.match_motifs(motifs, seqs[3], out="scores", bg="even").counts.toarray()
    np.testing.assert_array_equal(single, ref[3:4])
    # motif inputs: file path, dict, single matrix
    a = mp.match_motifs(DATA / "example_motifs.jaspar", seqs, bg="even").matches
    b = mp.match_motifs({m.label: m.matrix for m in motifs}, seqs, bg="even").matches
    assert (a != b).nnz == 0
    c = mp.match_motifs(motifs[0].matrix, seqs, bg="even").matches
    assert (c != a[:, [0]]).nnz == 0


def test_genomic_regions(motifs, genome, tmp_path):
    path, seqs = genome
    regions = pd.DataFrame({"chrom": ["chr1", "chr2", "chr1"], "start": [900, 0, 4000], "end": [1200, 500, 5000]})
    res = mp.match_motifs(motifs, regions, genome=path, out="scores")
    assert list(res.sequences.index) == ["chr1:900-1200", "chr2:0-500", "chr1:4000-5000"]
    assert res.matches[0, 0]
    assert res.sequences.loc["chr1:900-1200", "start"] == 900

    pos = mp.match_motifs(motifs, regions, genome=path, out="positions", p_cutoff=1e-3)
    assert {"chrom", "start", "end", "strand", "score", "motif", "seq"} <= set(pos.columns)
    m0 = motifs[0].log_odds(res.background)
    rc = m0[::-1, ::-1]
    from moodspy._alphabet import encode

    for r in pos[pos.motif == "MA0599.1_KLF5"].itertuples():
        site = encode(seqs[r.chrom][r.start : r.end])
        mat = m0 if r.strand == "+" else rc
        assert mat[site, np.arange(10)].sum() == pytest.approx(r.score)
        reg = regions.iloc[res.sequences.index.get_loc(r.seq)]
        assert reg.start <= r.start and r.end <= reg.end and reg.chrom == r.chrom

    # the same regions given as BED, region strings, dict genome, AnnData
    bed = tmp_path / "r.bed"
    regions.to_csv(bed, sep="\t", header=False, index=False)
    names = list(res.sequences.index)
    for subject, g in ((bed, path), (names, path), (regions, seqs), (regions, mp.FastaFile(path))):
        other = mp.match_motifs(motifs, subject, genome=g, out="scores")
        np.testing.assert_array_equal(other.counts.toarray(), res.counts.toarray())

    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(X=np.zeros((4, 3)), var=pd.DataFrame(index=names))
    other = mp.match_motifs(motifs, adata, genome=path, out="scores")
    np.testing.assert_array_equal(other.counts.toarray(), res.counts.toarray())


def test_background_options(motifs, genome):
    path, seqs = genome
    regions = pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [3000]})
    r = mp.match_motifs(motifs, regions, genome=path, bg="genome")
    allseq = "".join(seqs.values())
    expected = np.array([allseq.count(b) for b in "ACGT"]) / len(allseq)
    np.testing.assert_allclose(r.background, expected)
    r = mp.match_motifs(motifs, regions, genome=path, bg=[0.3, 0.2, 0.2, 0.3])
    np.testing.assert_allclose(r.background, [0.3, 0.2, 0.2, 0.3])
    r = mp.match_motifs(motifs, ["ACGT" * 10], bg={"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25})
    np.testing.assert_allclose(r.background, 0.25)
    with pytest.warns(UserWarning):
        mp.match_motifs(motifs, ["AAAACCCC" * 10])
    with pytest.raises(ValueError):
        mp.match_motifs(motifs, ["ACGT"], bg="genome")


def test_errors(motifs, genome):
    path, _ = genome
    with pytest.raises(ValueError, match="genome"):
        mp.match_motifs(motifs, pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10]}))
    with pytest.raises(ValueError):
        mp.match_motifs(motifs, pd.DataFrame({"chrom": ["chr1"], "start": [0], "end": [10**7]}), genome=path)
    with pytest.raises(ValueError):
        mp.match_motifs(motifs, ["ACGT"], out="bogus")
    with pytest.raises(ValueError):
        mp.match_motifs(motifs, [])


def test_string_inputs_fail_loudly(motifs):
    with pytest.raises(FileNotFoundError):
        mp.match_motifs(motifs, "peaks_typo.bed")
    with pytest.raises(ValueError, match="genome"):
        mp.match_motifs(motifs, ["chr1:100-200", "chr2:5-50"])
