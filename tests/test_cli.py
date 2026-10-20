import pandas as pd

from moodspy.cli import main
from conftest import DATA


def test_cli_positions_and_matrix(tmp_path):
    out = tmp_path / "hits.tsv"
    assert main(["scan", "-m", str(DATA / "motifs.jaspar"), "-s", str(DATA / "seqs.fa"), "-o", str(out), "-j", "2"]) == 0
    hits = pd.read_csv(out, sep="\t")
    assert list(hits.columns) == ["motif", "seq", "start", "end", "strand", "score"]
    assert len(hits) > 100

    out2 = tmp_path / "counts.tsv"
    assert main(["scan", "-m", str(DATA / "motifs.jaspar"), "-s", str(DATA / "seqs.fa"),
                 "--out", "counts", "--bg", "0.25,0.25,0.25,0.25", "-o", str(out2)]) == 0
    counts = pd.read_csv(out2, sep="\t", index_col=0)
    assert counts.shape == (300, 10)


def test_cli_regions_requires_genome(tmp_path, capsys):
    bed = tmp_path / "r.bed"
    bed.write_text("chr1\t0\t10\n")
    assert main(["scan", "-m", str(DATA / "motifs.jaspar"), "-r", str(bed)]) == 2
