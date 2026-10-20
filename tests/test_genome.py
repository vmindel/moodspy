import numpy as np
import pandas as pd
import pytest

import moodspy as mp
from moodspy.genome import _build_fai


def write_fasta(path, seqs, width=60):
    with open(path, "w") as fh:
        for name, s in seqs.items():
            fh.write(f">{name} some description\n")
            for i in range(0, len(s), width):
                fh.write(s[i : i + width] + "\n")


@pytest.mark.parametrize("width", [1, 7, 60, 10_000])
def test_fetch_matches_python_slicing(tmp_path, rng, width):
    seqs = {f"chr{i}": "".join(rng.choice(list("ACGTN"), size=int(rng.integers(1, 500)))) for i in range(5)}
    path = tmp_path / "g.fa"
    write_fasta(path, seqs, width)
    fa = mp.FastaFile(path)
    assert fa.lengths == {k: len(v) for k, v in seqs.items()}
    for name, s in seqs.items():
        for _ in range(20):
            a = int(rng.integers(0, len(s) + 1))
            b = int(rng.integers(a, len(s) + 1))
            assert fa.fetch(name, a, b) == s[a:b]
    assert (tmp_path / "g.fa.fai").exists()
    # second open reuses the index
    assert mp.FastaFile(path).index == fa.index
    counts = fa.base_counts()
    allseq = "".join(seqs.values())
    assert counts.tolist() == [allseq.count(b) for b in "ACGT"]


def test_fetch_errors(tmp_path):
    write_fasta(tmp_path / "g.fa", {"chr1": "ACGT" * 10})
    fa = mp.FastaFile(tmp_path / "g.fa")
    with pytest.raises(KeyError):
        fa.fetch("chrX", 0, 1)
    with pytest.raises(ValueError):
        fa.fetch("chr1", 0, 41)


def test_irregular_lines_rejected(tmp_path):
    (tmp_path / "bad.fa").write_text(">a\nACG\nA\nACG\n")
    with pytest.raises(ValueError):
        mp.FastaFile(tmp_path / "bad.fa")


def test_crlf(tmp_path):
    (tmp_path / "w.fa").write_bytes(b">a\r\nACGT\r\nAC\r\n>b\r\nGG\r\n")
    fa = mp.FastaFile(tmp_path / "w.fa", write_index=False)
    assert fa.fetch("a") == "ACGTAC" and fa.fetch("b") == "GG"


def test_regions_frame_variants(tmp_path):
    df = mp.regions_frame(pd.DataFrame({"Chromosome": ["chr1"], "Start": [5], "End": [9]}))
    assert df.index.tolist() == ["chr1:5-9"] and df.start.tolist() == [5]
    df = mp.regions_frame(["chr1:1,000-2,000", "chr2-5-10", "chrUn_gl000220-1-3"])
    assert df.chrom.tolist() == ["chr1", "chr2", "chrUn_gl000220"]
    assert df.end.tolist() == [2000, 10, 3]
    bed = tmp_path / "x.bed"
    bed.write_text("track name=x\nchr1\t1\t5\tpeakA\t0\t+\nchr1\t7\t9\tpeakB\n")
    df = mp.read_bed(bed)
    assert df.index.tolist() == ["peakA", "peakB"]
    with pytest.raises(ValueError):
        mp.regions_frame(pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError):
        mp.regions_frame(["nonsense"])


def test_build_fai_offsets():
    import mmap
    import tempfile

    with tempfile.TemporaryFile() as fh:
        fh.write(b">x\nAC\nG\n>y desc\nTTTT\n")
        fh.flush()
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        idx = _build_fai(mm)
    assert idx == {"x": (3, 3, 2, 3), "y": (4, 16, 4, 5)}
    assert np.isscalar(idx["x"][0])
