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


def _reference_fai(data: bytes):
    """The original line-by-line indexer, kept as an oracle for _build_fai."""
    index, pos, size, name = {}, 0, len(data), None
    length = offset = lb = lw = 0
    irregular = False
    while pos < size:
        nl = data.find(b"\n", pos)
        nl = size if nl < 0 else nl
        line, nxt = data[pos:nl], nl + 1
        if line.startswith(b">"):
            if name is not None:
                index[name] = (length, offset, lb, lw)
            name = line[1:].split()[0].decode() if line[1:].split() else ""
            length, offset, lb, lw, irregular = 0, nxt, 0, 0, False
        elif name is not None:
            bases = len(line.rstrip(b"\r"))
            if (irregular and bases) or (lb and bases > lb):
                raise ValueError("irregular")
            if lb == 0:
                lb, lw = bases, nl + 1 - pos
            elif bases != lb:
                irregular = True
            length += bases
        pos = nxt
    if name is not None:
        index[name] = (length, offset, lb, lw)
    return index


@pytest.mark.parametrize("seed", range(200))
def test_vectorised_index_matches_reference(seed):
    rng = np.random.default_rng(seed)
    eol = b"\r\n" if seed % 5 == 0 else b"\n"
    parts = []
    for i in range(int(rng.integers(1, 6))):
        n = int(rng.integers(1, 300))
        width = int(rng.integers(1, 80))
        seq = np.array(rng.choice(list(b"ACGTNacgt"), size=n), dtype=np.uint8).tobytes()
        lines = [seq[j:j + width] for j in range(0, n, width)]
        body = eol.join(lines) + (eol if rng.random() < 0.9 else b"")
        if rng.random() < 0.2:
            body += eol  # trailing blank line
        parts.append(b">s%d desc\n" % i + body)
        if not body.endswith(b"\n"):
            parts[-1] += b"\n"
    data = b"".join(parts)
    assert _build_fai(data) == _reference_fai(data)


def test_irregular_and_empty_records():
    with pytest.raises(ValueError):
        _build_fai(b">a\nACGT\nAC\nACGT\n")
    with pytest.raises(ValueError):
        _build_fai(b">a\nAC\nACGT\n")
    assert _build_fai(b">a\n>b\nAC\n") == {"a": (0, 3, 0, 0), "b": (2, 6, 2, 3)}


def test_stale_but_consistent_index_is_reused(tmp_path, monkeypatch):
    import os
    from moodspy import genome

    path = tmp_path / "g.fa"
    write_fasta(path, {"chr1": "ACGT" * 50, "chr2": "GG" * 33})
    mp.FastaFile(path)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**10))  # FASTA now newer than .fai
    genome._INDEX_MEMO.clear()
    monkeypatch.setattr(genome, "_build_fai", lambda mm: pytest.fail("index was rebuilt"))
    assert mp.FastaFile(path).fetch("chr2", 0, 4) == "GGGG"


def test_stale_inconsistent_index_is_rebuilt(tmp_path):
    import os
    from moodspy import genome

    path = tmp_path / "g.fa"
    write_fasta(path, {"chr1": "ACGT" * 50})
    mp.FastaFile(path)
    write_fasta(path, {"chr1": "T" * 7, "chrX": "ACGT" * 50}, width=13)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**10))
    genome._INDEX_MEMO.clear()
    fa = mp.FastaFile(path)
    assert fa.lengths == {"chr1": 7, "chrX": 200}


def test_read_only_directory_uses_cache(tmp_path, monkeypatch):
    import os
    from moodspy import genome

    ro = tmp_path / "ref"
    ro.mkdir()
    path = ro / "g.fa"
    write_fasta(path, {"chr1": "ACGT" * 100})
    os.chmod(ro, 0o555)
    try:
        fa = mp.FastaFile(path)
        assert not (ro / "g.fa.fai").exists()
        assert any(p.name.endswith(".fai") for p in __import__("pathlib").Path(genome.cache_dir()).iterdir())
        genome._INDEX_MEMO.clear()
        monkeypatch.setattr(genome, "_build_fai", lambda mm: pytest.fail("index was rebuilt"))
        assert mp.FastaFile(path).index == fa.index
    finally:
        os.chmod(ro, 0o755)


def test_base_counts_cached_on_disk(tmp_path, monkeypatch):
    path = tmp_path / "g.fa"
    write_fasta(path, {"chr1": "AACGTNacgt" * 30, "chr2": "GGGG"})
    first = mp.FastaFile(path).base_counts()
    assert first.tolist() == [90, 60, 64, 60]
    monkeypatch.setattr(mp.FastaFile, "_count_bases", lambda self: pytest.fail("recounted"))
    np.testing.assert_array_equal(mp.FastaFile(path).base_counts(), first)


def test_mixed_line_endings_on_last_line():
    data = b">a\r\n" + b"ACGTACGTAC\r\n" * 3 + b"ACGTACGTAC\n>b\nAC\n"
    assert _build_fai(data) == _reference_fai(data)
