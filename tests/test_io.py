import numpy as np
import pytest

import moodspy as mp
from conftest import DATA

MEME = """MEME version 5

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.25 C 0.25 G 0.25 T 0.25

MOTIF MA0004.1 Arnt
letter-probability matrix: alength= 4 w= 3 nsites= 20 E= 0
 0.200000  0.800000  0.000000  0.000000
 0.950000  0.000000  0.050000  0.000000
 0.000000  1.000000  0.000000  0.000000
URL http://jaspar.genereg.net/matrix/MA0004.1

MOTIF second
letter-probability matrix: alength= 4 w= 2
0.25 0.25 0.25 0.25
1 0 0 0
"""

HOMER = """>CACGTG\tbHLH(CACGTG)/Homer\t7.5\t-1.0e-10\t0
0.1\t0.7\t0.1\t0.1
0.9\t0.05\t0.025\t0.025
>TTT\tother\t3.0
0.0 0.0 0.0 1.0
"""

# HOCOMOCO v14 layout: >ID, then one A C G T row per position
HOCOMOCO_PWM = """>FOXB1.H14CORE.0.SM.B
0.8225739623327950	0.2233944032444520	-1.4492775391962918	-1.4329875901921556
-6.7056837627930941	-2.9480242014292228	-6.3341348497024166	1.3723369487936381
1.3005699668287771	-6.7056837627930941	-1.1203565269982128	-6.7056837627930941
-6.7056837627930941	-6.7056837627930941	-6.7056837627930941	1.3853759878921224
"""

HOCOMOCO_PCM = """>FOXB1.H14CORE.0.SM.B
120	30	0	50
0	1	0	199
"""

TRANSFAC = """AC  M00001
XX
ID  V$MYOD_01
NA  MyoD
P0      A      C      G      T
01      1      2      2      0      S
02      2      1      2      0      R
XX
//
AC  M00002
P0      A      C      G      T
01      5      0      0      0      A
//
"""


def test_jaspar_round_trip(tmp_path):
    motifs = mp.read_motifs(DATA / "example_motifs.jaspar")
    assert [m.label for m in motifs] == ["MA0599.1_KLF5", "MA0107.1_RELA", "MA0137.3_STAT1"]
    assert motifs[0].length == 10 and motifs[0].kind == "counts"
    out = tmp_path / "x.jaspar"
    mp.write_jaspar(motifs, out)
    again = mp.read_motifs(out)
    for a, b in zip(motifs, again):
        assert a.label == b.label
        np.testing.assert_array_equal(a.matrix, b.matrix)


def test_meme(tmp_path):
    a, b = mp.parse_meme(MEME)
    assert a.label == "MA0004.1_Arnt" and a.nsites == 20 and a.kind == "probability"
    np.testing.assert_allclose(a.matrix[:, 0], [0.2, 0.8, 0, 0])
    assert b.length == 2 and b.nsites is None
    path = tmp_path / "m.meme"
    mp.write_meme([a, b], path)
    a2, b2 = mp.read_motifs(path)
    np.testing.assert_allclose(a2.matrix, a.matrix, atol=1e-6)
    assert a2.label == a.label


def test_homer():
    a, b = mp.parse_homer(HOMER)
    assert a.name == "bHLH(CACGTG)/Homer" and a.length == 2
    assert a.meta["homer_threshold"] == 7.5
    assert b.consensus == "T"


def test_hocomoco():
    (m,) = mp.parse_hocomoco(HOCOMOCO_PWM)
    assert m.id == "FOXB1.H14CORE.0.SM.B" and m.name == ""
    assert m.kind == "log-odds" and m.log_base == np.e and m.length == 4
    assert m.consensus.upper() == "ATAT"
    # natural-log scores against a uniform background
    np.testing.assert_allclose(m.log_odds(log_base=np.e), m.matrix)
    np.testing.assert_allclose(m.probabilities().sum(axis=0), 1.0)
    (c,) = mp.parse_hocomoco(HOCOMOCO_PCM)
    assert c.kind == "counts"
    np.testing.assert_array_equal(c.matrix[:, 0], [120, 30, 0, 50])
    with pytest.raises(ValueError, match="4 values"):
        mp.parse_hocomoco(">x\n1 2 3\n")


def test_transfac():
    a, b = mp.parse_transfac(TRANSFAC)
    assert a.id == "M00001" and a.name == "MyoD" and a.length == 2
    np.testing.assert_array_equal(a.matrix[:, 0], [1, 2, 2, 0])
    assert b.consensus == "A"


def test_sniffing(tmp_path):
    for text, fmt in ((MEME, "meme"), (HOMER, "homer"), (TRANSFAC, "transfac")):
        p = tmp_path / f"motifs.{fmt}.txt"
        p.write_text(text)
        assert len(mp.read_motifs(p)) == 2
    for text, ext in ((HOCOMOCO_PWM, "pwm"), (HOCOMOCO_PCM, "pcm")):
        p = tmp_path / f"FOXB1.H14CORE.0.SM.B.{ext}"
        p.write_text(text)
        (m,) = mp.read_motifs(p)
        assert m.label == "FOXB1.H14CORE.0.SM.B"
    raw = tmp_path / "MA0004.1.pfm"
    raw.write_text(">MA0004.1 Arnt\n4 19 0\n16 0 20\n0 1 0\n0 0 0\n")
    (m,) = mp.read_motifs(raw)
    assert m.label == "MA0004.1_Arnt" and m.kind == "counts" and m.length == 3
    pfm = tmp_path / "MA0001.pfm"
    pfm.write_text("1 2 3\n0 0 0\n4 4 4\n1 1 1\n")
    (m,) = mp.read_motifs(pfm)
    assert m.label == "MA0001" and m.length == 3


def test_bad_pfm(tmp_path):
    p = tmp_path / "bad.pfm"
    p.write_text("1 2\n3 4\n")
    with pytest.raises(ValueError):
        mp.read_motifs(p)
