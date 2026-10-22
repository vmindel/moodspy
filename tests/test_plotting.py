import pytest

import moodspy as mp

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from moodspy import plotting


def test_plot_logo_kinds():
    m = mp.example_motifs()[0]
    for kind in ("ICM", "PPM", "PWM"):
        ax = plotting.plot_logo(m, kind=kind)
        assert len(ax.patches) > 10
        plt.close(ax.figure)
    with pytest.raises(ValueError):
        plotting.plot_logo(m, kind="XYZ")


def test_view_motifs_and_tree():
    ms = mp.example_motifs() + [mp.create_motif("GGGAATTTCC", name="kB")]
    for sort_by in ("none", "ic", "similarity"):
        fig = plotting.view_motifs(ms, sort_by=sort_by)
        assert len(fig.axes) == 4
        plt.close(fig)
    fig = plotting.view_motifs(ms, align=False, kind="PPM")
    plt.close(fig)
    titles = [ax.get_title(loc="left") for ax in plotting.view_motifs(ms).axes]
    assert {t.replace(" [RC]", "") for t in titles} == {m.label for m in ms}
    plt.close("all")
    fig = mp.motif_tree(ms, method="PCC").plot()
    assert len(fig.axes) == 2
    fig = plotting.plot_motif_tree(ms, logos=False)
    assert len(fig.axes) == 1
    plt.close("all")
