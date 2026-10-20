import re
import subprocess
import sys

import pytest

import moodspy as mp
from moodspy._skills.install import bundled_skill_dir, install_skill, main
from conftest import DATA

SKILL = bundled_skill_dir()


def test_skill_frontmatter_and_routes():
    text = (SKILL / "SKILL.md").read_text()
    front = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    assert front, "SKILL.md needs YAML frontmatter"
    fields = dict(line.split(": ", 1) for line in front.group(1).splitlines())
    assert fields["name"] == "moodspy"
    assert 100 < len(fields["description"]) <= 1024
    routed = set(re.findall(r"`((?:references|scripts)/[\w.-]+)`", text))
    assert routed, "router table is empty"
    for rel in routed:
        assert (SKILL / rel).is_file(), f"SKILL.md routes to missing {rel}"
    on_disk = {f"references/{p.name}" for p in (SKILL / "references").glob("*.md")}
    assert on_disk <= routed, f"reference files not routed from SKILL.md: {on_disk - routed}"


def test_references_only_mention_real_api():
    names = set()
    for path in [SKILL / "SKILL.md", *(SKILL / "references").glob("*.md")]:
        names |= set(re.findall(r"\bmp\.([A-Za-z_]\w*)", path.read_text()))
        for ref in re.findall(r"`(references/[\w.-]+)`", path.read_text()):
            assert (SKILL / ref).is_file(), f"{path.name} points to missing {ref}"
    missing = sorted(n for n in names if not hasattr(mp, n))
    assert not missing, f"skill mentions mp.{missing} which moodspy does not export"


def test_install(tmp_path, capsys):
    dest = install_skill(tmp_path / "skills" / "moodspy")
    assert (dest / "SKILL.md").read_text() == (SKILL / "SKILL.md").read_text()
    assert (dest / "references" / "match-motifs.md").is_file()
    assert (dest / "scripts" / "check_hits.py").is_file()
    with pytest.raises(FileExistsError):
        install_skill(dest)
    install_skill(dest, force=True)
    assert main(["--dest", str(dest)]) == 1
    assert main(["--dest", str(dest), "--force"]) == 0
    assert main(["--print-path"]) == 0
    assert capsys.readouterr().out.strip().endswith(str(SKILL))


def test_check_hits_script_passes():
    script = SKILL / "scripts" / "check_hits.py"
    out = subprocess.run(
        [sys.executable, str(script), str(DATA / "motifs.jaspar"), str(DATA / "seqs.fa"),
         "--n", "15", "--p", "1e-3", "--bg", "subject"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "0 missing, 0 extra" in out.stdout
