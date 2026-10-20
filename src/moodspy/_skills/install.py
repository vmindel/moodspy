"""Install the bundled moodspy Agent Skill for Claude Code.

This module backs the ``moodspy-install-skills`` console script. It copies
the skill that ships inside the installed package into a directory where
Claude Code discovers skills. That gives a coding agent moodspy-specific
guidance (API contracts, defaults that silently change results) in every
project.

Claude Code does not scan ``site-packages`` for skills, so the skill has to
be copied. The default destination is the personal skill root,
``~/.claude/skills/moodspy``, which makes the skill available in all of a
user's projects. ``--dest .claude/skills/moodspy`` installs it into a single
project instead. Copying is opt-in and never happens on import or
``pip install``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

SKILL_NAME = "moodspy"


def bundled_skill_dir() -> Path:
    """Directory of the skill shipped with this package (contains ``SKILL.md``)."""
    return Path(__file__).resolve().parent / "data"


def default_dest() -> Path:
    """``~/.claude/skills/moodspy``."""
    return Path.home() / ".claude" / "skills" / SKILL_NAME


def install_skill(dest: Path | None = None, force: bool = False) -> Path:
    """Copy the bundled skill to ``dest`` (default `default_dest`).

    An existing ``dest`` is only replaced when ``force`` is set, so that an
    upgrade reflects the installed moodspy version exactly.
    """
    src = bundled_skill_dir()
    if not (src / "SKILL.md").is_file():
        raise FileNotFoundError(f"bundled skill not found at {src}; the package may be installed without its data")
    dest = default_dest() if dest is None else Path(dest).expanduser()
    if dest.exists():
        if not force:
            raise FileExistsError(f"{dest} already exists; re-run with --force to overwrite")
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints", ".DS_Store"))
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="moodspy-install-skills",
        description="Install the moodspy Agent Skill for Claude Code.",
    )
    parser.add_argument("--dest", type=Path, default=None,
                        help="destination directory (default: ~/.claude/skills/moodspy)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing installation")
    parser.add_argument("--print-path", action="store_true",
                        help="print the bundled skill directory and exit without installing")
    args = parser.parse_args(argv)

    if args.print_path:
        print(bundled_skill_dir())
        return 0
    try:
        dest = install_skill(args.dest, args.force)
    except (FileExistsError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"Installed moodspy skill to {dest}")
    print("It will be available to Claude Code in your next session.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
