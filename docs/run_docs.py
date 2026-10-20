"""Execute the code in the tutorials and check (or refresh) their printed output.

Convention in docs/tutorials/*.md:

* ```` ```python ```` blocks are executed in order, one shared namespace per
  page, inside a temporary working directory.
* A ```` ```text ```` block directly after a python block is its expected
  stdout.
* ```` ```py ````, ```` ```bash ```` and ```` ```r ```` blocks are shown but
  never executed (for things needing a real genome, R, timings...).

    python docs/run_docs.py            # check all tutorials, exit 1 on mismatch
    python docs/run_docs.py --update   # rewrite the text blocks with fresh output
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path

TUTORIALS = Path(__file__).parent / "tutorials"
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```[ \t]*\n", re.S | re.M)


def _blocks(text: str):
    """Yield (lang, body, start, end) for every fenced block."""
    for m in FENCE.finditer(text):
        yield m.group(1), m.group(2), m.start(), m.end()


def run_page(path: Path) -> list[tuple[str, str | None, str, tuple[int, int] | None]]:
    """Run a page; return (code, expected, actual, span of expected block) per python block."""
    import numpy as np
    import pandas as pd

    pd.set_option("display.width", 100)
    pd.set_option("display.max_columns", 20)
    np.set_printoptions(precision=4, suppress=True, linewidth=100)

    text = path.read_text()
    blocks = list(_blocks(text))
    results = []
    namespace: dict = {"__name__": "__docs__"}
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            for i, (lang, body, _, end) in enumerate(blocks):
                if lang != "python":
                    continue
                expected, span = None, None
                if i + 1 < len(blocks):
                    nlang, nbody, nstart, nend = blocks[i + 1]
                    if nlang == "text" and text[end:nstart].strip() == "":
                        expected, span = nbody, (nstart, nend)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    exec(compile(body, f"{path.name}:block{i}", "exec"), namespace)
                results.append((body, expected, buf.getvalue(), span))
        finally:
            os.chdir(cwd)
    return results


def _norm(s: str | None) -> str:
    return "\n".join(line.rstrip() for line in (s or "").strip("\n").splitlines())


def check_page(path: Path) -> list[str]:
    errors = []
    for code, expected, actual, _ in run_page(path):
        if expected is not None and _norm(expected) != _norm(actual):
            first = code.strip().splitlines()[0]
            errors.append(f"{path.name}: output of block starting {first!r} changed:\n"
                          f"--- expected\n{_norm(expected)}\n--- actual\n{_norm(actual)}")
        elif expected is None and actual.strip():
            first = code.strip().splitlines()[0]
            errors.append(f"{path.name}: block starting {first!r} prints output but has no text block")
    return errors


def update_page(path: Path) -> None:
    text = path.read_text()
    out = []
    pos = 0
    for code, expected, actual, span in run_page(path):
        if span is None:
            continue
        out.append(text[pos : span[0]])
        out.append("```text\n" + _norm(actual) + "\n```\n")
        pos = span[1]
    out.append(text[pos:])
    path.write_text("".join(out))


def main(argv: list[str]) -> int:
    pages = sorted(TUTORIALS.glob("*.md"))
    if "--update" in argv:
        for p in pages:
            update_page(p)
            print(f"updated {p.name}")
        return 0
    errors = [e for p in pages for e in check_page(p)]
    for e in errors:
        print(e, end="\n\n")
    print(f"{len(pages)} pages, {len(errors)} problems")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
