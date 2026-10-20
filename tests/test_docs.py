"""Tutorial code must run and print what the docs say it prints (see docs/run_docs.py)."""

import importlib.util
from pathlib import Path

import pytest

DOCS = Path(__file__).parent.parent / "docs"
spec = importlib.util.spec_from_file_location("run_docs", DOCS / "run_docs.py")
run_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_docs)

pytest.importorskip("anndata")


@pytest.mark.parametrize("page", sorted((DOCS / "tutorials").glob("*.md")), ids=lambda p: p.stem)
def test_tutorial(page):
    errors = run_docs.check_page(page)
    assert not errors, "\n\n".join(errors) + "\n\nRun `python docs/run_docs.py --update` if intended."
