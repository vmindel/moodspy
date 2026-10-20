# Installation

moodspy needs Python 3.10 or newer. It depends on numpy, scipy, pandas and numba, which
all have binary wheels, so you don't need a compiler.

```bash
pip install git+https://github.com/vmindel/moodspy
```

From a clone:

```bash
git clone https://github.com/vmindel/moodspy
cd moodspy
pip install -e ".[test]"      # + pytest and anndata
pytest
```

Optional dependencies:

| package | for |
|---|---|
| `anndata` | `MotifMatches.to_anndata()` and AnnData input |
| `pysam` or `pyfaidx` | reading bgzip-compressed genomes |
| `pyjaspar` | fetching JASPAR motifs from Python |

## First call

The first call in a fresh environment compiles the scanning kernels with numba, which
takes a few seconds. The compiled code is cached on disk, so later runs start
immediately. On clusters with read-only environments, set `NUMBA_CACHE_DIR` to a
writable directory.

## Building these docs

```bash
pip install -e ".[docs]"
mkdocs serve                  # live preview at http://127.0.0.1:8000
python docs/run_docs.py       # check tutorial outputs (also run by pytest)
python docs/run_docs.py --update   # refresh them after an intentional change
```
