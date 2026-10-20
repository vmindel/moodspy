# Command line

## `moodspy scan`

```text
usage: moodspy scan [-h] -m MOTIFS (-s SEQUENCES | -r REGIONS) [-g GENOME]
                    [-p P_CUTOFF] [--bg BG] [--out {positions,matches,counts,scores}]
                    [-w WINDOW] [--pseudocount PSEUDOCOUNT] [--motifmatchr-compat]
                    [-j JOBS] [-o OUTPUT]
```

| option | meaning |
|---|---|
| `-m, --motifs` | motif file (JASPAR, MEME, HOMER, TRANSFAC, PFM) |
| `-s, --sequences` | FASTA file of sequences to scan |
| `-r, --regions` | BED file of regions (requires `--genome`) |
| `-g, --genome` | genome FASTA |
| `-p, --p-cutoff` | p-value cutoff (default `5e-5`) |
| `--bg` | `subject`, `genome`, `even` or `A,C,G,T` frequencies |
| `--out` | `positions` (one row per hit, default) or a `matches` / `counts` / `scores` table (sequences × motifs) |
| `-w, --window` | lookup window size (affects speed, not the hits) |
| `--pseudocount` | pseudocount for count matrices (default 0.8) |
| `--motifmatchr-compat` | reproduce motifmatchr scores exactly |
| `-j, --jobs` | threads (default: all cores) |
| `-o, --output` | output TSV (default: stdout) |

```bash
moodspy scan -m motifs.jaspar -r peaks.bed -g hg38.fa -o hits.tsv
moodspy scan -m motifs.meme -s seqs.fa --out counts --bg 0.25,0.25,0.25,0.25 -o counts.tsv
```

## `moodspy-install-skills`

This installs the bundled Claude Code skill. See [Claude Code skill](../claude-skill.md).

```text
usage: moodspy-install-skills [-h] [--dest DEST] [--force] [--print-path]
```
