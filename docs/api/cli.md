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
| `-w, --window` | lookup window size, 1-12 (affects speed, not the hits; default: automatic) |
| `--pseudocount` | pseudocount for count matrices (default 0.8) |
| `--motifmatchr-compat` | reproduce motifmatchr scores exactly |
| `-j, --jobs` | threads (default: all cores) |
| `-o, --output` | output TSV (default: stdout) |

```bash
moodspy scan -m motifs.jaspar -r peaks.bed -g hg38.fa -o hits.tsv
moodspy scan -m motifs.meme -s seqs.fa --out counts --bg 0.25,0.25,0.25,0.25 -o counts.tsv
```

## `moodspy track`

Scores the selected motifs at every position (1 bp step, no threshold), for plotting motif
affinity along peaks. See [Motif score tracks](../tutorials/peaks-and-genomes.md#motif-score-tracks).

```text
usage: moodspy track [-h] -m MOTIFS [--select SELECT] (-s SEQUENCES | -r REGIONS)
                     [-g GENOME] [--bg BG] [--strand {best,both,+,-}] [-p P_CUTOFF]
                     [--pseudocount PSEUDOCOUNT] [--motifmatchr-compat] [-j JOBS]
                     [-o OUTPUT] [--bedgraph] [--value {score,pvalue,neglog10p}]
                     [--anchor {center,start}]
```

| option | meaning |
|---|---|
| `--select` | comma-separated motif labels, IDs or names (default: all motifs) |
| `--strand` | `best` (higher-scoring strand per placement, default), `both`, `+` or `-` |
| `-p, --p-cutoff` | only fills the `hit` column (`score >=` the threshold for this p-value, as in `moodspy scan`); nothing is filtered |
| `-o, --output` | TSV (default stdout); with `--bedgraph`, a path containing `{motif}` |
| `--bedgraph` | write one bedGraph per motif instead of a TSV |
| `--value` | bedGraph value: `score`, `pvalue` or `neglog10p` |
| `--anchor` | bedGraph base of a placement: `center` of the motif window (default) or `start` |

`-m`, `-s`, `-r`, `-g`, `--bg`, `--pseudocount`, `--motifmatchr-compat` and `-j` are as for
`moodspy scan`. The TSV has columns `motif, seq, [chrom], start, end, strand, score, pvalue,
hit`.

```bash
moodspy track -m motifs.jaspar -r peaks.bed -g hg38.fa --select CTCF,KLF5 -o tracks.tsv
moodspy track -m motifs.jaspar -r peaks.bed -g hg38.fa --select CTCF --bedgraph \
    --value neglog10p -o 'tracks.{motif}.bedgraph'
```

## `moodspy-install-skills`

This installs the bundled Claude Code skill. See [Claude Code skill](../claude-skill.md).

```text
usage: moodspy-install-skills [-h] [--dest DEST] [--force] [--print-path]
```
