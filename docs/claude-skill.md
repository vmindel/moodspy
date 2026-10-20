# Claude Code skill

moodspy ships an [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) for
[Claude Code](https://claude.com/claude-code) that teaches the agent to use the library
correctly. It covers the API contracts, the defaults that silently change results, and
the common multi-step workflows.

## Install

Claude Code doesn't scan installed Python packages for skills, so copy the skill once
into your personal skills directory:

```bash
moodspy-install-skills
```

This installs it to `~/.claude/skills/moodspy/`, where Claude Code can use it in every
project from the next session on.

| command | effect |
|---|---|
| `moodspy-install-skills` | install to `~/.claude/skills/moodspy` |
| `moodspy-install-skills --dest .claude/skills/moodspy` | install into the current project only |
| `moodspy-install-skills --force` | overwrite, e.g. after upgrading moodspy |
| `moodspy-install-skills --print-path` | print the bundled copy's location |

Once it's installed, ask Claude Code for moodspy tasks in plain language. For example:
"annotate these ATAC peaks with JASPAR motifs", "port this motifmatchr script", or "find
all CTCF sites on chr1". It reads the skill automatically.

## What's inside

The skill is a short router, `SKILL.md`, that sends the agent to one topic file under
`references/`. The agent reads that file only when the task needs it, so the skill costs
almost no context until it's used.

| file | covers |
|---|---|
| `references/match-motifs.md` | inputs, `out=` modes, the result object, positions DataFrame |
| `references/genomes-regions.md` | coordinates, BED, chromosome names, FASTA readers |
| `references/single-cell-atac.md` | AnnData input, `varm` annotation, per-cell counts |
| `references/motifs-io.md` | matrix kinds, pseudocounts, labels, file formats |
| `references/thresholds-background.md` | `p_cutoff`, `bg=`, negative thresholds, compat mode |
| `references/scanner-performance.md` | `Scanner` reuse, threads, `w`, whole-genome scans |
| `references/from-motifmatchr.md` | porting R motifmatchr and MOODS-python code |
| `scripts/check_hits.py` | brute-force check of a scan on a sample of sequences |

Each file emphasizes footguns. Examples include `bg="subject"` coupling sequences scanned
together, absent sparse entries meaning "no hit" rather than 0, `observed=True` for
categorical `groupby`, and `Scanner` defaulting to a uniform background.

The test suite checks the skill:

- every routed file exists;
- every reference file is routed;
- every `mp.<name>` it mentions is a real export;
- the install command works;
- `check_hits.py` passes on the bundled test data.
