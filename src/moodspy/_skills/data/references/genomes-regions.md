# Genomes and regions (moodspy.genome)

## Genome objects

```python
g = mp.FastaFile("hg38.fa")          # mmap reader; builds/uses hg38.fa.fai
g.fetch("chr1", 1000, 1500)          # 0-based half-open -> str (case preserved)
g.lengths                            # {chrom: length}
g.base_counts()                      # genome-wide A, C, G, T counts (cached; first call reads the whole file)
```

`genome=` accepts a path, `mp.FastaFile`, `pysam.FastaFile` (anything with
`.fetch(chrom, start, end)`), `pyfaidx.Fasta`, or a `{chrom: str}` dict.
A path is opened as `mp.FastaFile`. Pass an already-open `FastaFile` to
avoid re-reading the index on every call.

- **Compressed FASTA raises.** That includes `.gz` and bgzip. Decompress the
  file, or pass a `pysam.FastaFile` / `pyfaidx.Fasta` object, which can
  read bgzip.
- Index lookup order: `<fasta>.fai` is used if it is newer than the FASTA,
  or if it is older but still consistent with the file. Otherwise the
  moodspy cache is tried (`$MOODSPY_CACHE_DIR`, default `~/.cache/moodspy`).
  As a last resort the index is rebuilt (about 0.2 s/Gb, vectorised) and
  saved next to the FASTA, or to the cache when the reference directory
  is read-only (shared `/shareDB`, iGenomes and so on). A `UserWarning` is
  raised only if neither location is writable.
- `g.base_counts()` and `bg="genome"` read the genome once (about 0.4 s/Gb)
  and then cache the counts in the same cache directory. Later processes
  get them instantly.
- FASTA lines must have equal length within a record, except the last line
  (the samtools rule). Otherwise `ValueError`.
- Record names are cut at the first whitespace, as in samtools.
- **Chromosome naming must match.** `chr1` against `1` raises `KeyError`.
  Rename the region chroms, e.g. `df.chrom = "chr" + df.chrom.astype(str)`.

## Regions

```python
mp.read_bed("peaks.bed")                  # chrom, start, end; index = BED name column if all present & unique
mp.regions_frame(df)                      # normalise any DataFrame / pyranges / list of "chr:start-end"
```

- Coordinates are 0-based half-open, so BED files can be used as they are.
  When regions come from R or IGV (1-based closed), subtract 1 from
  `start`.
- `regions_frame` maps column aliases case-insensitively:
  - chrom: `chrom`, `chr`, `chromosome`, `seqnames`, `seqname`, `contig`, `#chrom`
  - start: `start`, `chromstart`, `begin`
  - end: `end`, `chromend`, `stop`
  
  Extra columns are dropped from `res.sequences`. Join them back on the
  index if you need them.
- Region strings may contain thousands separators: `chr1:1,000-2,000`.
- A DataFrame with a non-default index keeps that index as the region
  names. A `RangeIndex` is replaced by `chrom:start-end`.
- Fixed-width peaks: build the summit windows yourself before calling
  `match_motifs`, e.g. `start = summit - 100; end = summit + 100`.
  moodspy does not resize regions.

## Background from the genome

`bg="genome"` counts every A, C, G and T in the FASTA (about 1–2 s for
hg38). The counts are cached on disk after the first call. Passing explicit
numbers is still the most reproducible option:

```python
g = mp.FastaFile("hg38.fa")
counts = g.base_counts(); bg = counts / counts.sum()
res = mp.match_motifs(motifs, peaks, genome=g, bg=bg)
```
