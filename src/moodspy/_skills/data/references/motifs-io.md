# Motifs and motif files (moodspy.motifs, moodspy.io)

## Motif

```python
m = mp.Motif(matrix, name="KLF5", id="MA0599.1", kind="auto", nsites=None, log_base=2.0, background=None)
m.length, m.consensus, m.label            # label "MA0599.1_KLF5"
m.log_odds(background=None, pseudocount=0.8, log_base=2.0, motifmatchr_compat=False)   # (4, L)
m.probabilities(pseudocount=0.8)          # (4, L), columns sum to 1
m.reverse_complement(), m.copy()
mp.Motif.from_counts / from_probabilities(..., nsites=) / from_log_odds(..., log_base=, background=) / from_biopython
mp.example_motifs()                       # KLF5, RELA, STAT1 (motifmatchr's example_motifs)
```

`kind="auto"` infers the matrix type like motifmatchr's `pwmType`:

- any negative value means `log-odds` (base `log_base`, default 2)
- non-negative columns that all sum to 1 (±1e-3) mean `probability`
- anything else non-negative means `counts`

**Pass `kind` explicitly when the guess can be wrong.** A log-odds matrix
with no negative entries is read as counts. Natural-log scores need
`log_base=np.e`.

## Pseudocounts

- `counts`: `(n + 0.25 * pc) / (N + pc)` per column, which is TFBSTools'
  `toPWM` with `pc = 0.8`. Columns may have different totals.
- `probability`: scaled to counts with `nsites` first. **When `nsites` is
  unknown it defaults to 20** (the MEME default), so pc=0.8 is a large
  prior for low-information motifs. HOMER files have no nsites. Pass
  `Motif.from_probabilities(p, nsites=...)` or `pseudocount=` to change
  this.
- `log-odds`: no pseudocount. It is only re-centred onto the scan
  background and rescaled to log2.

## Labels

Result columns use `m.label`. That is `id_name` when both exist and differ,
otherwise whichever exists, otherwise `motif_<i>`.

- Duplicate labels get `.1`, `.2` suffixes in the results.
- `{label: matrix}` dict keys become labels, and this overrides `id` and
  `name`. You can also assign `m.label = "..."`.
- Labels are what `res.motifs.index` and the positions `motif` column hold,
  not the Motif objects.

## Files

```python
mp.read_motifs(path, format=None)         # list[Motif]; format sniffed from content/extension
mp.write_jaspar(motifs, path)             # probabilities written as counts x nsites (or 100)
mp.write_meme(motifs, path, background=None)
mp.parse_jaspar(text) / parse_pfm / parse_meme / parse_homer / parse_hocomoco / parse_transfac
```

| format | id / name taken from | matrix kind |
|---|---|---|
| JASPAR (`>MA0599.1 KLF5` + `A [ ... ]` rows, brackets optional) | header words 1 / rest | counts |
| raw `.pfm` (4 numeric rows, optional `>ID name` header) | file stem as id | counts |
| MEME (`MOTIF id alt`, letter-probability matrix) | `id` / `alt`; `nsites=` kept | probability |
| HOMER (`>consensus<TAB>name<TAB>threshold`) | name (no id); threshold in `m.meta["homer_threshold"]` | probability |
| HOCOMOCO `.pcm`/`.pfm`/`.pwm` (`>ID` + one A C G T row per position) | header / none | auto; log-odds are natural log, uniform background |
| TRANSFAC (`AC`/`ID`, `NA`, `P0` table, `//`) | first AC or ID / NA | auto |

- MEME files must use the ACGT alphabet. RNA and protein files raise.
- The JASPAR "all vertebrates" download is a single `.txt` in JASPAR
  format. Sniffing handles it because its rows start with `A [`.
- pyjaspar / Biopython motifs work directly as `motifs`. Their `.counts`
  are used and `matrix_id` becomes the id.
